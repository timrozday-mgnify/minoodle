"""Simulated-dataset adapter (§5 M0).

Wraps genome-blender so that every dataset is regenerable from a manifest: seeds, error-model
parameters, fragment distribution, reference genomes and output checksums all recorded in one
JSON file next to the reads.

    python -m minoodle.simdata run datasets/L0.yaml --out ~/Documents/minoodle_run/L0
    python -m minoodle.simdata verify ~/Documents/minoodle_run/L0/manifest.json

Terminology (§2.6): fragment length is the **outer** distance, 5' of mate 1 to 5' of mate 2.
Inner distance is never stored. genome-blender's `fragment_mean` is the outer distance, so it
is recorded here as `fragment_length_mean`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class ErrorModelPhase(StrEnum):
    """Error-model provenance (§2.5). Every results table must state its phase.

    P1/P2 numbers are circular by construction and must never leave the repo (§6.8); only P3
    (skiver trained on real reads) is externally quotable.
    """

    P1 = "P1"  # same model genome-blender simulated with — fully circular
    P2 = "P2"  # simpler skiver model fitted to the synthetic data — bounded misspecification
    P3 = "P3"  # skiver trained on real reads — the only externally quotable phase


@dataclass
class FileRecord:
    """sha256 of a file's *content*, plus its on-disk size.

    For `.gz` files the hash is of the decompressed stream: gzip embeds an mtime in its
    header, so two byte-identical regenerations of the same FASTQ hash differently as
    containers. Reproducibility is a claim about the reads, not about the compressor.
    """

    path: str
    sha256: str
    bytes: int

    @classmethod
    def of(cls, path: Path, relative_to: Path | None = None) -> FileRecord:
        opener = gzip.open if path.suffix == ".gz" else open
        digest = hashlib.sha256()
        with opener(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        shown = path.relative_to(relative_to) if relative_to else path
        return cls(str(shown), digest.hexdigest(), path.stat().st_size)


@dataclass
class Manifest:
    name: str
    created_utc: str
    error_model_phase: str
    minoodle_sha: str | None
    genome_blender_sha: str | None
    generate_reads_cmd: list[str]
    genome_blender_config: dict[str, Any]
    fragment_length_mean: float  # outer distance (§2.6)
    fragment_length_variance: float
    read_length_mean: float
    references: list[FileRecord]
    outputs: list[FileRecord]
    skiver_model: FileRecord | None = None
    notes: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        raw = json.loads(path.read_text())
        raw["references"] = [FileRecord(**r) for r in raw["references"]]
        raw["outputs"] = [FileRecord(**r) for r in raw["outputs"]]
        if raw.get("skiver_model"):
            raw["skiver_model"] = FileRecord(**raw["skiver_model"])
        return cls(**raw)


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def verify(manifest_path: Path) -> list[str]:
    """Re-hash every file the manifest records. Returns a list of problems (empty == good)."""
    manifest = Manifest.read(manifest_path)
    base = manifest_path.parent
    problems: list[str] = []
    records = list(manifest.outputs) + list(manifest.references)
    if manifest.skiver_model:
        records.append(manifest.skiver_model)
    for rec in records:
        path = Path(rec.path)
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            problems.append(f"missing: {rec.path}")
            continue
        actual = FileRecord.of(path)
        if actual.sha256 != rec.sha256:
            problems.append(f"sha256 mismatch: {rec.path} ({rec.sha256} -> {actual.sha256})")
    return problems


def run(dataset_config: Path, out_dir: Path) -> Path:
    """Generate a dataset into `out_dir` and write `manifest.json`. Returns the manifest path."""
    spec = yaml.safe_load(dataset_config.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # genomes.csv with absolute fasta paths, resolved against the dataset config's directory.
    refs: list[Path] = []
    csv_path = out_dir / "genomes.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["genome_id", "fasta_path", "abundance"])
        for g in spec["genomes"]:
            fasta = (dataset_config.parent / g["fasta_path"]).resolve()
            if not fasta.exists():
                raise FileNotFoundError(f"reference genome not found: {fasta}")
            refs.append(fasta)
            writer.writerow([g["genome_id"], str(fasta), g["abundance"]])

    gb_config: dict[str, Any] = dict(spec["genome_blender"])
    gb_config["input_csv"] = "genomes.csv"
    gb_config.setdefault("output_prefix", "sim_reads")
    gb_config_path = out_dir / "genome_blender_config.yaml"
    gb_config_path.write_text(yaml.safe_dump(gb_config, sort_keys=True))

    cmd = shlex.split(spec["generate_reads_cmd"]) + ["--config", gb_config_path.name]
    subprocess.run(cmd, cwd=out_dir, check=True)

    prefix = gb_config["output_prefix"]
    stems = [f"{prefix}_R1", f"{prefix}_R2"] if gb_config.get("paired_end", False) else [prefix]
    outputs = []
    for stem in stems:
        # genome-blender gzips its FASTQ in current versions but not older ones.
        matches = [p for ext in (".fastq.gz", ".fastq") if (p := out_dir / f"{stem}{ext}").exists()]
        if not matches:
            raise FileNotFoundError(f"genome-blender did not produce {out_dir / stem}.fastq[.gz]")
        outputs.append(FileRecord.of(matches[0], relative_to=out_dir))
    bam = out_dir / f"{prefix}.bam"
    if not bam.exists():
        raise FileNotFoundError(f"genome-blender did not produce {bam}")
    outputs.append(FileRecord.of(bam, relative_to=out_dir))

    skiver_model = None
    if gb_config.get("skiver_model"):
        skiver_model = FileRecord.of(Path(gb_config["skiver_model"]).resolve())

    manifest = Manifest(
        name=spec["name"],
        created_utc=datetime.now(UTC).isoformat(),
        error_model_phase=ErrorModelPhase(spec["error_model_phase"]).value,
        minoodle_sha=_git_sha(REPO_ROOT),
        genome_blender_sha=_git_sha(Path(shlex.split(spec["generate_reads_cmd"])[-1]).parent),
        generate_reads_cmd=cmd,
        genome_blender_config=gb_config,
        fragment_length_mean=float(gb_config["fragment_mean"]),
        fragment_length_variance=float(gb_config["fragment_variance"]),
        read_length_mean=float(gb_config["read_length_mean"]),
        references=[FileRecord.of(p) for p in refs],
        outputs=outputs,
        skiver_model=skiver_model,
        notes=list(spec.get("notes", [])),
    )
    manifest_path = out_dir / "manifest.json"
    manifest.write(manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.simdata", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="generate a dataset and write its manifest")
    p_run.add_argument("dataset_config", type=Path)
    p_run.add_argument("--out", type=Path, required=True)

    p_verify = sub.add_parser("verify", help="re-hash the files a manifest records")
    p_verify.add_argument("manifest", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        print(run(args.dataset_config, args.out.expanduser()))
        return 0

    problems = verify(args.manifest.expanduser())
    for p in problems:
        print(p, file=sys.stderr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
