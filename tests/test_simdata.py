import gzip

import pytest

from minoodle.simdata import ErrorModelPhase, FileRecord, Manifest, verify


def _manifest(out_dir, ref, r1):
    return Manifest(
        name="T",
        created_utc="2026-01-01T00:00:00+00:00",
        error_model_phase=ErrorModelPhase.P1.value,
        minoodle_sha=None,
        genome_blender_sha=None,
        generate_reads_cmd=["true"],
        genome_blender_config={"seed": 42},
        fragment_length_mean=400.0,
        fragment_length_variance=10000.0,
        read_length_mean=150.0,
        references=[FileRecord.of(ref)],
        outputs=[FileRecord.of(r1, relative_to=out_dir)],
    )


def test_manifest_round_trip_and_verify(tmp_path):
    ref = tmp_path / "ref.fna"
    ref.write_text(">a\nACGT\n")
    r1 = tmp_path / "sim_reads_R1.fastq"
    r1.write_text("@r\nACGT\n+\nIIII\n")

    path = tmp_path / "manifest.json"
    _manifest(tmp_path, ref, r1).write(path)

    reread = Manifest.read(path)
    assert reread.error_model_phase == "P1"
    assert reread.fragment_length_mean == 400.0
    assert reread.outputs[0].path == "sim_reads_R1.fastq"  # relative to the manifest

    assert verify(path) == []


def test_verify_detects_a_changed_output(tmp_path):
    ref = tmp_path / "ref.fna"
    ref.write_text(">a\nACGT\n")
    r1 = tmp_path / "sim_reads_R1.fastq"
    r1.write_text("@r\nACGT\n+\nIIII\n")
    path = tmp_path / "manifest.json"
    _manifest(tmp_path, ref, r1).write(path)

    r1.write_text("@r\nTTTT\n+\nIIII\n")
    problems = verify(path)
    assert len(problems) == 1
    assert "sha256 mismatch" in problems[0]

    r1.unlink()
    assert verify(path) == ["missing: sim_reads_R1.fastq"]


def test_gz_hashes_content_not_container(tmp_path):
    """gzip embeds an mtime, so identical reads compress to different bytes (§M0 gate)."""
    payload = b"@r\nACGT\n+\nIIII\n"
    a, b = tmp_path / "a.fastq.gz", tmp_path / "b.fastq.gz"
    for i, p in enumerate((a, b)):
        with gzip.GzipFile(filename="", mode="wb", fileobj=p.open("wb"), mtime=i * 1000) as fh:
            fh.write(payload)

    assert a.read_bytes() != b.read_bytes()
    assert FileRecord.of(a).sha256 == FileRecord.of(b).sha256


def test_unknown_phase_rejected():
    with pytest.raises(ValueError):
        ErrorModelPhase("P4")
