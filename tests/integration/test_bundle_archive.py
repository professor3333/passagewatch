"""Published bundle archives: reproducible, and accepted only as the manifest's release."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from passagewatch.service.bundle import build_bundle
from passagewatch.service.bundle_archive import (
    BundleMismatchError,
    pack_bundle,
    sha256_file,
    unpack_bundle,
    verify_bundle,
)
from passagewatch.service.release import ReleaseManifest

from .test_release import manifest_for
from .test_worker import checkpoint

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return build_bundle(
        checkpoint=checkpoint(tmp_path / "model.pt"),
        tracker_config={},
        score_threshold=0.3,
        version="pw-a",
        bundles_dir=tmp_path / "bundles",
    )


def test_packing_is_reproducible_and_round_trips(bundle: Path, tmp_path: Path) -> None:
    manifest = manifest_for(bundle)
    first = pack_bundle(bundle, manifest, tmp_path / "out1", REPO)
    second = pack_bundle(bundle, manifest, tmp_path / "out2", REPO)

    assert first.name == "pw-a-bundle.tar.gz"
    assert sha256_file(first) == sha256_file(second)
    with tarfile.open(first) as tar:
        assert tar.getnames() == [
            "pw-a/LICENSE",
            "pw-a/THIRD_PARTY_NOTICES.md",
            "pw-a/bundle.json",
            "pw-a/detector.pt",
        ]
    unpacked = unpack_bundle(first, manifest, tmp_path / "fresh")
    assert unpacked == tmp_path / "fresh" / "pw-a"
    assert sha256_file(unpacked / "detector.pt") == sha256_file(bundle / "detector.pt")
    assert unpack_bundle(first, manifest, tmp_path / "fresh") == unpacked  # already there


def test_a_bundle_that_is_not_the_manifests_release_is_refused(
    bundle: Path, tmp_path: Path
) -> None:
    manifest = manifest_for(bundle)
    other = manifest.model_copy(
        update={"bundle": manifest.bundle | {"pipeline_config_sha256": "f" * 64}}
    )
    with pytest.raises(BundleMismatchError, match="pipeline_config_sha256"):
        verify_bundle(bundle, other)

    (bundle / "detector.pt").write_bytes(b"other weights")
    with pytest.raises(BundleMismatchError, match=r"detector\.pt: SHA-256"):
        verify_bundle(bundle, manifest)


def test_a_tampered_download_is_not_installed(bundle: Path, tmp_path: Path) -> None:
    manifest = manifest_for(bundle)
    good = pack_bundle(bundle, manifest, tmp_path / "out", REPO)
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(good) as src, tarfile.open(tampered, "w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read()  # type: ignore[union-attr]
            if member.name.endswith("detector.pt"):
                data = data[:-1] + bytes([data[-1] ^ 1])
            dst.addfile(member, io.BytesIO(data))

    with pytest.raises(BundleMismatchError, match="SHA-256"):
        unpack_bundle(tampered, manifest, tmp_path / "fresh")
    assert not (tmp_path / "fresh" / "pw-a").exists()


@pytest.mark.parametrize("name", ["../evil", "pw-a/../evil", "/etc/evil", "pw-a/sub/x"])
def test_unexpected_archive_paths_are_refused(name: str, tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    manifest = ReleaseManifest.model_validate(
        json.loads((REPO / "releases/manifests/passagewatch-0.3.0.json").read_text())
    ).model_copy(update={"release": "pw-a"})

    with pytest.raises(BundleMismatchError, match="unexpected archive member"):
        unpack_bundle(archive, manifest, tmp_path / "fresh")
    assert not (tmp_path / "evil").exists()
