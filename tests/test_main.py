from __future__ import annotations

from pyantique_prices.__main__ import _should_use_object_workflow


def test_should_use_object_workflow_for_directory_with_3_to_5_images(tmp_path):
    for idx in range(3):
        (tmp_path / f"img{idx}.jpg").write_bytes(b"x")

    assert _should_use_object_workflow(tmp_path, sorted(tmp_path.iterdir())) is True


def test_should_not_use_object_workflow_for_single_file(tmp_path):
    image = tmp_path / "single.jpg"
    image.write_bytes(b"x")

    assert _should_use_object_workflow(image, [image]) is False


def test_should_not_use_object_workflow_for_large_directory_batch(tmp_path):
    for idx in range(6):
        (tmp_path / f"img{idx}.jpg").write_bytes(b"x")

    assert _should_use_object_workflow(tmp_path, sorted(tmp_path.iterdir())) is False
