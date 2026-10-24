from segment.data import check_splits, load_chip, read_manifest


def test_fixture_builds_clean(fixture_manifest):
    records = read_manifest(fixture_manifest)
    assert len(records) == 8
    assert check_splits(records) == []
    image, mask = load_chip(fixture_manifest.parent / records[0].image_path_rel)
    assert image.shape == (5, 32, 32) and mask.shape == (32, 32)
