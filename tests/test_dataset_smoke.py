import importlib


def test_dataset_smoke():
    mod = importlib.import_module("vctp.data.loader")
    ds = mod.build_dataset({}, "val")
    it = iter(ds)
    first = next(it)
    assert "image_path" in first and "question" in first
