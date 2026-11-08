"""Independent Solubility driver for the unchanged fixed Chemprop fit source."""
from pathlib import Path
import hashlib, importlib.util, json, shutil, sys

ROOT = Path(__file__).resolve().parents[2]
FIT = Path(__file__).with_name("chemprop_solubility_fit_source.py")
OUT = ROOT / "results/review_2026_10_14/chemprop"
EXPECTED_FIT_SHA = "3f676f7135d16b472bd1581ae7b0555a21b541cf0ba0b4dd9b1fc6b140280734"
PROTOCOL_SHA = "8607185e8be70b332c209af34db67c611de97b6dea6ff057a8e17015cd723272"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(FIT) == EXPECTED_FIT_SHA, "Scientific fit implementation changed"
    assert sha(ROOT / "docs/review_2026_10_14/CHEMPROP_PROTOCOL.md") == PROTOCOL_SHA
    driver_sha = sha(__file__)
    snapshot = OUT / "code_snapshots" / f"{driver_sha}.py"
    if snapshot.exists():
        assert sha(snapshot) == driver_sha
    else:
        shutil.copyfile(__file__, snapshot)
    metadata = {"endpoint": "Solubility", "driver_sha256": driver_sha,
                "driver_source": str(Path(__file__).relative_to(ROOT)),
                "fitting_source": str(FIT.relative_to(ROOT)), "fitting_source_sha256": EXPECTED_FIT_SHA,
                "protocol_sha256": PROTOCOL_SHA, "threads": 2, "inter_op_threads": 1,
                "scientific_protocol_change": False,
                "shared_aggregate_writes": False,
                "scope": "Restrict unchanged fitting implementation to Solubility; suppress global registry/manifest updates. Primary process regenerates the final aggregate after both workers have stopped."}
    folder = OUT / "Solubility"
    folder.mkdir(exist_ok=True)
    launch = folder / "parallel_driver.json"
    if launch.exists():
        assert json.loads(launch.read_text()) == metadata
    else:
        launch.write_text(json.dumps(metadata, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("fixed_solubility_fits", FIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT and module.CODE_HASH == EXPECTED_FIT_SHA
    module.aggregate = lambda: None
    sys.argv = [str(FIT), "--tasks", "Solubility"]
    module.main()
    print("SOLUBILITY ENDPOINT WORKER COMPLETE; NO SHARED AGGREGATE WRITTEN", flush=True)


if __name__ == "__main__":
    main()
