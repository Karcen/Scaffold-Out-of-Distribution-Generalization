"""Five fixed Solubility fits for one regime; no shared aggregation."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil, sys

ROOT = Path(__file__).resolve().parents[2]
FIT = Path(__file__).with_name("chemprop_solubility_fit_source.py")
OUT = ROOT / "results/review_2026_10_14/chemprop"
EXPECTED_FIT_SHA = "3f676f7135d16b472bd1581ae7b0555a21b541cf0ba0b4dd9b1fc6b140280734"
PROTOCOL_SHA = "8607185e8be70b332c209af34db67c611de97b6dea6ff057a8e17015cd723272"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("descriptor", "morgan"), required=True)
    args = parser.parse_args()
    assert sha(FIT) == EXPECTED_FIT_SHA
    assert sha(ROOT / "docs/review_2026_10_14/CHEMPROP_PROTOCOL.md") == PROTOCOL_SHA
    driver_sha = sha(__file__)
    snapshot = OUT / "code_snapshots" / f"{driver_sha}.py"
    if snapshot.exists():
        assert sha(snapshot) == driver_sha
    else:
        shutil.copyfile(__file__, snapshot)
    spec = importlib.util.spec_from_file_location("fixed_solubility_regime_fits", FIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT and module.CODE_HASH == EXPECTED_FIT_SHA
    module.aggregate = lambda: None
    scientific_run, prepare = module.run, module.prepare_endpoint

    def selected_run(spec, frame, whole, graph_meta, regime, seed, environment_hash):
        if regime == args.regime:
            scientific_run(spec, frame, whole, graph_meta, regime, seed, environment_hash)

    def recorded_prepare(endpoint):
        result = prepare(endpoint)
        print("GRAPH METADATA PREPARATION COMPLETE", args.regime, flush=True)
        return result

    module.run, module.prepare_endpoint = selected_run, recorded_prepare
    metadata = {"endpoint": "Solubility", "regime": args.regime, "expected_fits": 5,
                "driver_sha256": driver_sha, "fitting_source_sha256": EXPECTED_FIT_SHA,
                "protocol_sha256": PROTOCOL_SHA, "threads": 2, "scientific_protocol_change": False,
                "shared_aggregate_writes": False,
                "scope": "Route only this regime to the byte-identical fitting implementation. Original Solubility worker is paused at its fifth scaffold fit until both regime workers complete."}
    launch = OUT / "Solubility" / f"parallel_{args.regime}_driver.json"
    if launch.exists():
        assert json.loads(launch.read_text()) == metadata
    else:
        launch.write_text(json.dumps(metadata, indent=2) + "\n")
    sys.argv = [str(FIT), "--tasks", "Solubility"]
    module.main()
    print("SOLUBILITY REGIME WORKER COMPLETE", args.regime, flush=True)


if __name__ == "__main__":
    main()
