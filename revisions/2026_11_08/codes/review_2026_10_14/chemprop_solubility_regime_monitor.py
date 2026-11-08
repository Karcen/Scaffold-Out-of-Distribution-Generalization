"""Resume the paused scaffold worker only after both other regimes complete."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, subprocess, time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_2026_10_14/chemprop"
RECORD = OUT.parent / "chemprop_solubility_regime_boundary.json"


def counts():
    records = [json.loads(p.read_text()) for p in (OUT / "Solubility").glob("runs/*/summary.json")]
    return {regime: sum(r["status"] == "complete" and r["regime"] == regime for r in records)
            for regime in ("scaffold", "descriptor", "morgan")}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scaffold-pid", type=int, required=True)
    args = p.parse_args()
    command = subprocess.check_output(["ps", "-p", str(args.scaffold_pid), "-o", "command="], text=True).strip()
    assert command.endswith("-u codes/review_2026_10_14/chemprop_solubility_worker.py"), command
    record = {"scaffold_worker_pid": args.scaffold_pid, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pause_confirmed_utc": utc(), "counts_at_monitor_launch": counts(), "resume_utc": None,
              "scope": "Scaffold worker was SIGSTOP-paused during its fifth fit. Resume only after all five descriptor and all five Morgan fits are complete; paused wallclock time is included in that scaffold fit's seconds."}
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    while time.monotonic() - started < 7200:
        c = counts()
        if c["descriptor"] == c["morgan"] == 5:
            os.kill(args.scaffold_pid, signal.SIGCONT)
            record["resume_utc"] = utc()
            record["counts_at_resume"] = c
            record["status"] = "boundary_safe"
            RECORD.write_text(json.dumps(record, indent=2) + "\n")
            print("SCAFFOLD WORKER RESUMED after both other regimes complete", flush=True)
            return
        time.sleep(2)
    record["status"] = "monitor_timeout_scaffold_worker_remains_paused"
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    raise RuntimeError(record["status"])


if __name__ == "__main__":
    main()
