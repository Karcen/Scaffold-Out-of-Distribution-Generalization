"""Prevent the primary worker reaching unfinished parallel Solubility fits."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, subprocess, time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_2026_10_14/chemprop"
RECORD = OUT.parent / "chemprop_parallel_boundary.json"


def counts(endpoint):
    records = [json.loads(p.read_text()) for p in (OUT / endpoint).glob("runs/*/summary.json")]
    return sum(r["status"] == "complete" for r in records)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary-pid", type=int, required=True)
    args = p.parse_args()
    command = subprocess.check_output(["ps", "-p", str(args.primary_pid), "-o", "command="], text=True).strip()
    assert command.endswith("-u codes/review_2026_10_14/chemprop_baseline.py"), command
    record = {"primary_pid": args.primary_pid, "primary_command": command,
              "monitor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_utc": utc(), "pause": None, "resume": None,
              "scope": "OS pause preserves the primary fitting state; only wallclock timings can include the pause. No overlapping endpoint/run writes are allowed."}
    started = time.monotonic()
    while time.monotonic() - started < 7200:
        cyp, sol = counts("CYP2D6"), counts("Solubility")
        if sol == 15:
            record["solubility_completed_utc"] = utc()
            if record["pause"]:
                os.kill(args.primary_pid, signal.SIGCONT)
                record["resume"] = utc()
                print("PRIMARY RESUMED: all 15 Solubility fits complete", flush=True)
            record["status"] = "boundary_safe"
            RECORD.write_text(json.dumps(record, indent=2) + "\n")
            return
        if cyp >= 14 and not record["pause"]:
            assert cyp < 15, "Boundary monitor was started too late"
            os.kill(args.primary_pid, signal.SIGSTOP)
            record["pause"] = {"utc": utc(), "cyp_complete": cyp, "solubility_complete": sol}
            RECORD.write_text(json.dumps(record, indent=2) + "\n")
            print("PRIMARY PAUSED before Solubility boundary", json.dumps(record["pause"]), flush=True)
        time.sleep(2)
    record["status"] = "monitor_timeout_primary_remains_paused_if_pause_recorded"
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    raise RuntimeError(record["status"])


if __name__ == "__main__":
    main()
