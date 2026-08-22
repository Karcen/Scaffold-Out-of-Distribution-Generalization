#!/usr/bin/env python3
"""Build a deterministic anonymous supplementary archive."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import zipfile


ARCHIVE = Path("supplementary_artifact.zip")
ARCHIVE_HASH = Path("artifact_manifest.sha256")
PREFIX = Path("scaffold_ood_artifact")

ROOT_FILES = {
    ".gitignore",
    "00_environment_diagnostic.md",
    "01_problem_formulation.md",
    "02_adversarial_defense.md",
    "03_methodology.md",
    "04_experimental_results.md",
    "05_figure_contracts.md",
    "ARTIFACT_NOTICE.md",
    "DATA_LICENSES.md",
    "README.md",
    "checklist.tex",
    "main.tex",
    "neurips_2026.sty",
    "pyproject.toml",
    "references.bib",
    "requirements-lock.txt",
    "requirements.txt",
}
INCLUDE_TREES = {
    "data/external",
    "figures",
    "output/pdf",
    "results",
    "scripts",
    "src",
    "tables",
    "templates",
    "tests",
}
EXCLUDED_RELATIVE = {
    "results/cmnpd_audit_predictions.csv.gz",
}
EXCLUDED_PARTS = {
    ".DS_Store",
    ".mplconfig",
    ".pytest_cache",
    "__pycache__",
    "tmp",
}
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def selected_files() -> list[Path]:
    selected: set[Path] = set()
    for name in ROOT_FILES:
        path = Path(name)
        if path.is_file():
            selected.add(path)
    for tree in INCLUDE_TREES:
        base = Path(tree)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(Path("."))
            if relative.as_posix() in EXCLUDED_RELATIVE:
                continue
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(Path(".")).as_posix())


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    files = selected_files()
    if not Path("output/pdf/manuscript.pdf").is_file():
        raise FileNotFoundError(
            "Build output/pdf/manuscript.pdf before packaging the artifact."
        )

    members: list[tuple[str, bytes]] = []
    manifest_lines: list[str] = []
    for path in files:
        relative = path.relative_to(Path("."))
        name = (PREFIX / relative).as_posix()
        data = path.read_bytes()
        members.append((name, data))
        manifest_lines.append(f"{sha256(data).hexdigest()}  {name}")

    manifest_name = (PREFIX / "MANIFEST.sha256").as_posix()
    manifest_data = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    members.append((manifest_name, manifest_data))

    with zipfile.ZipFile(
        ARCHIVE, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for name, data in members:
            handle.writestr(zip_info(name), data, compresslevel=9)

    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    ARCHIVE_HASH.write_text(
        f"{digest}  {ARCHIVE.name}\n",
        encoding="utf-8",
    )
    print(f"archive={ARCHIVE}")
    print(f"members={len(members)}")
    print(f"bytes={ARCHIVE.stat().st_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
