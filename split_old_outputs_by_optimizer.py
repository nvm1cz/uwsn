from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


SOURCE_DIR = Path("outputs/data/csv/output")
TARGET_DIR = SOURCE_DIR / "by_optimizer"
INPUT_DIR = Path("data/csv/input")
INPUT_TARGET_DIR = INPUT_DIR / "by_optimizer"
REFERENCE_FILE = SOURCE_DIR / "experiment_outputs.csv"
OPTIMIZER_RE = re.compile(r"_(pso|ga|de|leach|eeumc|ebrec|eulc)_", re.IGNORECASE)


def _safe_optimizer_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _load_input_optimizer_map() -> dict[str, str]:
    ref = pd.read_csv(REFERENCE_FILE, usecols=["input_id", "optimizer"], dtype=str)
    ref = ref.dropna(subset=["input_id", "optimizer"])
    ref["optimizer"] = ref["optimizer"].map(_safe_optimizer_name)
    return dict(zip(ref["input_id"], ref["optimizer"]))


def _optimizer_from_input_id(input_id: object) -> str | None:
    if input_id is None:
        return None
    match = OPTIMIZER_RE.search(str(input_id))
    if not match:
        return None
    return _safe_optimizer_name(match.group(1))


def _split_one_csv(path: Path, input_optimizer: dict[str, str]) -> list[dict[str, object]]:
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return []

    has_optimizer_column = "optimizer" in df.columns
    if has_optimizer_column:
        df["_split_optimizer"] = df["optimizer"].map(_safe_optimizer_name)
        output_df = df.drop(columns=["optimizer"])
    elif "input_id" in df.columns:
        df["_split_optimizer"] = df["input_id"].map(input_optimizer)
        parsed = df["input_id"].map(_optimizer_from_input_id)
        df["_split_optimizer"] = df["_split_optimizer"].fillna(parsed)
        missing = int(df["_split_optimizer"].isna().sum())
        if missing:
            print(f"Warning: {path.name} has {missing} rows without optimizer mapping; skipped.")
        output_df = df
    else:
        print(f"Skip {path.name}: no optimizer or input_id column.")
        return []

    rows: list[dict[str, object]] = []
    for optimizer, group in df.groupby("_split_optimizer", dropna=True):
        optimizer = _safe_optimizer_name(optimizer)
        out_dir = TARGET_DIR / optimizer
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{optimizer}.csv"
        if has_optimizer_column:
            clean = group.drop(columns=["optimizer", "_split_optimizer"])
        else:
            clean = group.drop(columns=["_split_optimizer"])
        clean.to_csv(out_path, index=False)
        rows.append(
            {
                "source_file": path.name,
                "optimizer": optimizer,
                "rows": len(clean),
                "output_file": str(out_path.as_posix()),
            }
        )
        print(f"Saved {out_path} ({len(clean)} rows)")
    return rows


def _split_input_with_optimizer(path: Path) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    df = pd.read_csv(path, dtype=str)
    if df.empty or "optimizer" not in df.columns:
        return [], {}

    rows: list[dict[str, object]] = []
    topology_ids_by_optimizer: dict[str, set[str]] = {}
    for optimizer, group in df.groupby(df["optimizer"].map(_safe_optimizer_name), dropna=True):
        optimizer = _safe_optimizer_name(optimizer)
        out_dir = INPUT_TARGET_DIR / optimizer
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{optimizer}.csv"
        clean = group.drop(columns=["optimizer"])
        clean.to_csv(out_path, index=False)

        if "topology_id" in clean.columns:
            topology_ids_by_optimizer[optimizer] = set(clean["topology_id"].dropna())

        rows.append(
            {
                "source_file": path.name,
                "optimizer": optimizer,
                "rows": len(clean),
                "output_file": str(out_path.as_posix()),
            }
        )
        print(f"Saved {out_path} ({len(clean)} rows)")
    return rows, topology_ids_by_optimizer


def _split_input_by_input_id(path: Path, input_optimizer: dict[str, str]) -> list[dict[str, object]]:
    df = pd.read_csv(path, dtype=str)
    if df.empty or "input_id" not in df.columns:
        return []

    df["_split_optimizer"] = df["input_id"].map(input_optimizer)
    parsed = df["input_id"].map(_optimizer_from_input_id)
    df["_split_optimizer"] = df["_split_optimizer"].fillna(parsed)

    rows: list[dict[str, object]] = []
    for optimizer, group in df.groupby("_split_optimizer", dropna=True):
        optimizer = _safe_optimizer_name(optimizer)
        out_dir = INPUT_TARGET_DIR / optimizer
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{optimizer}.csv"
        clean = group.drop(columns=["optimizer", "_split_optimizer"], errors="ignore")
        clean.to_csv(out_path, index=False)
        rows.append(
            {
                "source_file": path.name,
                "optimizer": optimizer,
                "rows": len(clean),
                "output_file": str(out_path.as_posix()),
            }
        )
        print(f"Saved {out_path} ({len(clean)} rows)")
    return rows


def _split_topology_file(path: Path, topology_ids_by_optimizer: dict[str, set[str]]) -> list[dict[str, object]]:
    df = pd.read_csv(path, dtype=str)
    if df.empty or "topology_id" not in df.columns:
        return []

    rows: list[dict[str, object]] = []
    for optimizer, topology_ids in topology_ids_by_optimizer.items():
        clean = df[df["topology_id"].isin(topology_ids)]
        if clean.empty:
            continue
        out_dir = INPUT_TARGET_DIR / optimizer
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{optimizer}.csv"
        clean.to_csv(out_path, index=False)
        rows.append(
            {
                "source_file": path.name,
                "optimizer": optimizer,
                "rows": len(clean),
                "output_file": str(out_path.as_posix()),
            }
        )
        print(f"Saved {out_path} ({len(clean)} rows)")
    return rows


def _split_inputs(input_optimizer: dict[str, str]) -> None:
    INPUT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    input_path = INPUT_DIR / "experiment_inputs.csv"
    input_rows, topology_ids_by_optimizer = _split_input_with_optimizer(input_path)
    manifest_rows.extend(input_rows)

    detail_path = INPUT_DIR / "representative_detail_inputs.csv"
    if detail_path.exists():
        manifest_rows.extend(_split_input_by_input_id(detail_path, input_optimizer))

    for filename in ("experiment_topologies.csv", "experiment_nodes.csv"):
        path = INPUT_DIR / filename
        if path.exists():
            manifest_rows.extend(_split_topology_file(path, topology_ids_by_optimizer))

    manifest_path = INPUT_DIR / "experiment_manifest.csv"
    if manifest_path.exists():
        out_path = INPUT_TARGET_DIR / manifest_path.name
        pd.read_csv(manifest_path, dtype=str).to_csv(out_path, index=False)
        manifest_rows.append(
            {
                "source_file": manifest_path.name,
                "optimizer": "all",
                "rows": len(pd.read_csv(manifest_path, dtype=str)),
                "output_file": str(out_path.as_posix()),
            }
        )
        print(f"Saved {out_path}")

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = INPUT_TARGET_DIR / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"Saved {manifest_path}")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    input_optimizer = _load_input_optimizer_map()
    manifest_rows: list[dict[str, object]] = []
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        manifest_rows.extend(_split_one_csv(path, input_optimizer))

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = TARGET_DIR / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"Saved {manifest_path}")

    _split_inputs(input_optimizer)


if __name__ == "__main__":
    main()
