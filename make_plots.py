#!/usr/bin/env python3
# make_plots.py
# usage:
#   python3 make_plots.py --roots dell_output orin_output thor_output --out analysis_out

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_run_dirs(roots):
    run_dirs = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for meta_path in root.rglob("run_meta.json"):
            run_dir = meta_path.parent
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def pick(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_one(run_dir: Path):
    meta = read_json(run_dir / "run_meta.json")
    summary = read_json(run_dir / "summary.json")

    # infer device label (prefer meta.host; fallback to root folder name)
    device = pick(meta, ["host", "machine", "node", "device_name"], None)
    if device is None:
        # use the first folder under roots as device-ish label
        # e.g. dell_output/... -> "dell_output"
        parts = run_dir.parts
        device = parts[-len(run_dir.parts)] if len(parts) > 0 else "unknown"
        # better fallback:
        device = next((p for p in parts if p.endswith("_output")), "unknown")

    model_key = pick(meta, ["model_key"], None) or run_dir.parts[-2]
    model_id = pick(meta, ["model_id", "model", "hf_model_id"], None)
    dtype = pick(meta, ["dtype"], None)

    # common workload knobs
    num_images = pick(meta, ["num_images"], pick(summary, ["num_images"], None))
    num_tasks = pick(summary, ["num_tasks"], pick(meta, ["num_tasks"], None))
    batch_size = pick(meta, ["batch_size"], pick(summary, ["batch_size"], None))
    repeats = pick(meta, ["repeats"], pick(summary, ["repeats"], None))
    suite = pick(meta, ["suite"], None)
    run_group = pick(meta, ["run_group"], run_dir.name)

    # latency + throughput (your summaries use these names based on the examples you uploaded)
    latency_mean_ms = pick(summary, ["latency_ms_mean", "latency_mean_ms"], None)
    latency_p50_ms = pick(summary, ["latency_ms_p50", "latency_p50_ms"], None)
    latency_p90_ms = pick(summary, ["latency_ms_p90", "latency_p90_ms"], None)
    latency_min_ms = pick(summary, ["latency_ms_min", "latency_min_ms"], None)
    latency_max_ms = pick(summary, ["latency_ms_max", "latency_max_ms"], None)

    ips_mean = pick(summary, ["images_per_second_mean", "ips_mean", "images_per_sec_mean"], None)
    ips_p50 = pick(summary, ["images_per_second_p50", "ips_p50"], None)
    ips_p90 = pick(summary, ["images_per_second_p90", "ips_p90"], None)

    total_elapsed_sec = pick(summary, ["total_elapsed_seconds", "elapsed_seconds"], None)
    num_errors = pick(summary, ["num_errors", "errors"], 0)

    # power (sometimes only available on some devices)
    power_available = bool(pick(meta, ["power_monitoring_available"], pick(summary, ["power_monitoring_available"], False)))
    power_method = pick(meta, ["power_monitoring_method"], pick(summary, ["power_monitoring_method"], None))

    # if your summary includes avg_power_watts at some point, capture it; otherwise None
    avg_power_watts = pick(summary, ["avg_power_watts", "power_avg_watts", "avg_total_power_watts"], None)

    return {
        "run_dir": str(run_dir),
        "device": device,
        "model_key": model_key,
        "model_id": model_id,
        "dtype": dtype,
        "suite": suite,
        "run_group": run_group,
        "num_images": num_images,
        "num_tasks": num_tasks,
        "batch_size": batch_size,
        "repeats": repeats,
        "num_errors": num_errors,
        "total_elapsed_sec": total_elapsed_sec,
        "latency_mean_ms": latency_mean_ms,
        "latency_p50_ms": latency_p50_ms,
        "latency_p90_ms": latency_p90_ms,
        "latency_min_ms": latency_min_ms,
        "latency_max_ms": latency_max_ms,
        "images_per_second_mean": ips_mean,
        "images_per_second_p50": ips_p50,
        "images_per_second_p90": ips_p90,
        "power_available": power_available,
        "power_method": power_method,
        "avg_power_watts": avg_power_watts,
    }


def save_bar_by_device(df, metric, out_png: Path, title, ylabel):
    # grouped bars: x=model_key, grouped by device
    pivot = df.pivot_table(index="model_key", columns="device", values=metric, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_latency_errorbars(df, out_png: Path):
    # show mean + p90 for each model/device (paper-friendly tail latency view)
    d = df.dropna(subset=["latency_mean_ms"]).copy()
    d["label"] = d["model_key"] + " | " + d["device"].astype(str)

    # if p90 missing, fall back to max, else no errorbars
    err = None
    if d["latency_p90_ms"].notna().any():
        err = (d["latency_p90_ms"].fillna(d["latency_mean_ms"]) - d["latency_mean_ms"]).clip(lower=0)
    elif d["latency_max_ms"].notna().any():
        err = (d["latency_max_ms"].fillna(d["latency_mean_ms"]) - d["latency_mean_ms"]).clip(lower=0)

    plt.figure(figsize=(11, 5))
    if err is not None:
        plt.errorbar(d["label"], d["latency_mean_ms"], yerr=err, fmt="o", capsize=4)
    else:
        plt.plot(d["label"], d["latency_mean_ms"], "o")

    plt.title("Latency (mean with tail) by model and device")
    plt.ylabel("latency (ms)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="top-level output folders (e.g., dell_output orin_output thor_output)")
    ap.add_argument("--out", required=True, help="output folder for csv + plots")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(args.roots)
    if not run_dirs:
        raise SystemExit("no run folders found (expected to find run_meta.json + summary.json in subfolders)")

    rows = []
    for rd in run_dirs:
        try:
            rows.append(normalize_one(rd))
        except Exception as e:
            print(f"skipping {rd} due to error: {e}")

    df = pd.DataFrame(rows)

    # save raw table
    df.to_csv(out_dir / "all_runs.csv", index=False)
    try:
        df.to_parquet(out_dir / "all_runs.parquet", index=False)
    except Exception:
        pass

    # basic plots
    if "images_per_second_mean" in df.columns:
        save_bar_by_device(
            df.dropna(subset=["images_per_second_mean"]),
            "images_per_second_mean",
            plots_dir / "throughput_images_per_sec_by_device.png",
            "throughput by model and device",
            "images / second",
        )

    if "latency_mean_ms" in df.columns:
        save_bar_by_device(
            df.dropna(subset=["latency_mean_ms"]),
            "latency_mean_ms",
            plots_dir / "latency_mean_ms_by_device.png",
            "mean latency by model and device",
            "latency (ms)",
        )
        save_latency_errorbars(df, plots_dir / "latency_mean_with_tail_by_device.png")

    # optional: success/failure rate if you want it later
    # df["success"] = (df["num_errors"].fillna(0) == 0).astype(int)

    print(f"wrote: {out_dir / 'all_runs.csv'}")
    print(f"plots in: {plots_dir}")


if __name__ == "__main__":
    main()
