#!/usr/bin/env python3
"""
Generate normalized benchmarking tables and plots, plus device-comparison charts.
Matches the earlier vision benchmarking PDF style and adds device vs baseline speedups.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# helpers
# -------------------------

def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def find_run_dirs(roots):
    run_dirs = []
    for r in roots:
        root = Path(r)
        for meta in root.rglob("run_meta.json"):
            run_dir = meta.parent
            if (run_dir / "summary.json").exists():
                run_dirs.append(run_dir)
    return sorted(set(run_dirs))

def infer_device_label(run_dir: Path):
    # choose something stable for legend/grouping
    for part in run_dir.parts:
        if part.endswith("_output"):
            return part.replace("_output", "")
    return "unknown"

def infer_family(model_key: str, model_id: str):
    s = f"{model_key} {model_id}".lower()
    if "gemma" in s:
        return "gemma"
    if "llava" in s:
        return "llava"
    if "internvl" in s:
        return "internvl"
    if "phi" in s:
        return "phi"
    if "moondream" in s:
        return "moondream"
    if "smolvlm" in s:
        return "smolvlm"
    return "other"

def infer_dtype_short(dtype: str):
    if not dtype:
        return "na"
    d = dtype.lower()
    if "bfloat16" in d or "bf16" in d:
        return "bf16"
    if "float16" in d or "fp16" in d:
        return "fp16"
    if "float32" in d or "fp32" in d:
        return "fp32"
    return d

def normalize_one(run_dir: Path):
    meta = read_json(run_dir / "run_meta.json")
    summ = read_json(run_dir / "summary.json")

    model_key = pick(meta, ["model_key"], None) or run_dir.parts[-2]
    model_id = pick(meta, ["model_id", "model"], "")

    device = infer_device_label(run_dir)
    dtype = infer_dtype_short(pick(meta, ["dtype"], ""))

    row = {
        "run_dir": str(run_dir),
        "device": device,
        "model_key": model_key,
        "model_id": model_id,
        "family": infer_family(model_key, model_id),
        "dtype": dtype,
        "run_group": pick(meta, ["run_group"], run_dir.name),

        "num_images": pick(meta, ["num_images"], pick(summ, ["num_images"], None)),
        "num_tasks": pick(summ, ["num_tasks"], None),

        "num_errors": pick(summ, ["num_errors"], 0),

        "lat_ms_mean": pick(summ, ["latency_ms_mean", "latency_mean_ms"], None),
        "lat_ms_p50": pick(summ, ["latency_ms_p50", "latency_p50_ms"], None),
        "lat_ms_p90": pick(summ, ["latency_ms_p90", "latency_p90_ms"], None),

        "ips_mean": pick(summ, ["images_per_second_mean"], None),
        "ips_p50": pick(summ, ["images_per_second_p50"], None),
        "ips_p90": pick(summ, ["images_per_second_p90"], None),

        "avg_power_w": pick(summ, ["avg_power_watts", "avg_total_power_watts", "power_avg_watts"], None),
        "power_available": bool(pick(meta, ["power_monitoring_available"], pick(summ, ["power_monitoring_available"], False))),
    }

    if row["ips_mean"] is not None and row["avg_power_w"] not in (None, 0):
        row["ips_per_watt"] = row["ips_mean"] / row["avg_power_w"]
        row["joules_per_image"] = row["avg_power_w"] / row["ips_mean"]
    else:
        row["ips_per_watt"] = None
        row["joules_per_image"] = None

    row["success_rate_pct"] = 100.0 if (row["num_errors"] == 0) else 0.0

    return row


# -------------------------
# plotting
# -------------------------

FAMILY_ORDER = ["gemma", "llava", "internvl", "phi", "moondream", "smolvlm", "other"]

def style_axes(ax):
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)

def plot_bar(df, value_col, title, ylabel, out_png):
    d = df.dropna(subset=[value_col]).copy()
    if d.empty:
        return False

    d["family_rank"] = d["family"].apply(lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 999)
    d = d.sort_values(["family_rank", "model_key", "dtype", "device"])

    labels = [f"{mk}:{dt}:{dev}" for mk, dt, dev in zip(d["model_key"], d["dtype"], d["device"])]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(labels, d[value_col].values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    style_axes(ax)
    plt.xticks(rotation=90, ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True

def plot_scatter_size(df, size_col, y_col, title, ylabel, out_png):
    d = df.dropna(subset=[size_col, y_col]).copy()
    if d.empty:
        return False

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for fam in FAMILY_ORDER:
        sub = d[d["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub[size_col], sub[y_col], label=fam, marker="x")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model size (B params)")
    ax.legend(loc="best", fontsize=8)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True


def pick_baseline_device(df, preferred=("dell", "dgx", "gb10")):
    devices = sorted(df["device"].dropna().unique().tolist())
    for p in preferred:
        for d in devices:
            if p in str(d).lower():
                return d
    return devices[0] if devices else None

def add_speedup_columns(df, baseline_device):
    d = df.copy()
    gcols = ["device", "model_key", "dtype"]
    agg = d.groupby(gcols, as_index=False).agg({
        "ips_mean": "mean",
        "lat_ms_mean": "mean",
        "lat_ms_p90": "mean",
    })

    base = agg[agg["device"] == baseline_device].copy()
    base = base.rename(columns={
        "ips_mean": "ips_base",
        "lat_ms_mean": "lat_base",
        "lat_ms_p90": "p90_base",
    })[["model_key", "dtype", "ips_base", "lat_base", "p90_base"]]

    merged = agg.merge(base, on=["model_key", "dtype"], how="left")

    merged["speedup_ips_vs_base"] = merged["ips_mean"].div(merged["ips_base"])
    merged["lat_ratio_vs_base"] = merged["lat_ms_mean"].div(merged["lat_base"])
    merged["p90_ratio_vs_base"] = merged["lat_ms_p90"].div(merged["p90_base"])

    return merged

def plot_speedup_bars(dfx, value_col, out_png, title, ylabel):
    piv = dfx.pivot_table(index="model_key", columns="device", values=value_col, aggfunc="mean")
    if piv.empty:
        return False
    ax = piv.plot(kind="bar", figsize=(12, 6))
    ax.axhline(1.0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True

def plot_heatmap(dfx, value_col, out_png, title):
    piv = dfx.pivot_table(index="model_key", columns="device", values=value_col, aggfunc="mean")
    if piv.empty:
        return False
    data = piv.values

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("device")
    ax.set_ylabel("model")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="top-level output dirs (e.g., dell_output orin_output thor_output)")
    ap.add_argument("--out", required=True, help="output directory for csv + plots")
    ap.add_argument("--sizes", default=None, help="optional CSV with columns: model_key, params_b (e.g., 7 for 7B)")
    ap.add_argument("--only_500", action="store_true", help="filter to runs with 500 images (by num_images or run_group contains 500)")
    ap.add_argument("--device_cmp", action="store_true", default=True, help="generate device comparison plots")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(args.roots)
    rows = [normalize_one(rd) for rd in run_dirs]
    df = pd.DataFrame(rows)

    if args.only_500:
        df = df[(df["num_images"] == 500) | df["run_group"].str.contains("500", na=False)].copy()

    df["params_b"] = None
    if args.sizes:
        sizes = pd.read_csv(args.sizes)
        df = df.merge(sizes[["model_key", "params_b"]], on="model_key", how="left", suffixes=("", "_y"))
        df["params_b"] = df["params_b_y"].combine_first(df["params_b"])
        df.drop(columns=[c for c in df.columns if c.endswith("_y")], inplace=True)

    df.to_csv(out / "all_runs_normalized.csv", index=False)

    plot_bar(df, "ips_mean", "Vision Models - Average Images per Second", "Images/sec", plots / "plot_01_vision_images_per_sec.png")
    plot_bar(df, "lat_ms_mean", "Vision Models - Average Latency (ms)", "Latency (ms)", plots / "plot_02_vision_latency_ms.png")
    plot_bar(df, "ips_per_watt", "Vision Models - Images per Second per Watt", "Images/sec/W", plots / "plot_03_vision_ips_per_watt.png")
    plot_bar(df, "joules_per_image", "Vision Models - Joules per Image", "Joules/image", plots / "plot_04_vision_joules_per_image.png")
    plot_bar(df, "success_rate_pct", "Vision Models - Success Rate (%)", "Success Rate (%)", plots / "plot_05_vision_success_rate.png")

    plot_scatter_size(df, "params_b", "ips_mean", "Model Size (Parameters) vs. Images/sec", "Images/sec", plots / "plot_06_size_vs_ips.png")
    plot_scatter_size(df, "params_b", "ips_per_watt", "Model Size (Parameters) vs. Images/sec/W", "Images/sec/W", plots / "plot_07_size_vs_ips_per_watt.png")
    plot_scatter_size(df, "params_b", "joules_per_image", "Model Size (Parameters) vs. Joules/Image", "Joules/image", plots / "plot_08_size_vs_joules_per_image.png")

    if args.device_cmp:
        baseline = pick_baseline_device(df)
        if baseline:
            dcmp = add_speedup_columns(df, baseline)

            plot_speedup_bars(
                dcmp.dropna(subset=["speedup_ips_vs_base"]),
                "speedup_ips_vs_base",
                plots / f"devcmp_01_speedup_ips_vs_{baseline}.png",
                f"throughput speedup vs {baseline} (images/sec ratio)",
                f"speedup vs {baseline} (×)",
            )

            plot_speedup_bars(
                dcmp.dropna(subset=["lat_ratio_vs_base"]),
                "lat_ratio_vs_base",
                plots / f"devcmp_02_latency_ratio_vs_{baseline}.png",
                f"latency ratio vs {baseline} (mean latency)",
                f"latency / {baseline} (×)",
            )

            plot_speedup_bars(
                dcmp.dropna(subset=["p90_ratio_vs_base"]),
                "p90_ratio_vs_base",
                plots / f"devcmp_03_p90_ratio_vs_{baseline}.png",
                f"tail latency ratio vs {baseline} (p90 latency)",
                f"p90 / {baseline} (×)",
            )

            plot_heatmap(
                df.dropna(subset=["ips_mean"]),
                "ips_mean",
                plots / "devcmp_04_heatmap_ips_mean.png",
                "heatmap: throughput (images/sec) by model and device",
            )

            plot_heatmap(
                df.dropna(subset=["lat_ms_mean"]),
                "lat_ms_mean",
                plots / "devcmp_05_heatmap_latency_mean.png",
                "heatmap: mean latency (ms) by model and device",
            )
        else:
            print("[device-cmp] skipped: no devices found to pick a baseline")

    print(f"wrote normalized table: {out/'all_runs_normalized.csv'}")
    print(f"plots saved to: {plots}")


if __name__ == "__main__":
    main()
