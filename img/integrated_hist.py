#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified distribution plotting for PubChem (local CSV) vs TDC benchmark.

Outputs:
  1) pubchem_tdc_dist.png   (Supplementary): 22 endpoints, each shown as PubChem vs TDC side-by-side
  2) 3endpoints.png         (Main): LD50, Solubility, HIA side-by-side for easy comparison

Key design choices (for fair comparison):
- Same bins for PubChem and TDC per endpoint (computed from combined values)
- Use density=True so shapes are comparable even if sample counts differ
- No KDE by default (recommended esp. for binary endpoints)
- Consistent titles/labels/layout across both figures

Requirements:
  pip install pandas matplotlib tdc
"""

import argparse
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TDC
from tdc.benchmark_group import admet_group


# ---- Endpoints (TDC-style order) ----
TDC_FEATURE_ORDER = [
    # Absorption
    "Caco2", "HIA", "Pgp", "Bioavailability", "Lipophilicity", "Solubility",
    # Distribution
    "BBB", "PPBR", "VDss",
    # Metabolism
    "CYP2C9", "CYP2D6", "CYP3A4", "CYP2C9_Substrate", "CYP2D6_Substrate", "CYP3A4_Substrate",
    # Excretion
    "Half_Life", "Clearance_Hepatocyte", "Clearance_Microsome",
    # Toxicity
    "LD50", "hERG", "AMES", "DILI"
]

MAIN_ENDPOINTS = ["LD50", "Solubility", "HIA"]


def read_tdc_name_map(path: str) -> Dict[str, str]:
    """
    Reads mapping file (feature -> tdc benchmark name).
    Supports common formats:
      - feature<whitespace>tdc_name
      - feature,tdc_name
      - feature:tdc_name
    Ignores blank lines and comment lines starting with '#'.
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
            elif ":" in s:
                parts = [p.strip() for p in s.split(":") if p.strip()]
            else:
                parts = s.split()

            if len(parts) >= 2:
                feature = parts[0]
                tdc_name = parts[1]
                mapping[feature] = tdc_name

    if not mapping:
        raise ValueError(f"Could not parse any mappings from {path}")

    return mapping


def load_pubchem_csv(csv_path: str, endpoints: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only endpoints that exist
    cols = [c for c in endpoints if c in df.columns]
    if not cols:
        raise ValueError(
            f"No endpoint columns found in PubChem CSV. "
            f"Expected one of: {endpoints}\n"
            f"CSV columns (sample): {list(df.columns)[:30]}"
        )
    return df[cols].copy()


def load_tdc_labels(
    feature: str,
    tdc_name_map: Dict[str, str],
    tdc_data_root: str = "data"
) -> np.ndarray:
    if feature not in tdc_name_map:
        raise KeyError(f"Feature '{feature}' not found in tdc_name mapping file.")

    tdc_name = tdc_name_map[feature]
    group = admet_group(path=tdc_data_root)
    benchmark = group.get(tdc_name)

    # benchmark['test']['Y'] is typically a list/array shape (N,) or (N,1)
    y = np.array(benchmark["test"]["Y"]).squeeze()
    # Drop NaNs if any
    y = y[~np.isnan(y)]
    return y


def is_binary(y: np.ndarray, tol: float = 1e-9) -> bool:
    """Heuristic: values are essentially subset of {0,1}."""
    if y.size == 0:
        return False
    uniq = np.unique(np.round(y.astype(float), 9))
    return uniq.size <= 2 and np.all(np.isin(uniq, [0.0, 1.0]))


def compute_shared_bins(pub: np.ndarray, tdc: np.ndarray, bins: int = 30) -> np.ndarray:
    """
    Compute bin edges shared between two datasets for the same endpoint.
    Uses combined min/max to ensure identical x-scale and bins.
    Handles binary endpoints with fixed bins.
    """
    pub = pub.astype(float)
    tdc = tdc.astype(float)

    combined = np.concatenate([pub, tdc]) if (pub.size and tdc.size) else (pub if pub.size else tdc)
    if combined.size == 0:
        # fallback
        return np.linspace(0, 1, bins + 1)

    if is_binary(combined):
        return np.array([-0.5, 0.5, 1.5])

    lo = np.nanmin(combined)
    hi = np.nanmax(combined)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(0, 1, bins + 1)

    if math.isclose(lo, hi):
        # Expand a tiny bit to avoid zero-width bins
        eps = 1e-6 if hi == 0 else abs(hi) * 1e-6
        lo -= eps
        hi += eps

    return np.linspace(lo, hi, bins + 1)


def plot_pair_hist(
    ax: plt.Axes,
    y: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    show_ylabel: bool = False
):
    ax.hist(y, bins=bin_edges, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_ylabel("")


def make_supp_figure(
    pub_df: pd.DataFrame,
    tdc_name_map: Dict[str, str],
    out_path: str,
    tdc_data_root: str,
    bins: int = 30,
    endpoints: Optional[List[str]] = None
):
    endpoints = endpoints or list(pub_df.columns)

    # Build 4-column grid: (PubChem, TDC) pairs => 2 endpoints per row
    n = len(endpoints)
    n_rows = math.ceil(n / 2)
    n_cols = 4

    fig_w = 16
    fig_h = max(10, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for i, feature in enumerate(endpoints):
        row = i // 2
        col_pair = (i % 2) * 2  # 0 or 2

        ax_pub = axes[row, col_pair]
        ax_tdc = axes[row, col_pair + 1]

        pub_y = pub_df[feature].dropna().to_numpy()

        try:
            tdc_y = load_tdc_labels(feature, tdc_name_map, tdc_data_root=tdc_data_root)
        except Exception as e:
            tdc_y = np.array([])
            ax_tdc.text(0.5, 0.5, f"TDC load failed:\n{e}", ha="center", va="center", fontsize=7)
            ax_tdc.set_axis_off()

        bin_edges = compute_shared_bins(pub_y, tdc_y, bins=bins)

        plot_pair_hist(ax_pub, pub_y, bin_edges, f"{feature} — PubChem", show_ylabel=(col_pair == 0))
        if tdc_y.size:
            plot_pair_hist(ax_tdc, tdc_y, bin_edges, f"{feature} — TDC", show_ylabel=False)

        # Tidy x-labels: only bottom row show xlabel
        if row == n_rows - 1:
            ax_pub.set_xlabel("Value", fontsize=8)
            if tdc_y.size:
                ax_tdc.set_xlabel("Value", fontsize=8)
        else:
            ax_pub.set_xlabel("")
            if tdc_y.size:
                ax_tdc.set_xlabel("")

    # Turn off unused axes (if odd number of endpoints, last two plots empty)
    total_axes = n_rows * n_cols
    used_axes = (n // 2) * 4 + (4 if n % 2 == 1 else 0)  # because one endpoint uses 2 axes, but row has 4 axes
    # More robust: explicitly clear any axes not touched
    touched = set()
    for i, feature in enumerate(endpoints):
        row = i // 2
        col_pair = (i % 2) * 2
        touched.add((row, col_pair))
        touched.add((row, col_pair + 1))
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in touched:
                axes[r, c].set_axis_off()

    fig.suptitle("PubChem vs TDC Label Distributions (Aligned Bins, Density)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_main_figure(
    pub_df: pd.DataFrame,
    tdc_name_map: Dict[str, str],
    out_path: str,
    tdc_data_root: str,
    bins: int = 30,
    endpoints: Optional[List[str]] = None
):
    endpoints = endpoints or MAIN_ENDPOINTS

    # 3 rows x 2 columns (PubChem | TDC)
    n_rows = len(endpoints)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
    axes = np.atleast_2d(axes)

    for r, feature in enumerate(endpoints):
        if feature not in pub_df.columns:
            raise ValueError(f"PubChem CSV does not contain endpoint column '{feature}'")

        pub_y = pub_df[feature].dropna().to_numpy()
        tdc_y = load_tdc_labels(feature, tdc_name_map, tdc_data_root=tdc_data_root)
        bin_edges = compute_shared_bins(pub_y, tdc_y, bins=bins)

        plot_pair_hist(axes[r, 0], pub_y, bin_edges, f"{feature} — PubChem", show_ylabel=True)
        plot_pair_hist(axes[r, 1], tdc_y, bin_edges, f"{feature} — TDC", show_ylabel=False)

        axes[r, 0].set_xlabel("Value", fontsize=8)
        axes[r, 1].set_xlabel("Value", fontsize=8)

    fig.suptitle("PubChem vs TDC: LD50 / Solubility / HIA (Aligned Bins, Density)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pubchem_csv",
        type=str,
        default="../data/pubchem/data_scaffold_5fold.csv",
        help="Path to PubChem CSV (default: ../data_scaffold_5fold.csv)"
    )

    parser.add_argument(
        "--tdc_name_txt",
        type=str,
        default="/home/jin/Lim/Aidan/AR/admet_finetune/tdc_name.txt",
        help="Path to mapping file feature->tdc_name (default: tdc_name.txt)"
    )

    parser.add_argument(
        "--tdc_data_root",
        type=str,
        default="../data",
        help="Directory where TDC stores/downloads benchmark data (default: data/)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for continuous endpoints (default: 30)"
    )

    parser.add_argument(
        "--supp_out",
        type=str,
        default="pubchem_tdc_dist.png",
        help="Filename for supplementary figure (default: pubchem_tdc_dist.png)"
    )

    parser.add_argument(
        "--main_out",
        type=str,
        default="3endpoints.png",
        help="Filename for main figure (default: 3endpoints.png)"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    supp_path = os.path.join(args.out_dir, args.supp_out)
    main_path = os.path.join(args.out_dir, args.main_out)

    tdc_map = read_tdc_name_map(args.tdc_name_txt)
    pub_df = load_pubchem_csv(args.pubchem_csv, TDC_FEATURE_ORDER)

    # For supplementary, we plot only endpoints that exist in PubChem CSV, in the standard order
    endpoints = [c for c in TDC_FEATURE_ORDER if c in pub_df.columns]

    make_supp_figure(
        pub_df=pub_df,
        tdc_name_map=tdc_map,
        out_path=supp_path,
        tdc_data_root=args.tdc_data_root,
        bins=args.bins,
        endpoints=endpoints
    )

    # Main figure endpoints
    make_main_figure(
        pub_df=pub_df,
        tdc_name_map=tdc_map,
        out_path=main_path,
        tdc_data_root=args.tdc_data_root,
        bins=args.bins,
        endpoints=MAIN_ENDPOINTS
    )

    print(f"[OK] Saved: {supp_path}")
    print(f"[OK] Saved: {main_path}")


if __name__ == "__main__":
    main()
