# evaluation.py
import os
import re
import argparse
import pandas as pd
import numpy as np

from eval_finetuned_tdc_2 import (
    load_model,
    predict_admet,
    get_benchmark_data,
    evaluate_tdc_predictions,
)

# output dir prefix -> huggingface base model name
BASE_MODEL_MAP = {
    "deberta": "sagawa/ZINC-deberta",
    "roberta": "entropy/roberta_zinc_480m",
    "bert": "unikei/bert-base-smiles",
}

FEATURE_NAMES = [
    "Caco2",
    "HIA",
    "Pgp",
    "Bioavailability",
    "Lipophilicity",
    "Solubility",
    "BBB",
    "PPBR",
    "VDss",
    "CYP2C9",
    "CYP2D6",
    "CYP3A4",
    "CYP2C9_Substrate",
    "CYP2D6_Substrate",
    "CYP3A4_Substrate",
    "Half_Life",
    "Clearance_Hepatocyte",
    "Clearance_Microsome",
    "LD50",
    "hERG",
    "AMES",
    "DILI",
]


def infer_base_model_name_from_output_dir(output_dir_name: str) -> str:
    """
    output dir name examples:
      - deberta_f0_lr3_g1
      - roberta_f0_lr5_g2
      - bert_f0_lr3_g1
    """
    prefix = output_dir_name.split("_", 1)[0].strip().lower()
    if prefix not in BASE_MODEL_MAP:
        raise ValueError(
            f"Unknown output dir prefix '{prefix}'. Expected one of: {list(BASE_MODEL_MAP.keys())}"
        )
    return BASE_MODEL_MAP[prefix]


def list_checkpoints(output_dir: str):
    """
    Find checkpoint files like:
      checkpoint_epoch_1.pth, checkpoint_epoch_2.pth, ...
    Return sorted list of (epoch:int, path:str)
    """
    pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
    items = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            epoch = int(m.group(1))
            items.append((epoch, os.path.join(output_dir, fn)))
    items.sort(key=lambda x: x[0])
    return items

def get_eval_header():
    score_cols = [f"{f}_score" for f in FEATURE_NAMES]
    rpp_cols = [f"{f}_rpp" for f in FEATURE_NAMES]
    return ["output_dir", "base_model", "epoch", "mean_rpp"] + score_cols + rpp_cols

def append_eval_row(eval_txt_path: str, header: list, row: dict):
    file_exists = os.path.exists(eval_txt_path)

    mode = "a" if file_exists else "w"
    with open(eval_txt_path, mode) as f:
        # 파일이 없을 때만 header 작성
        if not file_exists:
            f.write("\t".join(header) + "\n")

        values = []
        for col in header:
            v = row.get(col, "")
            if isinstance(v, float):
                values.append(f"{v:.6f}" if not np.isnan(v) else "nan")
            else:
                values.append(str(v))
        f.write("\t".join(values) + "\n")
        f.flush()  # 🔥 중요: 즉시 디스크에 반영


def load_leaderboard(leaderboard_csv: str) -> pd.DataFrame:
    """
    Expect a CSV that contains rows named 'SOTA' and 'ref',
    and columns as endpoints (Caco2, HIA, ...).
    """
    lb = pd.read_csv(leaderboard_csv)

    # If first column is an index-like column
    if lb.columns[0].lower().startswith("unnamed") or lb.columns[0] in ["index", "row", "name"]:
        lb = lb.set_index(lb.columns[0])

    # If still not indexed, try set first column as index if it contains SOTA/ref
    if "SOTA" not in lb.index or "ref" not in lb.index:
        first_col = lb.columns[0]
        if lb[first_col].astype(str).isin(["SOTA", "ref"]).any():
            lb = lb.set_index(first_col)

    if "SOTA" not in lb.index or "ref" not in lb.index:
        raise ValueError("leaderboard csv must contain 'SOTA' and 'ref' rows.")

    lb = lb.apply(pd.to_numeric, errors="coerce")
    return lb


def compute_rpp(score: float, sota: float, ref: float) -> float:
    """
    RPP = (S_model - S_SOTA) / (S_ref - S_SOTA)
    SOTA -> 0, ref -> 1
    """
    denom = (ref - sota)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (score - sota) / denom


def evaluate_one_checkpoint(
    checkpoint_path: str,
    mean_std_path: str,
    base_model_name: str,
    leaderboard: pd.DataFrame,
    num_classes: int,
    smiles: bool,
    batch_size: int,
    cached_benchmark=None,
):
    """
    Returns:
      scores: dict {feature: score}
      rpps:   dict {feature: rpp}
      metrics: dict {feature: metric_name}
    cached_benchmark:
      optional dict {feature: (rep_list, labels)} to avoid repeated downloads/processing
    """
    model, tokenizer, mean_values, std_values, column_names = load_model(
        checkpoint_path=checkpoint_path,
        mean_std_path=mean_std_path,
        num_classes=num_classes,
        base_model_name=base_model_name,
    )

    scores = {}
    rpps = {}
    metrics_used = {}

    for feature in FEATURE_NAMES:
        if cached_benchmark is not None and feature in cached_benchmark:
            rep_list, true_labels = cached_benchmark[feature]
        else:
            rep_list, true_labels = get_benchmark_data(feature, smiles)
            if cached_benchmark is not None:
                cached_benchmark[feature] = (rep_list, true_labels)

        pred_labels = predict_admet(model, tokenizer, rep_list, batch_size=batch_size)

        m = evaluate_tdc_predictions(
            preds=pred_labels,
            labels=true_labels,
            mean_values=mean_values,
            std_values=std_values,
            column_names=column_names,
            feature_name=feature,
            base_model_name=base_model_name,
        )
        score = float(m["Score"])
        metric = str(m["Metric"])

        scores[feature] = score
        metrics_used[feature] = metric

        # leaderboard columns assumed to be endpoint names
        if feature not in leaderboard.columns:
            rpps[feature] = np.nan
        else:
            sota = float(leaderboard.loc["SOTA", feature])
            ref = float(leaderboard.loc["ref", feature])
            rpps[feature] = compute_rpp(score, sota, ref)

    return scores, rpps, metrics_used


def write_eval_txt(
    out_path: str,
    output_dir_name: str,
    base_model_name: str,
    rows: list,
):
    """
    rows: list of dict
      each dict has:
        epoch, mean_rpp, and per-feature score/rpp entries
    """
    # Build a stable header:
    # epoch | mean_rpp | {feature_score}... | {feature_rpp}...
    score_cols = [f"{f}_score" for f in FEATURE_NAMES]
    rpp_cols = [f"{f}_rpp" for f in FEATURE_NAMES]
    header = ["output_dir", "base_model", "epoch", "mean_rpp"] + score_cols + rpp_cols

    with open(out_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            vals = []
            for col in header:
                v = row.get(col, "")
                if isinstance(v, float):
                    # keep readable precision
                    vals.append(f"{v:.6f}" if not np.isnan(v) else "nan")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True,
                   help="Target outputs directory containing checkpoint_epoch_*.pth and label_mean_std.npz")
    p.add_argument("--leaderboard_csv", type=str, default="data/tdc_leaderboard.csv",
                   help="CSV with rows SOTA/ref and columns as endpoints")
    p.add_argument("--num_classes", type=int, default=22)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--smiles", action="store_true", help="Use SMILES (default). If not set, uses SELFIES path in get_benchmark_data.")
    p.add_argument("--no_cache_benchmark", action="store_true",
                   help="Disable caching of benchmark test sets across epochs")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    output_dir_name = os.path.basename(output_dir.rstrip("/"))
    base_model_name = infer_base_model_name_from_output_dir(output_dir_name)

    mean_std_path = os.path.join(output_dir, "label_mean_std.npz")
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"label_mean_std.npz not found in: {output_dir}")

    ckpts = list_checkpoints(output_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pth found in: {output_dir}")

    leaderboard = load_leaderboard(args.leaderboard_csv)

    cached_benchmark = None if args.no_cache_benchmark else {}

    eval_txt_path = os.path.join(output_dir, "eval.txt")
    header = get_eval_header()

    for epoch, ckpt_path in ckpts:
        print(f"[Eval] epoch={epoch} ckpt={os.path.basename(ckpt_path)}")

        try:
            scores, rpps, metrics_used = evaluate_one_checkpoint(
                checkpoint_path=ckpt_path,
                mean_std_path=mean_std_path,
                base_model_name=base_model_name,
                leaderboard=leaderboard,
                num_classes=args.num_classes,
                smiles=True,
                batch_size=args.batch_size,
                cached_benchmark=cached_benchmark,
            )

            mean_rpp = float(np.nanmean(list(rpps.values())))

            row = {
                "output_dir": output_dir_name,
                "base_model": base_model_name,
                "epoch": epoch,
                "mean_rpp": mean_rpp,
            }
            for f in FEATURE_NAMES:
                row[f"{f}_score"] = float(scores.get(f, np.nan))
                row[f"{f}_rpp"] = float(rpps.get(f, np.nan))

        except Exception as e:
            print(f"⚠️ Failed at epoch {epoch}: {e}")

            row = {
                "output_dir": output_dir_name,
                "base_model": base_model_name,
                "epoch": epoch,
                "mean_rpp": "-",
            }
            for f in FEATURE_NAMES:
                row[f"{f}_score"] = "-"
                row[f"{f}_rpp"] = "-"

        # ✅ epoch 결과 즉시 기록
        append_eval_row(eval_txt_path, header, row)

        # ✅ 기록이 끝났으면 checkpoint 무조건 삭제
        try:
            os.remove(ckpt_path)
            print(f"🗑️ Deleted checkpoint: {os.path.basename(ckpt_path)}")
        except Exception as rm_e:
            print(f"⚠️ Failed to delete {ckpt_path}: {rm_e}")


if __name__ == "__main__":
    main()

