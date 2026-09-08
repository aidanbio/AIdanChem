# tdc_download.py
import os
import argparse
import pandas as pd
from tdc.benchmark_group import admet_group

# ✅ eval_finetuned_tdc_2.py를 single source of truth로 사용
import eval_finetuned_tdc_2 as E


FEATURE_NAMES = [
    "Caco2",  # absorption
    "HIA",
    "Pgp",
    "Bioavailability",
    "Lipophilicity",
    "Solubility",
    "BBB",  # distribution
    "PPBR",
    "VDss",
    "CYP2C9",  # metabolism
    "CYP2D6",
    "CYP3A4",
    "CYP2C9_Substrate",
    "CYP2D6_Substrate",
    "CYP3A4_Substrate",
    "Half_Life",  # excretion
    "Clearance_Hepatocyte",
    "Clearance_Microsome",
    "LD50",  # toxicity
    "hERG",
    "AMES",
    "DILI",
]


def infer_tdc_name_path_from_eval_module() -> str:
    """
    eval_finetuned_tdc_2.get_benchmark_data(feature_name, smiles, tdc_name_path=...)
    여기서 default tdc_name_path를 자동으로 가져온다.
    """
    defaults = getattr(E.get_benchmark_data, "__defaults__", None)
    if not defaults or len(defaults) < 1:
        raise RuntimeError(
            "Cannot infer tdc_name_path from eval_finetuned_tdc_2.get_benchmark_data defaults. "
            "Please ensure get_benchmark_data has default arg tdc_name_path=..."
        )
    # get_benchmark_data의 유일한 default는 tdc_name_path
    return defaults[0]


def load_feature_to_tdc_name_map(tdc_name_path: str) -> dict:
    """
    tdc_name.txt (csv) expected columns: feature, tdc_name
    """
    df = pd.read_csv(tdc_name_path)
    if "feature" not in df.columns or "tdc_name" not in df.columns:
        raise ValueError(f"{tdc_name_path} must have columns: feature, tdc_name")
    return dict(zip(df["feature"].astype(str), df["tdc_name"].astype(str)))


def benchmark_to_df(feature: str, tdc_name: str, benchmark: dict) -> pd.DataFrame:
    """
    benchmark:
      benchmark['train']['Drug'], benchmark['train']['Y']
      benchmark['valid']['Drug'], benchmark['valid']['Y']
      benchmark['test']['Drug'],  benchmark['test']['Y']
    """
    rows = []
    for split in ["train", "valid", "test"]:
        if split not in benchmark:
            continue
        part = benchmark[split]
        drugs = part["Drug"]
        ys = part["Y"]

        # pandas/np 대응
        drugs_list = list(drugs)
        ys_list = ys.tolist() if hasattr(ys, "tolist") else list(ys)

        if len(drugs_list) != len(ys_list):
            raise ValueError(f"[{feature}] split={split} length mismatch: Drug={len(drugs_list)} Y={len(ys_list)}")

        for smi, y in zip(drugs_list, ys_list):
            rows.append(
                {
                    "feature": feature,
                    "tdc_name": tdc_name,
                    "split": split,
                    "smiles": smi,
                    "Y": y,
                }
            )
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Download TDC ADMET benchmark datasets and export to CSV (train/valid/test).")
    p.add_argument("--out_dir", type=str, default="tdc_csv",
                   help="Where to save per-endpoint CSVs (default: tdc_csv)")
    p.add_argument("--tdc_cache_dir", type=str, default="data",
                   help="TDC cache dir passed to admet_group(path=...) (default: data)")
    p.add_argument("--features", type=str, default="",
                   help="Comma-separated features to download (default: all 22)")
    p.add_argument("--write_all", action="store_true",
                   help="Also write combined CSV: all_endpoints.csv")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) feature list 결정
    if args.features.strip():
        features = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        features = FEATURE_NAMES

    # 2) eval_finetuned_tdc_2.py에서 tdc_name.txt 경로 자동 추출
    tdc_name_path = infer_tdc_name_path_from_eval_module()
    if not os.path.exists(tdc_name_path):
        raise FileNotFoundError(f"tdc_name_path inferred but not found: {tdc_name_path}")

    feature_to_tdc = load_feature_to_tdc_name_map(tdc_name_path)

    # 3) TDC group 준비 (다운로드/캐시 포함)
    group = admet_group(path=args.tdc_cache_dir)

    failures = []
    all_dfs = []

    for feat in features:
        if feat not in feature_to_tdc:
            failures.append((feat, "", f"feature not found in mapping file: {tdc_name_path}"))
            print(f"[Fail] {feat}: not found in mapping")
            continue

        tdc_name = feature_to_tdc[feat]
        print(f"\n[Download] feature={feat} | tdc_name={tdc_name}")

        try:
            benchmark = group.get(tdc_name)
            df = benchmark_to_df(feat, tdc_name, benchmark)

            out_path = os.path.join(args.out_dir, f"{feat}.csv")
            df.to_csv(out_path, index=False)
            print(f"[Saved] {out_path} | rows={len(df)}")

            all_dfs.append(df)

        except Exception as e:
            failures.append((feat, tdc_name, str(e)))
            print(f"[Fail] feature={feat} | tdc_name={tdc_name} | error={e}")

    # 4) 실패 로그 저장
    if failures:
        fail_path = os.path.join(args.out_dir, "download_failures.csv")
        pd.DataFrame(failures, columns=["feature", "tdc_name", "error"]).to_csv(fail_path, index=False)
        print(f"\n[Done with failures] failures saved: {fail_path}")
    else:
        print("\n[Done] all endpoints downloaded successfully.")

    # 5) 전체 합본 CSV (옵션)
    if args.write_all and all_dfs:
        all_path = os.path.join(args.out_dir, "all_endpoints.csv")
        pd.concat(all_dfs, ignore_index=True).to_csv(all_path, index=False)
        print(f"[Saved] {all_path}")


if __name__ == "__main__":
    main()
