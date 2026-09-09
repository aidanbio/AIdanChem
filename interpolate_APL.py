# interpolate_APL.py
"""
APL (ADMET Path Length) on scaffold 5-fold validation split.

✅ 요구사항 반영
- --load_model (checkpoint .pth 경로)만 필수
- checkpoint 경로의 상위 폴더명에서
  - base model prefix: deberta / roberta / bert  -> BASE_MODEL_MAP으로 HF 모델명 자동 설정
  - fold: f0, f1, ... 를 자동 추출 (예: deberta_f0_lr3_g1 -> fold=0)
- val.csv 대신 data_scaffold_5fold.csv 사용
  - data_scaffold_5fold.csv의 'fold' 컬럼을 보고, 추출된 fold == val fold 로 사용
  - (학습 코드에서도 fold 컬럼으로 train/val split을 나누는 구조) :contentReference[oaicite:0]{index=0}

APL 정의(기본):
  repA, repB를 alpha로 선형 보간한 경로에서
    step_ratio_i = ||ΔADMET||2 / (||Δrep||2 + eps)
    APL(pair) = mean_i step_ratio_i
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from finetuning import CustomRegModel


BASE_MODEL_MAP = {
    "deberta": "sagawa/ZINC-deberta",
    "roberta": "entropy/roberta_zinc_480m",
    "bert": "unikei/bert-base-smiles",
}


def infer_base_model_and_fold_from_checkpoint_path(load_model_path: str):
    """
    Example:
      outputs/deberta_f0_lr3_g1/checkpoint_epoch_10.pth
        -> dir_name: deberta_f0_lr3_g1
        -> prefix : deberta
        -> fold   : 0 (from f0)
    """
    dir_name = os.path.basename(os.path.normpath(os.path.dirname(load_model_path))).lower()

    # base model prefix
    prefix = dir_name.split("_", 1)[0]
    if prefix not in BASE_MODEL_MAP:
        raise ValueError(
            f"Cannot infer base model from output dir '{dir_name}'. "
            f"Expected prefix in {list(BASE_MODEL_MAP.keys())}"
        )
    base_model_name = BASE_MODEL_MAP[prefix]

    # fold (f0, f1, ...)
    m = re.search(r"(?:^|_)f(\d+)(?:_|$)", dir_name)
    if not m:
        raise ValueError(
            f"Cannot infer fold from output dir '{dir_name}'. "
            f"Expected pattern like '_f0_' in directory name (e.g., deberta_f0_lr3_g1)."
        )
    fold = int(m.group(1))

    return base_model_name, fold, dir_name


@torch.no_grad()
def calculate_APL(
    model: CustomRegModel,
    inputs_A: dict,
    inputs_B: dict,
    alpha_values: np.ndarray,
    device: torch.device,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Returns:
      apl_value: (batch,) numpy array
    """
    # (batch, hidden)
    repA = model.extract_rep(inputs_A["input_ids"], inputs_A["attention_mask"]).detach()
    repB = model.extract_rep(inputs_B["input_ids"], inputs_B["attention_mask"]).detach()

    # (num_alpha, batch, hidden) via broadcasting
    alphas = torch.tensor(alpha_values, device=device, dtype=repA.dtype).view(-1, 1, 1)
    interp_reps = (1.0 - alphas) * repA.unsqueeze(0) + alphas * repB.unsqueeze(0)

    # Δrep: (num_alpha-1, batch, hidden)
    rep_diffs = interp_reps[1:] - interp_reps[:-1]

    # Predict ADMET for all alphas in one forward
    na, bsz, hdim = interp_reps.shape
    flat_reps = interp_reps.reshape(na * bsz, hdim)          # (na*bsz, hidden)
    flat_preds = model.regressor(flat_reps)                  # (na*bsz, 22)
    admet_preds = flat_preds.reshape(na, bsz, -1)            # (na, bsz, 22)

    # ΔADMET: (num_alpha-1, batch, 22)
    admet_diffs = admet_preds[1:] - admet_preds[:-1]

    # step_ratio_i = ||ΔADMET|| / (||Δrep|| + eps)
    rep_step_norm = torch.linalg.norm(rep_diffs, ord=2, dim=2)        # (na-1, bsz)
    admet_step_norm = torch.linalg.norm(admet_diffs, ord=2, dim=2)    # (na-1, bsz)
    step_ratio = admet_step_norm / (rep_step_norm + eps)             # (na-1, bsz)

    # APL(pair) = mean over steps
    apl_value = step_ratio.mean(dim=0)  # (bsz,)
    return apl_value.detach().cpu().numpy()


# 여러 alpha 값에 따른 ADMET 변화 분석 함수
def calculate_PPL(model, inputs_A, inputs_B, alpha_values, device):
    repA = model.extract_rep(inputs_A["input_ids"], inputs_A["attention_mask"]).detach().cpu().numpy()
    repB = model.extract_rep(inputs_B["input_ids"], inputs_B["attention_mask"]).detach().cpu().numpy()     # (batch, 768)

    interp_reps = np.stack([(1 - alpha) * repA + alpha * repB for alpha in alpha_values])     #(num_alpha, batch, 768)

    # 연속된 rep 간의 차이 계산
    rep_diffs = np.diff(interp_reps, axis=0)   # Δrep_i = rep_{i+1} - rep_i       (num_alpha-1, batch, 768)
    admet_preds = np.array([model.regressor(torch.tensor(rep, device=device)).detach().cpu().numpy() for rep in interp_reps])   # (num_alpha, batch, 22)
    admet_diffs = np.diff(admet_preds, axis=0)  # ΔM_i = M(rep_{i+1}) - M(rep_i)                     (num_alpha-1, batch, 22)

    # PPL 계산
    ppl_per_step = np.sum(admet_diffs ** 2, axis=2) / (np.sum(rep_diffs ** 2, axis=2) + 1e-8)  # 안정화 위해 작은 값 추가 (num_alpha-1, batch)
    ppl_value = np.mean(ppl_per_step, axis=0)  # 분자쌍 개별 평균 (batch,)

    return ppl_value


class APLSMILESDataset(Dataset):
    """Randomly pairs SMILES and returns tokenized A/B."""

    def __init__(self, tokenizer, smiles, max_length: int = 512, seed: int = 42):
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.max_length = max_length

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(self.smiles))

        self.pair_indices = [(int(shuffled[i]), int(shuffled[i + 1]))
                             for i in range(0, len(shuffled) - 1, 2)]

        # odd leftover
        if len(self.smiles) % 2 != 0 and len(self.smiles) > 1:
            self.pair_indices.append((int(shuffled[-1]), int(rng.choice(shuffled[:-1]))))

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        iA, iB = self.pair_indices[idx]
        sA, sB = self.smiles[iA], self.smiles[iB]

        encA = self.tokenizer(
            sA,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encB = self.tokenizer(
            sB,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids_A": encA["input_ids"].squeeze(0),
            "attention_mask_A": encA["attention_mask"].squeeze(0),
            "input_ids_B": encB["input_ids"].squeeze(0),
            "attention_mask_B": encB["attention_mask"].squeeze(0),
        }


def parse_args():
    p = argparse.ArgumentParser(description="Interpolation APL on scaffold 5-fold validation split")
    p.add_argument("--load_model", type=str, required=True, help="Path to checkpoint (.pth)")
    p.add_argument(
        "--data_csv",
        type=str,
        default="/home/jin/Lim/Aidan/AR/admet_ai/output/22prop_reg_300k/data_scaffold_5fold.csv",
        help="CSV containing columns: smiles, fold, and label columns",
    )
    p.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name")
    p.add_argument("--fold_col", type=str, default="fold", help="Fold column name")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_alpha", type=int, default=51, help="Number of interpolation points in [0,1]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_pairs_limit", type=int, default=0, help="If >0, evaluate only this many pairs")
    p.add_argument("--out_path", type=str, default="", help="Optional: save per-pair APL to .npy or .csv")
    return p.parse_args()


def main():
    args = parse_args()

    load_model_path = os.path.abspath(args.load_model)
    if not os.path.exists(load_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_model_path}")

    base_model_name, fold, out_dir_name = infer_base_model_and_fold_from_checkpoint_path(load_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CustomRegModel(
        base_model_name,
        num_classes=22,
        freeze_encoder=False,
        load_model=load_model_path,
        retrain=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load scaffold 5fold CSV
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"data_csv not found: {args.data_csv}")

    df = pd.read_csv(args.data_csv)

    for col in [args.smiles_col, args.fold_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {args.data_csv}. Available columns: {list(df.columns)}"
            )

    # Select validation set by inferred fold
    val_df = df[df[args.fold_col].astype(int) == fold].reset_index(drop=True)
    if len(val_df) == 0:
        raise ValueError(f"Validation set is empty for fold={fold}. Check '{args.fold_col}' values in CSV.")

    smiles_list = val_df[args.smiles_col].astype(str).tolist()

    dataset = APLSMILESDataset(tokenizer, smiles_list, max_length=args.max_length, seed=args.seed)
    if args.num_pairs_limit and args.num_pairs_limit > 0:
        dataset.pair_indices = dataset.pair_indices[: args.num_pairs_limit]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    alpha_values = np.linspace(0.0, 1.0, args.num_alpha)

    all_apl_values = []

    for batch in tqdm(dataloader, desc=f"Computing APL (fold={fold})", bar_format="{l_bar}{bar:20}{r_bar}"):
        inputs_A = {k.replace("_A", ""): v.to(device) for k, v in batch.items() if k.endswith("_A")}
        inputs_B = {k.replace("_B", ""): v.to(device) for k, v in batch.items() if k.endswith("_B")}

        apl = calculate_PPL(model, inputs_A, inputs_B, alpha_values, device)
        all_apl_values.append(apl)

    all_apl_values = np.hstack(all_apl_values) if all_apl_values else np.array([])
    final_apl = float(np.mean(all_apl_values)) if all_apl_values.size else float("nan")

    print(f"[APL] output_dir={out_dir_name}")
    print(f"[APL] base_model={base_model_name}")
    print(f"[APL] inferred_fold={fold}")
    print(f"[APL] checkpoint={os.path.basename(load_model_path)}")
    print(f"[APL] num_pairs={all_apl_values.size}")
    print(f"[APL] final={final_apl:.6f}")

    if args.out_path:
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if out_path.lower().endswith(".npy"):
            np.save(out_path, all_apl_values)
        elif out_path.lower().endswith(".csv"):
            pd.DataFrame({"APL": all_apl_values}).to_csv(out_path, index=False)
        else:
            np.save(out_path + ".npy", all_apl_values)
        print(f"[APL] saved to: {out_path}")


if __name__ == "__main__":
    main()
