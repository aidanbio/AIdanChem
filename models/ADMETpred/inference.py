import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Loading multiple fold checkpoints in one process triggers PyTorch's legacy
# JIT profiling executor to fuse repeated pointwise ops (e.g. the DeBERTa
# attention-mask arithmetic) via NVRTC starting on the *second* model's
# forward call. On hosts missing libnvrtc-builtins for the installed CUDA
# version, that fusion attempt crashes with "nvrtc: failed to open
# libnvrtc-builtins.so". Disabling the profiling executor keeps everything
# in eager mode, which is what single-fold runs use implicitly (their first
# and only forward call never reaches the fusion path).
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

# NOTE: the previously imported `eval_finetuned_tdc_2` module does not exist in
# this repo. `CustomRegModel` below is copied verbatim from finetuning.py (the
# actual training script) so state_dict keys line up with saved checkpoints.

FEATURE_NAMES = [
    "Caco2", "HIA", "Pgp", "Bioavailability", "Lipophilicity",
    "Solubility", "BBB", "PPBR", "VDss", "CYP2C9", "CYP2D6",
    "CYP3A4", "CYP2C9_Substrate", "CYP2D6_Substrate", "CYP3A4_Substrate",
    "Half_Life", "Clearance_Hepatocyte", "Clearance_Microsome",
    "LD50", "hERG", "AMES", "DILI",
]

# BKCS 2026 (doi:10.1002/bkcs.70177) Table 1 model: lr=3e-5, focal MAE gamma=2.0,
# best epoch=15 selected by ABPS, averaged over the TDC scaffold 5-fold split.
# Reproducing that headline number at inference time means ensembling all 5
# fold checkpoints, not picking one.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLD_DIRS = [
    os.path.join(_THIS_DIR, "outputs", f"debeta_f{i}_lr3_g2_15ep") for i in range(5)
]


class CustomRegModel(nn.Module):
    """Must match finetuning.py's CustomRegModel exactly (state_dict compatibility)."""

    def __init__(self, base_model_name, num_classes=22):
        super().__init__()
        self.base = AutoModelForMaskedLM.from_pretrained(base_model_name)
        self.model_type = self.base.config.model_type
        self.regressor = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        if self.model_type == "bert":
            outputs = self.base.bert(input_ids=input_ids, attention_mask=attention_mask)
        elif self.model_type == "roberta":
            outputs = self.base.roberta(input_ids=input_ids, attention_mask=attention_mask)
        elif self.model_type == "deberta":
            outputs = self.base.deberta(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        mean_hidden = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.regressor(mean_hidden)


def _resolve_checkpoint(output_dir):
    """Pick the highest-epoch checkpoint_epoch_N.pth; fall back to best_model.pth."""
    epoch_ckpts = glob.glob(os.path.join(output_dir, "checkpoint_epoch_*.pth"))
    if epoch_ckpts:
        def _epoch_num(p):
            m = re.search(r"checkpoint_epoch_(\d+)\.pth$", p)
            return int(m.group(1)) if m else -1
        return max(epoch_ckpts, key=_epoch_num)

    best = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best):
        return best

    raise FileNotFoundError(f"No checkpoint (checkpoint_epoch_*.pth or best_model.pth) found in {output_dir}")


def _load_fold(output_dir, base_model_name, num_classes, device):
    """Load one fold's model + its label_mean_std.npz (name-indexed, not positional --
    the npz column order is the training CSV's column order, which is NOT the same
    order as FEATURE_NAMES)."""
    ckpt_path = _resolve_checkpoint(output_dir)
    mean_std_path = os.path.join(output_dir, "label_mean_std.npz")
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"label_mean_std.npz not found in {output_dir}")

    print(f"[*] Loading checkpoint: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = CustomRegModel(base_model_name, num_classes=num_classes)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    base_sd, reg_sd = {}, {}
    for key, val in state_dict.items():
        if key.startswith("base."):
            base_sd[key[len("base."):]] = val
        elif key.startswith("regressor."):
            reg_sd[key[len("regressor."):]] = val
        else:
            base_sd[key] = val
    model.base.load_state_dict(base_sd)
    model.regressor.load_state_dict(reg_sd)
    model.to(device).eval()

    npz = np.load(mean_std_path, allow_pickle=True)
    train_columns = [str(c) for c in npz["columns"]]
    mean_map = dict(zip(train_columns, npz["mean"].astype(np.float32)))
    std_map = dict(zip(train_columns, npz["std"].astype(np.float32)))

    return model, tokenizer, mean_map, std_map, train_columns


@torch.no_grad()
def _predict_raw(model, tokenizer, smiles_list, batch_size, max_length, device):
    """Raw (unnormalized) model output, in `train_columns` order (see _load_fold)."""
    all_preds = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        encoded = tokenizer(batch, max_length=max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        all_preds.append(model(input_ids, attention_mask).cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def run_inference(smiles_list, output_dir, base_model_name, num_classes=22,
                   batch_size=32, max_length=512, device=None):
    """
    Predict all 22 ADMET endpoints for `smiles_list`.

    `output_dir` may be a single fold directory (str) or a list of fold
    directories. When multiple fold directories are given, predictions are
    averaged across folds and the per-endpoint fold-to-fold standard
    deviation is reported as `<endpoint>_std` -- this is the only fold
    combination that reproduces the paper's Table 1 (which is itself a
    5-fold average), and it doubles as an applicability-domain-style
    uncertainty signal that a single fold's checkpoint cannot provide.
    """
    fold_dirs = [output_dir] if isinstance(output_dir, str) else list(output_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = None
    fold_preds = []  # each: (N, 22) denormalized, in FEATURE_NAMES order
    for fold_dir in fold_dirs:
        model, tok, mean_map, std_map, train_columns = _load_fold(
            fold_dir, base_model_name, num_classes, device
        )
        if tokenizer is None:
            tokenizer = tok

        raw = _predict_raw(model, tokenizer, smiles_list, batch_size, max_length, device)

        denorm_by_name = {}
        for i, col in enumerate(train_columns):
            denorm_by_name[col] = raw[:, i] * std_map[col] + mean_map[col]
        # Reorder to the canonical FEATURE_NAMES order for a stable output schema.
        fold_preds.append(np.stack([denorm_by_name[f] for f in FEATURE_NAMES], axis=1))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = np.stack(fold_preds, axis=0)  # (n_folds, N, 22)
    mean_preds = stacked.mean(axis=0)

    results_df = pd.DataFrame(mean_preds, columns=FEATURE_NAMES)
    if len(fold_dirs) > 1:
        std_preds = stacked.std(axis=0)
        for i, col in enumerate(FEATURE_NAMES):
            results_df[f"{col}_std"] = std_preds[:, i]
    results_df.insert(0, "SMILES", smiles_list)
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, help="Single SMILES string to predict")
    parser.add_argument("--file", type=str, help="Path to a text file containing SMILES (one per line)")
    parser.add_argument("--output_dir", type=str, nargs="+", default=DEFAULT_FOLD_DIRS,
                         help="One or more directories, each containing a checkpoint_epoch_*.pth "
                              "and label_mean_std.npz. Defaults to the 5-fold "
                              "debeta_f{0..4}_lr3_g2_15ep ensemble.")
    parser.add_argument("--base_model", type=str, default="sagawa/ZINC-deberta",
                         help="Base model name (must match the one used for training)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.smiles:
        input_smiles = [args.smiles]
    elif args.file:
        with open(args.file, "r") as f:
            input_smiles = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide --smiles or --file")
        return

    try:
        results = run_inference(
            smiles_list=input_smiles,
            output_dir=args.output_dir,
            base_model_name=args.base_model,
            batch_size=args.batch_size,
        )

        print("\n" + "=" * 30)
        print("ADMET Inference Results")
        print("=" * 30)
        print(results.to_string(index=False))

    except Exception as e:
        print(f"[-] Inference failed: {e}")


if __name__ == "__main__":
    main()
