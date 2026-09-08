import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import numpy as np
import argparse
import itertools

import torch
from transformers import AutoModelForMaskedLM

torch.manual_seed(42)

class CustomRegModel(nn.Module):
    def __init__(self, base_model_name, num_classes=22, freeze_encoder=False, load_model=None, retrain=False):
        super(CustomRegModel, self).__init__()
        # Load the pre-trained BERT model for Masked Language Modeling
        self.base = AutoModelForMaskedLM.from_pretrained(base_model_name)
        self.model_type = self.base.config.model_type
        # Regressor head for multi-label
        self.regressor = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        if load_model is not None:
            checkpoint_state_dict = torch.load(load_model, map_location="cuda")['model_state_dict']
            base_state_dict, reg_state_dict = {}, {}
            for key, value in checkpoint_state_dict.items():
                if key.startswith("base."):        # "base.deberta." → "deberta."
                    new_key = key.replace("base.", "", 1)
                    base_state_dict[new_key] = value
                elif key.startswith("regressor"):  # "regressor.~~" 부분은 제외
                    new_key = key.replace("regressor.", "", 1)
                    reg_state_dict[new_key] = value
                else:
                    base_state_dict[key] = value
            self.base.load_state_dict(base_state_dict)
            if retrain:
                self.regressor.load_state_dict(reg_state_dict)
            print(f"Model loaded from {load_model}")

        if freeze_encoder:
            for param in self.base.parameters():
                param.requires_grad = False

            if self.model_type == "bert" or self.model_type == "deberta":
                # Freeze cls.predictions parameters (MaskedLM-specific)
                for param in self.base.cls.parameters():
                    param.requires_grad = True
            elif self.model_type == 'roberta':
                # Freeze cls.predictions parameters (MaskedLM-specific)
                for param in self.base.lm_head.parameters():
                    param.requires_grad = True
        else:
            for param in self.base.parameters():
                param.requires_grad = True

            if self.model_type == "bert" or self.model_type == "deberta":
                # Freeze cls.predictions parameters (MaskedLM-specific)
                for param in self.base.cls.parameters():
                    param.requires_grad = False
            elif self.model_type == 'roberta':
                # Freeze cls.predictions parameters (MaskedLM-specific)
                for param in self.base.lm_head.parameters():
                    param.requires_grad = False


    def forward(self, input_ids, attention_mask, **kwargs):
        return_mlm = kwargs.pop("return_mlm", False)
        if return_mlm:
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return out.logits

        # Forward pass through BERT
        if self.model_type == "bert":
            outputs = self.base.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        elif self.model_type == 'roberta':
            outputs = self.base.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        elif self.model_type == 'deberta':
            outputs = self.base.deberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        # Calculate the mean of all token embeddings (excluding padding)
        masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)  # Mask padding tokens
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)  # Count valid tokens
        mean_hidden_state = masked_hidden_state.sum(dim=1) / valid_tokens  # Mean over valid tokens

        # Pass through the classifier head
        return self.regressor(mean_hidden_state)

    def extract_rep(self, input_ids, attention_mask, **kwargs):
        return_seq = kwargs.pop("return_seq", False)
        # Forward pass through BERT
        if self.model_type == "bert":
            outputs = self.base.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        elif self.model_type == 'roberta':
            outputs = self.base.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        elif self.model_type == 'deberta':
            outputs = self.base.deberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        if return_seq:
            return last_hidden_state

        # Calculate the mean of all token embeddings (excluding padding)
        masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)  # Mask padding tokens
        valid_tokens = attention_mask.sum(dim=1, keepdim=True)  # Count valid tokens
        mean_hidden_state = masked_hidden_state.sum(dim=1) / valid_tokens  # Mean over valid tokens

        # Pass through the classifier head
        return mean_hidden_state


class SMILESDataset(Dataset):
    def __init__(self, tokenizer, smiles, labels, mean, std, max_length=512):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = (labels - mean) / std

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "smiles": self.smiles[idx],
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

class FocalMAELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalMAELoss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, targets):
        loss = torch.abs(preds - targets)  # MAE 계산
        scaling_factor = 1 - torch.exp(-self.gamma * loss)  # 오차가 클수록 작은 값
        return (scaling_factor * loss).mean()  # Focal Scaling 적용


# MLM Loss 계산 함수 추가
def calculate_mlm_loss(model, tokenizer, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0  # 정확히 맞춘 토큰 수
    total_masked_tokens = 0  # 마스킹된 토큰 수

    with torch.no_grad():
        # for batch in tqdm(data_loader, desc="MLM steps", bar_format="{l_bar}{bar:20}{r_bar}"):
        for batch in itertools.islice(data_loader, 10):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()

            # Randomly token masking
            probability_matrix = torch.full(labels.shape, 0.15, device=device)  # 15% Masking
            mask_token_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[mask_token_indices] = tokenizer.mask_token_id  # [MASK] token

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask, return_mlm=True)

            # MLM Loss (only Masked token for Loss)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # MLM Accuracy
            predictions = logits.argmax(dim=-1)  # 가장 확률이 높은 토큰 선택
            correct_predictions += (predictions[mask_token_indices] == labels[mask_token_indices]).sum().item()
            total_masked_tokens += mask_token_indices.sum().item()
            accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0

            # average Loss
            avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    return avg_loss, accuracy


def move_optimizer_to_device(optimizer, device):
    for param_group in optimizer.state.values():
        for param_name, param_tensor in param_group.items():
            if isinstance(param_tensor, torch.Tensor):
                param_group[param_name] = param_tensor.to(device)


# 명령줄 인자 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model for regression tasks without freezing parameters.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name (e.g., bert-base-uncased, roberta-base).")
    parser.add_argument("--output_dir", type=str, default=None)
    # parser.add_argument("--retrain", action='store_true', help="Retrain the model")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from checkpoint")
    parser.add_argument("--load_model", type=str, default=None, help="Load model for retraining")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal Loss gamma value")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--num_classes", type=int, default=22, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load data
    data = pd.read_csv("data/pubchem/data_scaffold_5fold.csv")

    mean_std_dict = {
        "mean": data.drop(columns=["smiles", "fold"]).mean(),
        "std": data.drop(columns=["smiles", "fold"]).std()
    }

    train_data = data[data['fold'] != args.fold].reset_index(drop=True)
    val_data = data[data['fold'] == args.fold].reset_index(drop=True)

    print(f"Selected Fold: {args.fold}")
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # Prepare datasets
    train_dataset = SMILESDataset(
        tokenizer,
        train_data["smiles"].tolist(),
        train_data.drop(columns=["smiles", "fold"]).values,
        mean_std_dict['mean'].values,
        mean_std_dict['std'].values
    )
    val_dataset = SMILESDataset(
        tokenizer,
        val_data["smiles"].tolist(),
        val_data.drop(columns=["smiles", "fold"]).values,
        mean_std_dict['mean'].values,
        mean_std_dict['std'].values
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Directory for saving outputs with timestamp
    output_base_dir = "outputs/"
    if args.output_dir:
        output_dir = os.path.join(output_base_dir, args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")
    log_exists = os.path.exists(log_file)
    np.savez(os.path.join(output_dir, "label_mean_std.npz"),
             mean=mean_std_dict["mean"].values,
             std=mean_std_dict["std"].values,
             columns=train_data.drop(columns=["smiles", "fold"]).columns.values)

    # Open log file
    if not log_exists:
        with open(log_file, "w") as log:
            log.write("========== Experiment Configuration ==========\n")
            # vars(args)는 인자 이름을 키(key)로, 설정값을 값(value)으로 하는 딕셔너리를 반환합니다.
            for arg, value in vars(args).items():
                log.write(f"{arg}: {value}\n")
            log.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("==============================================\n\n")
            log.write("Steps\tEpoch\tTrain Loss\tValidation Loss\tMLM Loss\tMLM accuracy\n")

    # Initialize model
    if args.load_model:
        retrain =True
        load_model = os.path.join(output_dir, args.load_model)
    else:
        retrain = False
        if args.base_model_name == "sagawa/ZINC-deberta":
            load_model = 'outputs/cls/deberta_cls.pth'
        elif args.base_model_name == "DeepChem/ChemBERTa-77M-MLM":
            load_model = 'outputs/cls/chemberta_cls.pth'
        elif args.base_model_name == "entropy/roberta_zinc_480m":
            load_model = 'outputs/cls/roberta_cls.pth'
        elif args.base_model_name == 'unikei/bert-base-smiles':
            load_model = 'outputs/cls/bert_cls.pth'
    # load_model = '/home/jin/Lim/Aidan/AR/admet_finetune/outputs/deberta_41prop_unfreez_acc/checkpoint_25.pth'
    model = CustomRegModel(args.base_model_name, num_classes=args.num_classes, freeze_encoder=False, load_model=load_model, retrain=retrain)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    if args.focal_gamma:
        loss_fn = FocalMAELoss(gamma=args.focal_gamma)
    else:
        loss_fn = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    eval_steps = 500
    best_val_loss = float("inf")
    num_epochs = args.num_epochs
    start_epoch = 0
    if args.resume and args.load_model:
        chk = torch.load(load_model, map_location=device)
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", start_epoch)
        global_step = chk.get("global_step", global_step)

        # best_chk = torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device)
        # best_val_loss = best_chk.get("val_loss", best_val_loss)

    # Initial MLM Loss (Before Fine-tuning)
    model.eval()
    initial_mlm_loss, initial_acc = calculate_mlm_loss(model, tokenizer, val_loader, device)
    print(f"Initial MLM Loss: {initial_mlm_loss:.4f} Initial MLM Accuracy: {initial_acc:.4f}")
    with open(log_file, "a") as log:
        log.write(f"0\t0\t-\t-\t{initial_mlm_loss:.4f}\t{initial_acc:.4f}\n")

    print("Training Starts")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", bar_format="{l_bar}{bar:20}{r_bar}")

        for batch in pbar:
            global_step += 1

            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(batch_input_ids, batch_attention_mask)
            loss = loss_fn(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step_loss = loss.item()
            total_train_loss += current_step_loss

            pbar.set_postfix({"Step_Loss": f"{current_step_loss:.4f}"})

            # Evaluation steps
            if global_step % eval_steps == 0:
                model.eval()
                total_val_loss = 0

                # 1. Regression validation loss
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Evaluation", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
                    for v_batch in val_pbar:
                        v_ids = v_batch["input_ids"].to(device)
                        v_mask = v_batch["attention_mask"].to(device)
                        v_labels = v_batch["labels"].to(device)

                        v_outputs = model(v_ids, v_mask)
                        v_loss = loss_fn(v_outputs, v_labels)
                        total_val_loss += v_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                avg_train_loss = total_train_loss / (
                    global_step % len(train_loader) if global_step % len(train_loader) != 0 else len(train_loader))

                # 2. MLM loss/acc
                avg_mlm_loss, avg_acc = calculate_mlm_loss(model, tokenizer, val_loader, device)

                print(
                    f"\n[Step {global_step}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MLM Acc: {avg_acc:.4f}")

                # 3. log
                with open(log_file, "a") as log:
                    # epoch_progress
                    epoch_progress = epoch + (global_step % len(train_loader)) / len(train_loader)
                    log.write(
                        f"{global_step}\t{epoch_progress:.2f}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{avg_mlm_loss:.4f}\t{avg_acc:.4f}\n")

                # 4. Best Model save based on Validation Loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, "best_model.pth")
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'mean': mean_std_dict["mean"].values,
                        'std': mean_std_dict["std"].values
                    }, best_model_path)
                    print(f"--- Best model saved at step {global_step} (Val Loss: {best_val_loss:.4f}) ---")

                # training mode
                model.train()

        # checkpoint save after every epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Epoch {epoch + 1} completed and checkpoint saved.")
    

