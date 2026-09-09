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
from transformers import get_linear_schedule_with_warmup

class CustomRegModel(nn.Module):
    def __init__(self, base_model_name, num_classes=22, freeze_encoder=False, load_model=None):
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
                if key.startswith("base."):  # "base.deberta." → "deberta."
                    new_key = key.replace("base.", "", 1)
                    base_state_dict[new_key] = value
                elif key.startswith("regressor"):  # "regressor.~~" 부분은 제외
                    new_key = key.replace("regressor.", "", 1)
                    reg_state_dict[new_key] = value
                else:
                    base_state_dict[key] = value
            self.base.load_state_dict(base_state_dict)
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
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

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

            # 랜덤하게 일부 토큰을 마스킹
            probability_matrix = torch.full(labels.shape, 0.15, device=device)  # 15% Masking
            mask_token_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[mask_token_indices] = tokenizer.mask_token_id  # [MASK] 토큰 적용

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask, return_mlm=True)

            # MLM Loss 계산 (Mask된 토큰만 Loss에 포함)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Accuracy 계산 (Mask된 토큰이 정답을 맞춘 비율)
            predictions = logits.argmax(dim=-1)  # 가장 확률이 높은 토큰 선택
            correct_predictions += (predictions[mask_token_indices] == labels[mask_token_indices]).sum().item()
            total_masked_tokens += mask_token_indices.sum().item()

            # 평균 Loss 계산
            avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

            # Accuracy 계산
            accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0

    return avg_loss, accuracy


def move_optimizer_to_device(optimizer, device):
    for param_group in optimizer.state.values():
        for param_name, param_tensor in param_group.items():
            if isinstance(param_tensor, torch.Tensor):
                param_group[param_name] = param_tensor.to(device)


def train_cls_head(model, tokenizer, train_loader, val_loader, device, mlm_log, start_epoch=0, num_epochs=3, retrain_model=None):
    model.train()
    model_type = model.base.config.model_type
    if model_type == "bert" or model_type == "deberta":
        optimizer = AdamW(model.base.cls.parameters(), lr=5e-5)
    elif model_type == "roberta":
        optimizer = AdamW(model.base.lm_head.parameters(), lr=5e-5)

    # total_steps = total number of training steps
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 예: 앞 10% 구간은 warm-up
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")

    if retrain_model is not None:
        checkpoint = torch.load(retrain_model, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        move_optimizer_to_device(optimizer, device)
        best_val_acc = checkpoint['val_acc']
        best_val_loss = checkpoint['val_loss']

    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training cls.predictions", bar_format="{l_bar}{bar:20}{r_bar}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            # 랜덤하게 일부 토큰을 마스킹
            probability_matrix = torch.full(labels.shape, 0.15, device=device)  # 15% Masking
            mask_token_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[mask_token_indices] = tokenizer.mask_token_id  # [MASK] 토큰 적용

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask, return_mlm=True)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_acc = calculate_mlm_loss(model, tokenizer, val_loader, device)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f} Validation MLM Loss: {val_loss:.4f} Validation MLM Accuracy: {val_acc:.4f}")

        with open(mlm_log, "a") as log:
            log.write(f"{epoch+1}\t{avg_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")

        # Save checkpoint
        # if val_loss < best_val_loss:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, f"best_cls_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Best CLS model saved at epoch {epoch+1} with Validation accuracy: {val_acc:.4f}")

    print("Finished training cls.predictions!")




# 명령줄 인자 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model for regression tasks without freezing parameters.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name (e.g., bert-base-uncased, roberta-base).")
    # parser.add_argument("--unfrz_layers", type=int, required=True, help="Number of layers to unfreeze.")
    parser.add_argument("--load_model", type=str, default=None, help="Load model for retraining lm_head")
    parser.add_argument("--retrain", action="store_true", help="Retrain model")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load data
    train_data = pd.read_csv("data/pubchem/train.csv")
    val_data = pd.read_csv("data/pubchem/val.csv")

    mean_std_dict = {
        "mean": train_data.drop(columns=["smiles"]).mean(),
        "std": train_data.drop(columns=["smiles"]).std()
    }

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # Prepare datasets
    train_dataset = SMILESDataset(
        tokenizer,
        train_data["smiles"].tolist(),
        train_data.drop(columns=["smiles"]).values,
        mean_std_dict['mean'].values,
        mean_std_dict['std'].values
    )
    val_dataset = SMILESDataset(
        tokenizer,
        val_data["smiles"].tolist(),
        val_data.drop(columns=["smiles"]).values,
        mean_std_dict['mean'].values,
        mean_std_dict['std'].values
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    if args.load_model is not None:
        model = CustomRegModel(args.base_model_name, num_classes=22, freeze_encoder=True, load_model=args.load_model)
    else:
        model = CustomRegModel(args.base_model_name, num_classes=22, freeze_encoder=True)

    # Directory for saving outputs with timestamp
    output_base_dir = "outputs/"
    if args.output_dir:
        output_dir = os.path.join(output_base_dir, args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    # log_file = os.path.join(output_dir, "log.txt")
    mlm_log_file = os.path.join(output_dir, "mlm_log.txt")
    mlm_log_exists = os.path.exists(mlm_log_file)
    np.savez(os.path.join(output_dir, "label_mean_std.npz"), 
             mean=mean_std_dict["mean"].values, 
             std=mean_std_dict["std"].values, 
             columns=train_data.drop(columns=["smiles"]).columns.values)

    # Open log file
    if not mlm_log_exists:
        with open(mlm_log_file, "w") as log:
            log.write("Epoch\tTrain Loss\tValidation Loss\tValidation Accuracy\n")

    # Open log file
    # with open(log_file, "w") as log:
    #     log.write("Epoch\tTrain Loss\tValidation Loss\tMLM Loss\tMLM accuracy\n")

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0
    if args.retrain:
        start_epoch = torch.load(args.load_model, map_location="cuda")['epoch']

    initial_mlm_loss, initial_acc = calculate_mlm_loss(model, tokenizer, val_loader, device)
    print(f"Initial MLM Loss: {initial_mlm_loss:.4f} Initial MLM Accuracy: {initial_acc:.4f}")

    train_cls_head(model, tokenizer, train_loader, val_loader, device, mlm_log_file, start_epoch=start_epoch, num_epochs=args.num_epochs, retrain_model=args.load_model)
