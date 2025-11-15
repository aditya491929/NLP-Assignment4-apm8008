import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0


def get_args():
    """
    Arguments for training. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Whether to finetune T5 or not (otherwise train from scratch)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="AdamW",
        choices=["AdamW"],
        help="What optimizer to use",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Base learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW"
    )

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Whether to use a LR scheduler and what type to use if so",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=1,
        help="How many epochs to warm up the learning rate for if using a scheduler",
    )
    parser.add_argument(
        "--max_n_epochs",
        type=int,
        default=15,
        help="How many epochs to train the model for",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=3,
        help=(
            "If validation performance stops improving, how many epochs should we wait "
            "before stopping?"
        ),
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, we will use wandb to keep track of experiments",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="ft_experiment",
        help="How should we name this experiment?",
    )

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join(
        "checkpoints", f"{model_type}_experiments", args.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    experiment_name = "ft_experiment"
    gt_sql_path = os.path.join("data", "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_dev.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_dev.pkl"
    )

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
        )
        print(
            f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, "
            f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, "
            f"SQL EM: {sql_em:.4f}"
        )
        print(
            f"Epoch {epoch}: {error_rate * 100:.2f}% of the generated "
            f"outputs led to SQL errors"
        )

        if args.use_wandb:
            result_dict = {
                "train/loss": tr_loss,
                "dev/loss": eval_loss,
                "dev/record_f1": record_f1,
                "dev/record_em": record_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": error_rate,
            }
            wandb.log(result_dict, step=epoch)

        # Track best F1
        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Always save "last" and conditionally save "best"
        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print("Early stopping triggered.")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for (
        encoder_input,
        encoder_mask,
        decoder_input,
        decoder_targets,
        _,
    ) in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )
        logits = outputs["logits"]  # (B, T, V)

        non_pad = decoder_targets != PAD_IDX  # (B, T)
        # reshape for CE: N x V vs N
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def eval_epoch(
    args,
    model,
    dev_loader,
    gt_sql_pth,
    model_sql_path,
    gt_record_path,
    model_record_path,
):
    """
    Evaluation loop used during training.

    We compute:
      - Average cross-entropy loss on decoder targets
      - Record F1, Record EM, SQL EM via compute_metrics
      - SQL syntax error rate via compute_metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    all_pred_sql = []

    with torch.no_grad():
        for (
            encoder_input,
            encoder_mask,
            decoder_input,
            decoder_targets,
            _,
        ) in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generation (beam search)
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                decoder_start_token_id=bos_id,
            )
            preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            all_pred_sql.extend(preds)

    avg_loss = total_loss / max(total_tokens, 1)

    # Save predicted SQL and their DB records
    save_queries_and_records(all_pred_sql, model_sql_path, model_record_path)

    # Compute metrics against ground truth
    record_f1, record_em, sql_em, error_info = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    if isinstance(error_info, (list, tuple)):
        total = len(error_info)
        if total > 0:
            # count entries that are non-empty / non-None (i.e., had an error)
            num_errors = sum(1 for e in error_info if e)
            error_rate = num_errors / float(total)
        else:
            error_rate = 0.0
    else:
        # already a scalar
        error_rate = float(error_info)

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Inference on the held-out test set: generate SQL queries and associated records.
    """
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    all_pred_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(
            test_loader, desc="Test inference"
        ):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                decoder_start_token_id=bos_id,
            )
            preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            all_pred_sql.extend(preds)

    # Write SQL + DB records for Gradescope submission
    save_queries_and_records(all_pred_sql, model_sql_path, model_record_path)


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate with the best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"

    # Dev set
    gt_sql_path = os.path.join("data", "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_dev.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_dev.pkl"
    )

    (
        dev_loss,
        dev_record_f1,
        dev_record_em,
        dev_sql_em,
        dev_error_rate,
    ) = eval_epoch(
        args,
        model,
        dev_loader,
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    print(
        f"Dev set results: Loss: {dev_loss:.4f}, "
        f"Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, "
        f"SQL EM: {dev_sql_em:.4f}"
    )
    print(
        f"Dev set results: {dev_error_rate * 100:.2f}% of the generated "
        f"outputs led to SQL errors"
    )

    # Test set
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_test.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_test.pkl"
    )
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()