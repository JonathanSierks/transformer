import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"

import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

from .utils import load_config, load_toy, batchify_rand_transformer, sample_sentence_transformer
from .model import AutoRegressiveTransformer



def run_one_experiment(cfg, seed=42):
    
    # ---------------------------------
    # SET UP
    # ---------------------------------
    # set device and seeds
    requested_device = cfg["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # read in data
    (train, test), (i2c, c2i) = load_toy(final=False)
    train_data = train[:int(len(train) * 0.9)]
    val_data = train[int(len(train) * 0.9):]

    vocab_size = len(i2c)

    # define results object structure
    results = {
        "config": cfg,
        "seed": seed,
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "gradient_norms": [],
        "batch_loss": [],
        "test_loss": None,
        "test_acc": None,
        "best_epoch": None,
        "stopped_early": False,
        "sample_sentence": None
    }

    model = AutoRegressiveTransformer(
        vocab_size,
        cfg["num_heads"],
        cfg["num_Tblocks"],
        cfg["max_sequence_length"],
        cfg["embedding_dim"],
        cfg["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_loss = float("inf")
    best_epoch = None

    # early stopping setup
    patience = cfg.get("early_stopping_patience", 10)
    min_delta = cfg.get("early_stopping_min_delta", 0.0)
    epochs_without_improvement = 0
    best_model_path = cfg.get("best_model_path", "model_best.pt")

    # ---------------------------------
    # TRAINING
    # ---------------------------------
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        num_data_pts_train = 0

        # ---------------------------------
        # TRAINING
        # ---------------------------------
        model.train()
        for batch in range(cfg["n_batches"]):
            # create batch
            train_batch = batchify_rand_transformer(train_data, cfg["bsz"], cfg["T"])   # (B, T+1)
            x_train = train_batch[:, :-1].long().to(device)                 # (B, T)
            y_train = train_batch[:, 1:].long().to(device)                  # (B, T)

            optimizer.zero_grad()
            logits = model(x_train)                                         # (B, T, C)
            B, T, C = logits.size()

            logits_flat = logits.reshape(B * T, C)
            targets_flat = y_train.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)
            results["batch_loss"].append(loss.item())
            total_loss += loss.item() * targets_flat.size(0)
            num_data_pts_train += targets_flat.size(0)

            loss.backward()

            # compute gradient norms for easier detection of exploding/vanishing gradients
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            results["gradient_norms"].append(total_norm)

            optimizer.step()

        results["train_loss"].append(total_loss / num_data_pts_train)

        # ---------------------------------
        # VALIDATION
        # ---------------------------------
        model.eval()
        val_losses = []
        num_correct = 0
        num_data_pts_val = 0

        with torch.no_grad():
            for batch in range(cfg["n_batches"] // 4):   # run validation for 1/4 of the training loop amount
                val_batch = batchify_rand_transformer(val_data, cfg["bsz"], cfg["T"])   # WICHTIG: val_data
                x_val = val_batch[:, :-1].long().to(device)
                y_val = val_batch[:, 1:].long().to(device)

                # retrieve last logit and target to only compute loss and acc for the last token
                logits = model(x_val)                       # (B, T, C)
                last_logits = logits[:, -1, :]             # (B, C)
                last_targets = y_val[:, -1]                # (B,)

                # compute loss
                loss = F.cross_entropy(last_logits, last_targets)
                val_losses.append(loss.item())

                # retrieve prediction
                preds_last = last_logits.argmax(dim=-1)    # (B,)
                num_correct += (preds_last == last_targets).sum().item()
                num_data_pts_val += last_targets.size(0)

        # compute and store validation loss and accuracy
        avg_val_loss = sum(val_losses) / len(val_losses)
        results["val_loss"].append(avg_val_loss)

        val_acc = num_correct / num_data_pts_val
        results["val_acc"].append(val_acc)

        # early stopping / best model tracking
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        # debug & status per epoch
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"Train Loss: {results['train_loss'][-1]:.4f} | "
            f"Val Last-Token Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            results["stopped_early"] = True
            break

    results["best_epoch"] = best_epoch

    # load best model before final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ---------------------------------
    # RESULTS OUTPUT
    # ---------------------------------
    # output final results after training has finished
    test_losses = []
    num_correct = 0
    num_data_pts_test = 0

    with torch.no_grad():
        for batch in range(cfg["n_batches"] // 4):
            test_batch = batchify_rand_transformer(test, cfg["bsz"], cfg["T"])
            x_test = test_batch[:, :-1].long().to(device)
            y_test = test_batch[:, 1:].long().to(device)

            logits = model(x_test)
            last_logits = logits[:, -1, :]
            last_targets = y_test[:, -1]

            loss = F.cross_entropy(last_logits, last_targets)
            test_losses.append(loss.item())

            preds_last = last_logits.argmax(dim=-1)
            num_correct += (preds_last == last_targets).sum().item()
            num_data_pts_test += last_targets.size(0)

    results["test_loss"] = sum(test_losses) / len(test_losses)
    results["test_acc"] = num_correct / num_data_pts_test

    # sample a sentence
    start_idx = np.random.randint(0, len(train) - cfg["T"])
    start_sequence = train[start_idx:start_idx + cfg["T"]].to(device)
    results["sample_sentence"] = sample_sentence_transformer(
        model, i2c, start_sequence, device, steps=100, temperature=1.0
    )
    #print(f"Final test accuracy is: {results["test_acc"]}")
    #print(f"sampled output: {results["sample_sentence"]}")

    # save final/best model to model.pt file so we can reload it later
    torch.save(model.state_dict(), "model.pt")

    return results


# tba later
def run_many_experiments(cfg, seeds):
    results = []
    for seed in seeds:
        result = run_one_experiment(cfg, seed)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--setting", default="single")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    if args.setting == "single":
        results = run_one_experiment(cfg)
        print(f"Best epoch: {results['best_epoch']}")
        print(f"Test loss: {results['test_loss']:.4f}")
        print(f"Test acc: {results['test_acc']:.4f}")
        print(results["sample_sentence"])
    elif args.setting == "multiple":
        raise NotImplementedError("run_many_experiments not finalized yet.")
    else:
        raise ValueError(f"Unknown setting: {args.setting}")


if __name__ == "__main__":
    main()