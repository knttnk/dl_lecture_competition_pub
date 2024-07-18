import os
import sys
import numpy as np
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
# 学習率スケジューラー
from torch.optim import lr_scheduler

from src.datasets import ThingsMEGDataset
from src.models import MyConvClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDataset(
        "train",
        args.data_dir,
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset(
        "val",
        args.data_dir,
    )
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset(
        "test",
        args.data_dir,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = MyConvClassifier(
        train_set.num_classes,
        train_set.seq_len,
        train_set.num_channels,
        hid_dim=128,
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_start,
        weight_decay=args.weight_decay,
        # amsgrad=False,
    )

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    # lr は args.lr_start から args.lr_end まで args.epochs で指数関数的に減少
    lr_sch = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (args.lr_end / args.lr_start) ** (epoch / args.epochs)
    )

    elapsed = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            subject_idxs = subject_idxs.to(args.device)
            # subject_idxs を one-hot に変換
            # subject_idxs = F.one_hot(subject_idxs, num_classes=4).to(args.device)

            y_pred = model(X, subject_idxs)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            subject_idxs = subject_idxs.to(args.device)

            with torch.no_grad():
                y_pred = model(X, subject_idxs)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            elapsed = 0
        else:
            elapsed += 1
            if elapsed >= 30:
                cprint("Early stopping.", "red")
                break

        lr_sch.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission_best"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    # ----------------------------------
    #  Start evaluation with the last model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission_last"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
