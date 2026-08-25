"""
The training loop: forward -> loss -> backward -> step -> zero gradients,
with validation monitoring, early stopping, best-model checkpointing,
optional gradient clipping, LR scheduling, and TensorBoard logging.
"""

import copy
import time

import numpy as np
import torch


def fit(model, train_loader, val_loader, *, epochs=100, lr=1e-3, patience=10,
        weight_decay=1e-5, grad_clip=None, use_scheduler=False,
        device=None, log_dir=None, verbose_every=10, criterion=None):
    """
    Train with early stopping on validation loss. Restores the best weights
    before returning. Returns a history dict.

    `criterion` defaults to MSELoss (regression); pass e.g. CrossEntropyLoss
    for classification.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion or torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3))
        if use_scheduler else None)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)                    # 1. forward
            loss = criterion(pred, y)          # 2. loss
            loss.backward()                    # 3. backward
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()                   # 4. update
            optimizer.zero_grad()              # 5. zero gradients
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_losses.append(criterion(model(X), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_loss)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}"
                  f"{'  *best*' if epoch == best_epoch else ''}")

        if epochs_without_improvement >= patience:
            print(f"  early stop at epoch {epoch} "
                  f"(best val {best_val:.6f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if writer is not None:
        writer.close()

    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    history["train_seconds"] = time.time() - start
    return history


@torch.no_grad()
def predict(model, loader, device=None):
    """Run inference over a DataLoader; returns (predictions, targets) arrays."""
    device = device or next(model.parameters()).device
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).ravel(), np.concatenate(targets).ravel()
