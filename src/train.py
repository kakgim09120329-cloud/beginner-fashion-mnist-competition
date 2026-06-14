# uv run src/train.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleCNN

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 20
HIDDEN_SIZE2 = 64
LEARNING_RATE = 0.05
BATCH_SIZE = 64
SEED = 42


def evaluate_accuracy(model: SimpleCNN, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = x.shape[0]
    with torch.no_grad():
        for start in range(0, total, batch_size):
            x_batch = x[start : start + batch_size].to(device)
            y_batch = y[start : start + batch_size].to(device)
            preds = model(x_batch)
            correct += int((preds.argmax(dim=1) == y_batch).sum().item())
    return float(correct) / float(total)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    x_train = torch.from_numpy(x_train).to(torch.float32)
    x_valid = torch.from_numpy(x_valid).to(torch.float32)
    t_train = torch.from_numpy(t_train).to(torch.long)
    t_valid = torch.from_numpy(t_valid).to(torch.long)

    model = SimpleCNN(
        NetworkConfig(
            in_channels=x_train.shape[1],
            hidden_size2=HIDDEN_SIZE2,
            output_size=10,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = float("-inf")
    best_state = None
    rng = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        indices = rng.permutation(x_train.shape[0])
        total_loss = 0.0

        for start in range(0, x_train.shape[0], BATCH_SIZE):
            batch_idx = indices[start : start + BATCH_SIZE]
            x_batch = x_train[batch_idx].to(device)
            y_batch = t_train[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_acc = evaluate_accuracy(model, x_train, t_train, BATCH_SIZE, device)
        valid_acc = evaluate_accuracy(model, x_valid, t_valid, BATCH_SIZE, device)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={total_loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = model.to_state()
            print(f"  Best model updated: valid_acc={best_valid_acc:.4f}")

    if best_state is None:
        best_state = model.to_state()

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(best_state, f)

    print(f"Saved best model: {OUTPUT_PATH.resolve()} (valid_acc={best_valid_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
