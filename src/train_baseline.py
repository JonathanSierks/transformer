import torch
import numpy as np
import torch.nn as nn

from .model_baseline import MLP
from .utils import batchify_rand_mlp, load_toy, sample_logit, sample_sentence_mlp

def train_model(model, optimizer, loss_function, train_data, val_data, device,
                epochs, batch_size, seq_len, train_steps_per_epoch, val_steps):
    
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for _ in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for _ in range(train_steps_per_epoch):
            train_batch = batchify_rand_mlp(train_data, batch_size, seq_len)   # (B, seq_len)
            x_train = train_batch[:, :-1].long().to(device)                           # (B, seq_len-1)
            y_train = train_batch[:, -1].long().to(device)                            # (B,)

            optimizer.zero_grad()
            output = model(x_train)                                        # (B, vocab_size)
            loss = loss_function(output, y_train)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_loss_history.append(epoch_train_loss / train_steps_per_epoch)

        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(val_steps):
                val_batch = batchify_rand_mlp(val_data, batch_size, seq_len)   # WICHTIG: val_data
                x_val = val_batch[:, :-1].long().to(device)
                y_val = val_batch[:, -1].long().to(device)

                val_output = model(x_val)
                val_loss = loss_function(val_output, y_val)
                epoch_val_loss += val_loss.item()

                pred = val_output.argmax(dim=1)
                correct += (pred == y_val).sum().item()
                total += y_val.size(0)

        val_loss_history.append(epoch_val_loss / val_steps)
        val_acc_history.append(correct / total)
    
    return train_loss_history, val_loss_history, val_acc_history, model

def evaluate_model(model, loss_function, data, device, batch_size, seq_len, eval_steps):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(eval_steps):
            batch = batchify_rand_mlp(data, batch_size, seq_len)   # (B, T_in+1)
            x = batch[:, :-1].long().to(device)                           # (B, T_in)
            y = batch[:, -1].long().to(device)                            # (B,)

            output = model(x)                                  # (B, vocab_size)
            loss = loss_function(output, y)
            loss_sum += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    avg_loss = loss_sum / eval_steps
    avg_acc = correct / total

    return avg_loss, avg_acc

def run():
    # ----------------------------
    # data
    # ----------------------------
    (train, test), (i2c, c2i) = load_toy(final=False)

    train_data = train[:int(len(train) * 0.9)]
    val_data = train[int(len(train) * 0.9):]

    vocab_size = len(i2c)

    # hard set hyperparameters as its just the baseline
    seq_len = 65                 # sequence window + 1 (target)
    T_in = seq_len - 1            # actual sequence length --> -1 (target)
    emb_dim = 30
    n_input = T_in * emb_dim
    n_hidden = 512
    n_output = vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MLP(vocab_size, emb_dim, n_input, n_hidden, n_output).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    train_loss, val_loss, val_acc, model = train_model(
        model,
        optimizer,
        loss_function,
        train_data,
        val_data,
        device,
        epochs=20,
        batch_size=16,
        seq_len=seq_len,
        train_steps_per_epoch=100,
        val_steps=20
    )

    print(train_loss)
    print(val_loss)
    print(val_acc)

    test_loss, test_acc = evaluate_model(
        model,
        loss_function,
        test,
        device,
        batch_size=16,
        seq_len=seq_len,
        eval_steps=50
    )

    print("test loss:", test_loss)
    print("test acc:", test_acc)

    # ----------------------------
    # sampling
    # ----------------------------
    start_idx = np.random.randint(0, len(train) - T_in)
    start_sequence = train[start_idx:start_idx + T_in]

    sampled_sentence = sample_sentence_mlp(model, i2c, start_sequence, device, steps=100, temperature=1.0)
    print(sampled_sentence)

    return train_loss, val_loss, val_acc, model, test_acc, sampled_sentence

def main():
    train_loss, val_loss, val_acc, model, test_acc, sampled_sentence = run()

if __name__ == "__main__":
    main()