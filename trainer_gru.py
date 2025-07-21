# %%
import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb

from source.utils import CharDataset
from source.model_gru import GRULanguageModel

# %%
wandb.init(project="tiny-llm", name="gru-run")

# %%
def train():
    dataset = CharDataset("data/input.txt", block_size=128)
    model = GRULanguageModel(vocab_size=dataset.tokenizer.vocab_size)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(2000):
        x, y = dataset.get_batch()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            wandb.log({"loss": loss.item(), "step": step})
            print(f"[GRU:{step}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/gru_model.pt")
    print("✅ GRU model saved to models/gru_model.pt")

# %%
if __name__ == "__main__":
    train()
