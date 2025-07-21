# %% Imports
import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
    wandb.init(project="tiny-llm", name="transformer-long-run")
except Exception as e:
    print(f"⚠️ wandb not available or failed to initialize: {e}")
    wandb = None

from source.utils import CharDataset
from source.model_transformer import TransformerLanguageModel

# %% Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {DEVICE}")

BLOCK_SIZE = 256
EMBED_DIM = 512
N_LAYERS = 8
N_HEADS = 8
BATCH_SIZE = 64              # 🔽 Reduced for faster test
LEARNING_RATE = 3e-4
TOTAL_STEPS = 100            # 🧪 Test run
SAVE_EVERY = 50
MODEL_DIR = "models"
FINAL_PATH = os.path.join(MODEL_DIR, "transformer_final.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

# %% Load Dataset
dataset = CharDataset("data/input_large.txt", block_size=BLOCK_SIZE)

# %% Initialize Model
vocab_size = dataset.tokenizer.vocab_size
model = TransformerLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    embed_dim=EMBED_DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS
).to(DEVICE)

# %% Optimizer, Scheduler, Scaler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# %% Load latest checkpoint if available
def load_checkpoint():
    latest_ckpt = None
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
            latest_ckpt = max(latest_ckpt or "", fname)
    
    if latest_ckpt:
        ckpt_path = os.path.join(MODEL_DIR, latest_ckpt)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"📦 Resumed training from: {latest_ckpt}")
    else:
        print("🚫 No checkpoint found. Starting from scratch.")

# %% Train Loop
def train():
    model.train()
    load_checkpoint()

    for step in range(1, TOTAL_STEPS + 1):
        try:
            print(f"🟢 Step {step} running...")  # 🔄 Status update before processing

            x, y = dataset.get_batch(batch_size=BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                logits, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            print(f"[Step {step}] Loss: {loss.item():.4f}")
            if wandb:
                wandb.log({
                    "loss": loss.item(),
                    "step": step,
                    "learning_rate": scheduler.get_last_lr()[0]
                })

            if step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ Saved checkpoint at: {ckpt_path}")

        except KeyboardInterrupt:
            print("🛑 Interrupted. Saving final model...")
            torch.save(model.state_dict(), FINAL_PATH)
            break
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            continue

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"🎯 Final model saved to: {FINAL_PATH}")

# %% Entry Point
if __name__ == "__main__":
    train()
