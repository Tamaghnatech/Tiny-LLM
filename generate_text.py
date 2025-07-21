import torch
from source.utils import CharTokenizer
from source.model_transformer import TransformerLanguageModel

# 🚀 Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# 📖 Load data for tokenizer
with open("data/input_large.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# ⚙️ Model parameters (must match training)
model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=256,
    embed_dim=512,
    n_layers=8,
    n_heads=8
).to(DEVICE)

# 🎯 Load checkpoint
try:
    checkpoint_path = "models/transformer_final.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model from: {checkpoint_path}")
except Exception as e:
    print(f"❌ Failed to load checkpoint: {e}")
    exit(1)

# 🎲 Sampling function
@torch.no_grad()
def generate(model, context, length, temperature=1.0, top_k=0):
    for _ in range(length):
        input_ids = context[:, -256:]  # use block size
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, [-1]]] = -float('Inf')

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

    return context.squeeze().tolist()

# 🧠 Interactive CLI
print("\n💬 Welcome to Tiny LLM CLI! Type your prompt below.")
print("Type `exit` or `quit` to stop.\n")

while True:
    user_input = input("🔹 Prompt: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting. Until next time, Lord Nag!")
        break

    try:
        length = int(input("🔢 Length (e.g. 300): ") or "300")
        temperature = float(input("🌡️ Temperature (e.g. 1.0): ") or "1.0")
        top_k = int(input("🎯 Top-k (0 = disabled): ") or "0")
    except ValueError:
        print("⚠️ Invalid input. Try again.")
        continue

    encoded = tokenizer.encode(user_input)
    context = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)
    output_ids = generate(model, context, length, temperature, top_k)
    generated_text = tokenizer.decode(output_ids)

    print("\n📝 Generated Text:\n")
    print(generated_text)
    print("\n" + "-"*60 + "\n")
