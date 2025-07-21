# app.py

from datetime import datetime
import torch
import gradio as gr
from source.model_transformer import TransformerLanguageModel
from source.utils import CharTokenizer
import os

# Load tokenizer and model
with open("data/input_large.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

tokenizer = CharTokenizer(text_data)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=256,
    embed_dim=512,
    n_layers=8,
    n_heads=8
).to(DEVICE)

model.load_state_dict(torch.load("models/transformer_final.pt", map_location=DEVICE))
model.eval()

# Session log
os.makedirs("logs", exist_ok=True)
session_path = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(session_path, "w", encoding="utf-8") as f:
    f.write("🧠 Tiny LLM Interactive Session Log\n")
    f.write(f"📅 Started: {datetime.now()}\n")
    f.write("=" * 50 + "\n")

@torch.no_grad()
def generate_text(prompt, length, temperature, top_k, history):
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(length):
        input_ids = context[:, -256:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

    output_text = tokenizer.decode(context.squeeze().tolist())
    timestamp = datetime.now().strftime('%H:%M:%S')
    history.append(f"[{timestamp}] Prompt: {prompt} → {length} chars")
    with open(session_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Prompt: {prompt}\n{output_text}\n")
    return output_text, history, session_path

# 🔧 Interface setup
description_md = """
## 🤖 Tiny-LLM: Transformer Text Generator
Generate Shakespearean/AI-speak text using a Transformer model trained **from scratch**!

> Type in a prompt, play with creativity, and watch your words evolve!

- 🔹 Prompt → What should the model start with?
- 🔢 Length → How much should it write?
- 🌡️ Temperature → Lower = logical, Higher = wild 🎭
- 🎯 Top-k → Controls randomness (0 = disabled)

⚠️ Everything runs **locally** and uses **your GPU** if available.
"""

timeline_md = """
### 🛣️ Project Timeline
| Stage | Description |
|-------|-------------|
| ✅ | Preprocessing, Tokenizer Design |
| ✅ | Transformer Architecture |
| ✅ | 100-Step Training on Custom Text |
| ✅ | CLI Text Generator |
| ✅ | WandB Logging |
| ✅ | Gradio App (You're here!) |
| 🔜 | BPE Tokenizer Upgrade |
| 🔜 | HuggingFace Deployment |
| 🔜 | SHAP / Attention Visualizations |
| 🔜 | Real-time Streaming UI |
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(description_md)
    gr.Markdown(timeline_md)

    with gr.Accordion("📊 Model Stats", open=False):
        gr.Markdown(f"""
        - **Layers**: 8  
        - **Embed Dim**: 512  
        - **Heads**: 8  
        - **Block Size**: 256  
        - **Vocab Size**: {tokenizer.vocab_size}  
        - **Device**: `{DEVICE}`  
        """)

    with gr.Row():
        prompt = gr.Textbox(label="🔹 Prompt", placeholder="Once upon a time...")
        length = gr.Slider(label="🔢 Length", minimum=50, maximum=1000, step=50, value=300)

    with gr.Row():
        temperature = gr.Slider(label="🌡️ Temperature", minimum=0.5, maximum=1.5, step=0.1, value=1.0)
        top_k = gr.Slider(label="🎯 Top-k Sampling", minimum=0, maximum=100, step=5, value=40)

    btn = gr.Button("🚀 Generate Text")
    output = gr.Textbox(label="📝 Generated Text", lines=12, max_lines=30)
    history_box = gr.HighlightedText(label="🕘 History (Prompts Used)")
    log_file = gr.File(label="📥 Download Full Session Log")

    state = gr.State([])

    btn.click(fn=generate_text, inputs=[prompt, length, temperature, top_k, state],
              outputs=[output, state, log_file])

demo.launch()
