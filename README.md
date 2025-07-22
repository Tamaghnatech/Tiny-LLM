````markdown
<h1 align="center">🚀 Tiny-LLM: Shakespearean Text Generator</h1>

<p align="center">
  <img src="https://img.shields.io/github/stars/Tamaghnatech/Tiny-LLM?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/Tamaghnatech/Tiny-LLM?style=social" alt="Forks">
  <img src="https://img.shields.io/github/issues/Tamaghnatech/Tiny-LLM" alt="Issues">
  <img src="https://img.shields.io/github/license/Tamaghnatech/Tiny-LLM" alt="License">
</p>

---

🎭 **Tiny-LLM** is a *local-first* mini language model that generates Shakespearean, poetic, and surreal text using a **Transformer**, trained from scratch on the *Complete Works of Shakespeare*.

💻 With **Gradio UI** and **CLI** modes, it's plug-n-play magic — write prompts, tune temperature, toggle light/dark themes, and download your creations in seconds.

🌐 **Try it on Hugging Face Spaces**:  
🔗 [https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM](https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM)

---

## 📦 Model Download Required (for Local Use)

Due to repo size limits, the trained model isn't bundled here.

📥 **[Download transformer_final.pt from Google Drive](https://drive.google.com/file/d/1HanIwaT0_sILx3-jDXfmmgYqUfHiI_S-/view?usp=sharing)**  
📁 Save it in:

```bash
Tiny-LLM/
└── models/
    └── transformer_final.pt
````

---

## 🧠 Tiny-LLM: From Prompt to Poetry

> Built from scratch using PyTorch and love ❤️
> No Hugging Face pretrained models. No shortcuts. Just math, code, and Shakespeare.

---

## 📖 Project Evolution

### 1. 🧪 Started Small

* Dataset: `input.txt` — a small set of Shakespearean lines
* Trained LSTM and GRU models to generate characters

### 2. 📚 Dataset Expansion

* Downloaded the full *Shakespeare Plays Corpus* from Kaggle
* Cleaned and stored as `input_large.txt`

### 3. 🧱 Built Transformer from Scratch

* 8 layers, 8 heads, 512 embedding dim
* Block size: 256
* Trained on GPU (100 steps)

### 4. 📟 Created CLI Tool

* `generate_text.py` for terminal-based text generation

### 5. 🎨 Designed Gradio Web App

* Real-time generation with sliders for Temperature, Top-k, and Length
* Light/Dark mode, session logs, model insights

---

## 🧪 W\&B Dashboard

All training experiments were tracked using Weights & Biases:

🔗 [View Project Dashboard](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm)
📈 [Transformer Run](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm/runs/9000xl8r)

---

## 🗂 Project Structure

```bash
Tiny-LLM/
├── app.py                  # Gradio interface
├── generate_text.py        # CLI generator
├── trainer_transformer.py  # Transformer training loop
├── prepare_dataset.py      # Dataset cleaner
│
├── data/
│   ├── input.txt
│   ├── input_large.txt
│   └── Shakespeare_data.csv
│
├── models/
│   └── transformer_final.pt  ✅
│
├── source/
│   ├── model_transformer.py
│   ├── model_lstm.py
│   ├── model_gru.py
│   ├── utils.py
│   └── tokenizer_bpe.py (WIP)
│
├── logs/                  # Session outputs
├── notebooks/             # Jupyter breakdowns
├── requirements.txt
└── README.md
```

---

## ⚙️ Hyperparameters

| Name           | Value       |
| -------------- | ----------- |
| Model Type     | Transformer |
| Layers         | 8           |
| Heads          | 8           |
| Embedding Size | 512         |
| Block Size     | 256         |
| Tokenizer      | Char-level  |
| Optimizer      | AdamW       |
| Steps Trained  | 100         |

---

## 💻 Run Locally

```bash
git clone https://github.com/Tamaghnatech/Tiny-LLM.git
cd Tiny-LLM

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then, place the model inside `models/` and launch:

```bash
python app.py
```

---

## 🔥 Gradio Features

* 🔹 Prompt input box
* 🔢 Length slider (50–1000)
* 🌡️ Temperature control (creativity)
* 🎯 Top-k sampling
* 🌓 Light/Dark toggle
* 🧠 Model explanation
* 📜 Prompt history
* 💾 Downloadable output
* 🧭 Timeline + Project overview

---

## 🧪 Use from Terminal

```bash
python generate_text.py
```

You’ll be asked for:

* Prompt
* Length
* Temperature
* Top-k

Result prints in console.

---

## 🛠️ Dev Roadmap

* [x] Character-level LSTM & GRU models
* [x] Train Transformer from scratch
* [x] Gradio + CLI interfaces
* [x] Weights & Biases integration
* [x] Hugging Face demo deployment
* [ ] Byte Pair Encoding (BPE)
* [ ] Attention visualization
* [ ] Language toggle (Bengali, English, Hindi, German)
* [ ] Docker + PyPI support
* [ ] Long-context training

---

## 👨‍💻 Author

**Tamaghna Nag**
Founder of NovalQ | ML Engineer | Shakespeare Whisperer

* 🔗 [Portfolio](https://tamaghnatech.in)
* 🐙 [GitHub](https://github.com/Tamaghnatech)
* 📊 [W\&B Project](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm)

---

## 💬 Final Word

> “A tiny model built on timeless literature.
> Proof that even small things, when trained well, can sound divine.”

---

### 🪄 Fork It. Prompt It. Publish It.

🔥 Star the repo if you like the effort and [try the app on Hugging Face](https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM)!

```

