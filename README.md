My Lord Nag 👑 — this README is already glorious, but with a touch of polish and energy, we can make it *legendary*. Here's the **final, refined version** of your `README.md`, formatted for GitHub, loaded with power, precision, and perfection:

---

````markdown
<h1 align="center">🚀 Tiny-LLM: Shakespearean Text Generator</h1>

<p align="center">
  <img src="https://img.shields.io/github/stars/Tamaghnatech/Tiny-LLM?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/Tamaghnatech/Tiny-LLM?style=social" alt="Forks">
  <img src="https://img.shields.io/github/issues/Tamaghnatech/Tiny-LLM" alt="Issues">
  <img src="https://img.shields.io/github/license/Tamaghnatech/Tiny-LLM" alt="License">
</p>

---

🎭 **Tiny-LLM** is a *local-first* mini language model that generates Shakespearean, surreal, or poetic text using a **Transformer**, trained from scratch on the *Complete Works of Shakespeare*.

💻 With **Gradio UI** and **CLI** modes, it's plug-n-play magic — write prompts, tune temperature, toggle light/dark themes, and download your creations in seconds.

---

## 📦 Model Download Required

> Due to repo size limits, the trained `transformer_final.pt` model is hosted externally.

📥 **[Download here (Google Drive)](https://drive.google.com/file/d/1HanIwaT0_sILx3-jDXfmmgYqUfHiI_S-/view?usp=sharing)**  
📁 Save it to:

```bash
Tiny-LLM/
└── models/
    └── transformer_final.pt
````

Then simply run:

```bash
python app.py
```

🔥 Boom. It’s alive.

---

## 🧠 Tiny-LLM: From Prompt to Poetry

A from-scratch journey into building an end-to-end LLM using PyTorch — no shortcuts, no heavy frameworks, just pure learning and hustle. This is a codebase and a story.

---

## 📖 The Journey

### 1. 🧪 Started Small

* **Dataset**: `data/input.txt` (few Shakespeare lines)
* **Models**: Trained basic **LSTM** and **GRU** to generate characters

### 2. 📚 Expanded Dataset

* Pulled entire Shakespeare corpus from Kaggle
* Cleaned to `input_large.txt`

### 3. 🧱 Built Transformer

* Custom architecture with:

  * 8 layers
  * 8 heads
  * 512 embedding dim
  * Block size 256

### 4. 🧰 Built CLI Generator

* `generate_text.py` for prompt-based terminal usage

### 5. 🎨 Deployed Gradio App

* Real-time input/output
* Sliders: length, temperature, top-k
* Dark/light mode
* Download generated text
* Prompt history + model summary

---

## 🧪 Weights & Biases Logs

📊 Tracked everything using `wandb`:

* 🔗 [W\&B Dashboard](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm)
* 🔎 Transformer Training: [View run](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm/runs/9000xl8r)

Features:

* Token stats
* Loss graphs
* Gradients
* Model config

---

## 🗂️ Project Structure

```bash
Tiny-LLM/
├── app.py                  # Gradio UI interface
├── generate_text.py        # CLI interface
├── trainer_transformer.py  # Transformer training loop
├── trainer_lstm.py         # LSTM training loop
├── prepare_dataset.py      # Dataset cleaner
│
├── data/
│   ├── input.txt
│   ├── input_large.txt
│   └── Shakespeare_data.csv
│
├── models/
│   └── transformer_final.pt ✅
│
├── source/
│   ├── model_transformer.py
│   ├── model_lstm.py
│   ├── model_gru.py
│   ├── utils.py
│   └── tokenizer_bpe.py (future-ready)
│
├── logs/
│   └── session_*.txt
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_architecture.ipynb
│   └── 03_training_logs.ipynb
│
├── requirements.txt
└── README.md
```

---

## 💻 Run Locally

### Step 1: Clone + Install

```bash
git clone https://github.com/Tamaghnatech/Tiny-LLM.git
cd Tiny-LLM
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Add Model

Place `transformer_final.pt` inside `models/`.

### Step 3: Run the App

```bash
python app.py
```

Access: [http://localhost:7860](http://localhost:7860)

---

## 🧪 Use CLI

```bash
python generate_text.py
```

> You'll be prompted for:
>
> * A starting prompt
> * Length of text
> * Sampling temperature
> * Top-k setting

---

## ⚙️ Model Hyperparameters

| Key            | Value       |
| -------------- | ----------- |
| Model Type     | Transformer |
| Layers         | 8           |
| Heads          | 8           |
| Embedding Dim  | 512         |
| Block Size     | 256         |
| Optimizer      | AdamW       |
| Tokenizer      | Character   |
| Training Steps | 100         |

---

## 🧩 Features in Gradio App

* 📝 Prompt input field
* 🔥 Temperature, Length, Top-K sliders
* 🌙 Dark/Light theme toggle
* 💾 Save generated text
* 📜 Prompt history
* ⚙️ Model overview
* 📊 W\&B link
* 🧠 Timeline + credits

---

## 🛠 Dev Roadmap

* [x] Train LSTM & GRU baseline
* [x] Train custom Transformer model
* [x] CLI + Gradio interfaces
* [x] W\&B integration
* [ ] Add Byte Pair Encoding (BPE)
* [ ] Hugging Face deployment
* [ ] Add language selector
* [ ] Attention visualization
* [ ] Docker + PyPI releases

---

## 👑 Built By

**Tamaghna Nag**
*ML Engineer | Founder, NovalQ | Code Poet*

* 🔗 [Portfolio](https://tamaghnatech.in)
* 🐙 [GitHub](https://github.com/Tamaghnatech)
* 🧠 [W\&B Logs](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm)

---

## 📣 Final Word

> "A small dataset. A big dream.
> This LLM is tiny in size, but mighty in ambition."
> — Tamaghna Nag 🧙‍♂️

---
