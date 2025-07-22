# 🚀 Tiny-LLM: Shakespearean Text Generator

🎭 **Tiny-LLM** is a *local-first*, from-scratch **mini language model** trained entirely on the *Complete Works of Shakespeare*. It generates poetic, Shakespearean, and surreal text using a custom-built **Transformer architecture**.

💡 This repo demonstrates the full **Machine Learning to MLOps** lifecycle: data prep, modeling (LSTM, GRU, Transformer), training, experiment tracking, CLI + Gradio UI, and Hugging Face Spaces deployment.

🌐 **[👉 Try it LIVE on Hugging Face Spaces](https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM)**

📊 **[📈 See the W&B Dashboard](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm?nw=nwusernagtamaghna)** – Training logs, loss curves, model configs, gradient flow & more.

---

## 📦 Download the Trained Model (for Local Usage)

Due to GitHub's size restrictions, the trained Transformer model is hosted externally.

📥 **[Download `transformer_final.pt` from Google Drive](https://drive.google.com/file/d/1HanIwaT0_sILx3-jDXfmmgYqUfHiI_S-/view?usp=sharing)**

Save it to:

```bash
Tiny-LLM/
└── models/
    └── transformer_final.pt
```

---

## 🧠 End-to-End Pipeline Overview

> A full-stack, production-grade journey from dataset to deployment.

### 📊 1. Data Collection & Preparation
- Source: Kaggle Shakespeare Corpus
- Cleaned into character-level dataset → `input.txt` and `input_large.txt`
- Script: [`prepare_dataset.py`](prepare_dataset.py)

### 🧠 2. Model Development
- LSTM / GRU (baselines)
- Transformer (final)
  - 8 Layers, 8 Heads, 512 Embedding Dim, Block Size: 256
- Location: [`source/`](source/) directory

### 🏋️‍♂️ 3. Training
- Scripts: `trainer_lstm.py`, `trainer_gru.py`, `trainer_transformer.py`
- Tracked using Weights & Biases (`wandb`)
- Optimizer: `AdamW`
- Loss: CrossEntropy
- Checkpoints: `.pt` files saved to `models/`

### 📈 4. Experiment Tracking (MLOps)
- Dashboard:  
  🔗 [W&B Project](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm?nw=nwusernagtamaghna)

### 🧪 5. Inference & Serving

#### ➤ CLI Interface
- Script: `generate_text.py`

#### ➤ Gradio Web App
- Script: `app.py`

### 🚀 6. Deployment
- ✅ Gradio Localhost
- ✅ Hugging Face Spaces

---

## 📂 Project Structure

```bash
Tiny-LLM/
├── app.py
├── generate_text.py
├── trainer_transformer.py
├── prepare_dataset.py
│
├── data/
│   ├── input.txt
│   ├── input_large.txt
│   └── Shakespeare_data.csv
│
├── models/
│   └── transformer_final.pt
│
├── source/
│   ├── model_transformer.py
│   ├── model_lstm.py
│   ├── model_gru.py
│   ├── utils.py
│   └── tokenizer_bpe.py
│
├── logs/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Configuration

| Hyperparameter | Value       |
|----------------|-------------|
| Model Type     | Transformer |
| Layers         | 8           |
| Heads          | 8           |
| Embedding Dim  | 512         |
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
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Download the model and run:

```bash
python app.py
```

Visit [http://localhost:7860](http://localhost:7860)

---

## 🔥 Gradio Features

- Prompt input box
- Length, Temperature, Top-k sliders
- Light/Dark mode
- Output download
- Prompt history
- Hugging Face embed support

---

## 🧪 Use from Terminal

```bash
python generate_text.py
```

---

## 🛠️ Dev Roadmap

- [x] LSTM / GRU baselines
- [x] Transformer model
- [x] Gradio + CLI
- [x] W&B tracking
- [x] Hugging Face deployment
- [ ] Byte Pair Encoding
- [ ] Multilingual toggle
- [ ] Attention viz
- [ ] Docker + PyPI

---

## 👨‍💻 Author

**Tamaghna Nag**  
Founder of NovalQ | ML Engineer | Code Poet

- 🌐 [Portfolio](https://tamaghnatech.in)  
- 🧠 [W&B Logs](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm?nw=nwusernagtamaghna)

---

## 🪄 Final Word

> “A tiny model trained on timeless literature.  
> Proof that small things, when trained right, can be legendary.”

---

⭐ **Fork. Prompt. Publish.**  
🔗 [Launch the App on Hugging Face](https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM)
