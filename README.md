<h1 align="center">🚀 Tiny-LLM: Shakespearean Text Generator</h1>

<p align="center">
  <strong>“A tiny model built on timeless literature. Proof that small things, when trained right, can be legendary.”</strong><br><br>
  <em>by <a href="https://tamaghnatech.in">Tamaghna Nag</a> | Founder, NovalQ | ML Engineer | Code Poet</em>
</p>

---

## 🌐 Live Demo

<div align="center">
  <a href="https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM" target="_blank">
    <img src="https://img.shields.io/badge/Try%20on%20Hugging%20Face-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black" alt="Try on Hugging Face">
  </a>
</div>

---

## 📜 About the Project

🎭 **Tiny-LLM** is a *local-first*, completely from-scratch **mini language model** that generates surreal and Shakespearean text using a **Transformer**, trained solely on the *Complete Works of Shakespeare*.

🧱 No Hugging Face pretrained weights. No shortcuts. Pure PyTorch.

🎨 With **Gradio UI** and **CLI**, it’s plug-and-play Shakespeare generation at your fingertips.

---

## 🔁 Full MLOps Pipeline (End-to-End)

> A practical implementation of an entire ML workflow from scratch.

### 1. 📥 Data Collection & Preprocessing
- Downloaded from **Kaggle's Shakespeare Corpus**
- Cleaned & prepared as `input.txt` and `input_large.txt`
- Script: [`prepare_dataset.py`](prepare_dataset.py)

### 2. 🧠 Model Building
- ✅ Baselines: `LSTM` and `GRU` (for fun and comparison)
- 🚀 Final: Transformer
  - Layers: 8
  - Heads: 8
  - Embedding Dim: 512
  - Block Size: 256
  - Tokenizer: Character-level

### 3. 🏋️‍♂️ Model Training
- Using `trainer_transformer.py` (and LSTM/GRU scripts)
- Optimizer: `AdamW`
- Loss: `CrossEntropy`
- Checkpoints: Saved in `/models`

### 4. 📈 Experiment Tracking with Weights & Biases
- Training metrics, loss curves, gradients, configs
- 🔗 [View Dashboard](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm?nw=nwusernagtamaghna)

### 5. 🌍 Inference
- `generate_text.py`: CLI mode
- `app.py`: Gradio app with sliders, dark/light theme, prompt history

### 6. 🚀 Deployment
- Hugging Face Spaces + GitHub Repo
- Manual download for model weights

---

## 📦 Model Download (for Local Use)

The final trained model isn't included due to GitHub size limits.

📥 [Download `transformer_final.pt`](https://drive.google.com/file/d/1HanIwaT0_sILx3-jDXfmmgYqUfHiI_S-/view?usp=sharing) and save it to:

```bash
Tiny-LLM/
└── models/
    └── transformer_final.pt
```

Then run locally:

```bash
python app.py
```

---

## 🗂️ Project Structure

```
Tiny-LLM/
├── app.py                  # Gradio UI
├── generate_text.py        # CLI tool
├── trainer_transformer.py  # Transformer trainer
├── prepare_dataset.py      # Data cleaner
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
│   └── tokenizer_bpe.py (WIP)
│
├── notebooks/              # Jupyter logs
├── logs/                   # Session files
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Configuration

| Parameter      | Value       |
|----------------|-------------|
| Architecture   | Transformer |
| Layers         | 8           |
| Heads          | 8           |
| Embedding Dim  | 512         |
| Block Size     | 256         |
| Tokenizer      | Char-level  |
| Optimizer      | AdamW       |
| Trained Steps  | 100         |

---

## 💻 Setup Locally

```bash
git clone https://github.com/Tamaghnatech/Tiny-LLM.git
cd Tiny-LLM

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download the model weights, place them in `models/`, then run:

```bash
python app.py
```

Visit `http://localhost:7860`

---

## 🧪 CLI Usage

```bash
python generate_text.py
```

You'll be prompted for:
- Starting prompt
- Length
- Temperature
- Top-k

---

## 🎛 Gradio App Features

- 📝 Prompt input
- 🔥 Temperature, Top-k, Length sliders
- 🌙 Light/Dark theme toggle
- 💾 Downloadable output
- 📜 Prompt history
- ⚙️ Model details
- 🚀 Hosted on Hugging Face

---

## 🛠 Future Roadmap

- [x] LSTM & GRU prototypes
- [x] Full Transformer training
- [x] W&B integration
- [x] Gradio & CLI interface
- [x] Hugging Face deployment
- [ ] Byte Pair Encoding (BPE)
- [ ] Attention visualization
- [ ] Multilingual toggle
- [ ] Docker + PyPI release

---

## 👨‍💻 Author

**Tamaghna Nag**  
*ML Engineer | Founder – NovalQ | Shakespeare Whisperer*

- 🌐 [Portfolio](https://tamaghnatech.in)
- 🐙 [GitHub](https://github.com/Tamaghnatech)
- 📊 [W&B Logs](https://wandb.ai/nagtamaghna-oxford-vision-and-sensor-technology/tiny-llm)

---

## 🌟 Star, Fork, Prompt!

If you liked the project, ⭐ star it, 🍴 fork it, and try writing your own AI-generated poetry!

💬 Questions? Drop an issue or ping me on GitHub.

```bash
# Launch it now →
👉 https://huggingface.co/spaces/Tamaghnatech/Tiny-LLM
```

```
