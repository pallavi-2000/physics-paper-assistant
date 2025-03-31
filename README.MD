### 📁 File: README.md

# 🧠 Physics Paper Assistant + Simulator

A smart AI-powered assistant that helps students and researchers **understand physics papers**, **explain equations**, and even **simulate concepts** directly from research snippets — built with LLMs, LangChain, OpenRouter, and Streamlit.

## 🚀 Features
- 📄 Paste a paragraph from any physics paper
- 🧠 Summarize and simplify the language
- 🔢 Explain key equations step-by-step
- 💻 Generate simulation code (Python)
- 📊 Visualize concepts (WIP)

## 📦 Folder Structure

```
physics-paper-assistant/
├── app/
│   ├── main.py              ← Streamlit app
│   ├── prompts.py           ← Prompt templates
├── .env.example             ← Template for API key
├── requirements.txt
├── README.md
```

## 🔧 Setup

```bash
git clone https://github.com/your-username/physics-paper-assistant.git
cd physics-paper-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OpenRouter key
streamlit run app/main.py
```

---
