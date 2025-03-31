# Physics Paper Assistant + Simulator

A smart AI-powered assistant that helps students and researchers **understand physics papers**, **explain equations**, and even **simulate concepts** directly from research snippets — built with LLMs, LangChain, OpenRouter, and Streamlit.

This tool is built to bridge the gap between research literature and actionable insight. It's designed for those who often find themselves overwhelmed by dense academic texts, yet deeply curious about the science beneath.

## 🚀 Features

* 📄 **Paper Explanation**: Paste a paragraph from any physics paper and get a simplified explanation
* 🧠 **Equation Extraction**: Automatically identify and explain equations from papers
* 🔢 **Equation Breakdown**: Get step-by-step explanations of complex physics equations
* 💻 **Code Generation**: Automatically generate simulation code in Python or Julia
* 📊 **Interactive Visualization**: Run simulations and view results directly in the app
* 🧪 **Multi-language Support**: Choose between Python for accessibility or Julia for performance

## 📦 Folder Structure

```
physics-paper-assistant/
├── app/
│   ├── main.py              ← Streamlit app
│   ├── prompts.py           ← Prompt templates
├── examples/
│   ├── wave_equation_prompt.txt
│   ├── wave_equation.jl
├── .env.example             ← Template for API key
├── requirements.txt
├── README.md
```

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/pallavi-2000/physics-paper-assistant.git
cd physics-paper-assistant
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**

```bash
cp .env.example .env
```

Edit the `.env` file to add your [OpenRouter](https://openrouter.ai/) API key.

5. **Install Julia (optional, for Julia simulations)**

If you want to run Julia simulations, [install Julia](https://julialang.org/downloads/) and make sure it's available in your PATH.

6. **Run the application**

```bash
streamlit run app/main.py
```

## 💡 Usage Examples

### Understanding Complex Papers

Paste a paragraph from a physics paper to get:
- A plain-language summary of the main concept
- Explanation of key technical terms
- Real-world analogies to aid understanding
- Automatic extraction and explanation of equations

### Equation Explanation

Enter equations in LaTeX or plain text format to receive:
- Identification of the equation type
- Definition of all variables and constants
- Step-by-step physical interpretation
- Intuitive meaning and implications
- Connection to fundamental physics principles

### Physics Simulations

Describe a physics concept to:
- Generate simulation code in Python or Julia
- Run the simulation directly in the browser
- Visualize the phenomenon with interactive plots
- Download the code for further exploration

#### Example: 1D Wave Equation Simulation

Try pasting the following prompt in the Simulation tab:

```
Simulate the 1D wave equation (∂²u/∂t² = c² ∂²u/∂x²) with:
- Domain: x ∈ [0, 1]
- Initial condition: u(x,0) = sin(2π