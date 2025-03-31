import streamlit as st
import openai
import os
from dotenv import load_dotenv
from prompts import (
    PAPER_EXPLANATION_PROMPT,
    EQUATION_EXPLANATION_PROMPT,
    PYTHON_SIMULATION_PROMPT,
    JULIA_SIMULATION_PROMPT,
    VISUALIZATION_PROMPT
)
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
from contextlib import redirect_stdout
import traceback
import base64
import subprocess
import tempfile

# Load environment variables
load_dotenv()

# Configure OpenAI client for OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY")

def get_llm_response(prompt, model="anthropic/claude-3-opus"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        st.write("🔍 Raw LLM response:", response)  # Debug print
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error communicating with LLM: {e}")
        return None


def run_python_simulation(code):
    """Execute the Python simulation code and capture output and figures."""
    # Create a StringIO object to capture print outputs
    output_buffer = io.StringIO()
    
    # Create a place to store figures
    figures = []
    
    # Save the current state of plt.figure
    original_figure = plt.figure
    
    # Override plt.figure to keep track of figures created
    def custom_figure(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        figures.append(fig)
        return fig
    
    # Set the custom function
    plt.figure = custom_figure
    
    try:
        # Execute the code and capture stdout
        with redirect_stdout(output_buffer):
            exec(code, globals())
        
        # Get the captured output
        output = output_buffer.getvalue()
        
        return {
            "success": True,
            "output": output,
            "figures": figures,
            "error": None
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        return {
            "success": False,
            "output": output_buffer.getvalue(),
            "figures": figures,
            "error": error_msg
        }
    finally:
        # Restore the original plt.figure function
        plt.figure = original_figure

def run_julia_simulation(code):
    """
    Execute Julia code and return the results.
    Note: This requires Julia to be installed on the system.
    """
    try:
        # Create a temporary file to store the Julia code
        with tempfile.NamedTemporaryFile(suffix='.jl', delete=False) as tmp:
            tmp.write(code.encode())
            tmp_name = tmp.name
        
        # Run the Julia code
        result = subprocess.run(
            ['julia', tmp_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Process the result
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "error": None,
                "figures": []  # Julia figures would need a different handling mechanism
            }
        else:
            return {
                "success": False,
                "output": result.stdout,
                "error": result.stderr,
                "figures": []
            }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "figures": []
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

def get_download_link(code, filename, language):
    """Generate a download link for the code."""
    b64 = base64.b64encode(code.encode()).decode()
    extension = ".jl" if language.lower() == "julia" else ".py"
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}{extension}">Download {language} Code</a>'
    return href

def extract_latex_equations(text):
    """
    Extract LaTeX equations from text.
    This is a simplified version - a more sophisticated extraction would use regex
    """
    equations = []
    # Look for content between $ symbols (inline math)
    dollar_split = text.split('$')
    
    # If there are an odd number of $ symbols, they're probably used as equation delimiters
    if len(dollar_split) > 1 and len(dollar_split) % 2 == 1:
        for i in range(1, len(dollar_split), 2):
            equations.append(dollar_split[i])
    
    # Look for content between \begin{equation} and \end{equation} tags
    if "\\begin{equation}" in text and "\\end{equation}" in text:
        eq_blocks = text.split("\\begin{equation}")
        for block in eq_blocks[1:]:  # Skip the first split which is before any equation
            if "\\end{equation}" in block:
                eq = block.split("\\end{equation}")[0].strip()
                equations.append(eq)
    
    return equations

# App title and description
st.set_page_config(page_title="Physics Paper Assistant", layout="wide")
st.title("Physics Paper Assistant + Simulator")
st.markdown("""
This tool helps you understand physics papers by:
- Summarizing and simplifying complex language
- Extracting and explaining equations step-by-step
- Generating simulation code in Python or Julia
- Visualizing physics concepts
""")

# Sidebar for settings
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Select LLM Model",
    ["anthropic/claude-3-opus", "anthropic/claude-3-sonnet", "google/gemini-pro"]
)

language_choice = st.sidebar.radio(
    "Simulation Language",
    ["Python", "Julia"]
)

# Main app interface
tab1, tab2, tab3, tab4 = st.tabs(["Paper Explanation", "Equation Explanation", "Simulation", "About"])

# Tab 1: Paper Explanation
with tab1:
    st.header("Paper Explanation")
    
    # Add file uploader for PDF
    uploaded_file = st.file_uploader("Upload a physics paper (PDF)", type=["pdf"])
    
    paper_text = st.text_area(
        "Or paste a paragraph from a physics paper:", 
        height=200,
        help="Paste text from a physics paper that you want explained in simpler terms."
    )
    
    extract_equations = st.checkbox("Automatically extract and explain equations", value=True)
    
    if st.button("Explain Paper", key="explain_paper"):
        if paper_text:
            with st.spinner("Analyzing the paper content..."):
                # First, provide the general explanation
                prompt = PAPER_EXPLANATION_PROMPT.format(paper_text=paper_text)
                explanation = get_llm_response(prompt, model=model_choice)
                if explanation:
                    st.markdown("### Explanation")
                    st.markdown(explanation)
                    
                    # If the extract equations option is checked, try to find and explain equations
                    if extract_equations:
                        equations = extract_latex_equations(paper_text)
                        if equations:
                            st.markdown("### Detected Equations")
                            for i, eq in enumerate(equations):
                                st.latex(eq)
                                with st.spinner(f"Explaining equation {i+1}..."):
                                    eq_prompt = EQUATION_EXPLANATION_PROMPT.format(
                                        equation=eq, 
                                        context=paper_text
                                    )
                                    eq_explanation = get_llm_response(eq_prompt, model=model_choice)
                                    if eq_explanation:
                                        st.markdown(f"**Explanation of Equation {i+1}:**")
                                        st.markdown(eq_explanation)
                        else:
                            st.info("No LaTeX equations were detected in the text.")
        elif uploaded_file:
            st.warning("PDF processing is still in development. Please paste text directly for now.")
        else:
            st.warning("Please paste some text from a physics paper or upload a PDF.")

# Tab 2: Equation Explanation
with tab2:
    st.header("Equation Explanation")
    equation = st.text_area("Paste an equation (LaTeX or plain text):", height=100)
    
    if equation:
        try:
            # Display the equation using LaTeX rendering
            st.latex(equation)
        except:
            st.info("Enter LaTeX format for proper equation rendering")
    
    context = st.text_area("Additional context (optional):", height=100)
    
    if st.button("Explain Equation", key="explain_equation"):
        if equation:
            with st.spinner("Analyzing the equation..."):
                prompt = EQUATION_EXPLANATION_PROMPT.format(
                    equation=equation, 
                    context=context if context else "No additional context provided."
                )
                explanation = get_llm_response(prompt, model=model_choice)
                if explanation:
                    st.markdown("### Equation Explanation")
                    st.markdown(explanation)
                    
                    # Suggest simulation of this equation if applicable
                    st.markdown("### Would you like to simulate this equation?")
                    if st.button("Generate Simulation for this Equation", key="sim_from_eq"):
                        with st.spinner("Generating simulation code..."):
                            if language_choice == "Python":
                                sim_prompt = PYTHON_SIMULATION_PROMPT.format(
                                    concept=f"Simulate the following equation with appropriate parameters: {equation}"
                                )
                            else:
                                sim_prompt = JULIA_SIMULATION_PROMPT.format(
                                    concept=f"Simulate the following equation with appropriate parameters: {equation}"
                                )
                            
                            simulation_code = get_llm_response(sim_prompt, model=model_choice)
                            if simulation_code:
                                st.session_state.simulation_code = simulation_code
                                st.session_state.simulation_language = language_choice
                                st.success(f"Simulation code generated in {language_choice}!")
                                st.code(simulation_code, language=language_choice.lower())
        else:
            st.warning("Please enter an equation to explain.")

# Tab 3: Simulation
with tab3:
    st.header("Physics Simulation")
    concept = st.text_area(
        "Describe the physics concept you want to simulate:", 
        height=150,
        placeholder="E.g., 'A damped harmonic oscillator with mass=1kg, spring constant=10N/m, and damping coefficient=0.5kg/s'"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Simulation Code", key="gen_simulation"):
            if concept:
                with st.spinner(f"Generating {language_choice} simulation code..."):
                    if language_choice == "Python":
                        prompt = PYTHON_SIMULATION_PROMPT.format(concept=concept)
                    else:
                        prompt = JULIA_SIMULATION_PROMPT.format(concept=concept)
                    
                    simulation_code = get_llm_response(prompt, model=model_choice)
                    if simulation_code:
                        # Try to extract code from markdown if necessary
                        if "```" in simulation_code:
                            code_blocks = simulation_code.split("```")
                            for i, block in enumerate(code_blocks):
                                if i % 2 == 1:  # This is a code block
                                    lang_line = block.split('\n')[0].strip()
                                    if lang_line.lower() in ["python", "julia", ""]:
                                        # Extract the code without the language identifier
                                        code_part = '\n'.join(block.split('\n')[1:]) if lang_line else block
                                        st.session_state.simulation_code = code_part.strip()
                                        st.session_state.simulation_language = language_choice
                                        break
                        else:
                            st.session_state.simulation_code = simulation_code
                            st.session_state.simulation_language = language_choice
                        
                        st.success(f"{language_choice} simulation code generated!")
            else:
                st.warning("Please describe a physics concept to simulate.")
    
    with col2:
        if st.button("Run Simulation", key="run_simulation"):
            if 'simulation_code' in st.session_state:
                sim_lang = st.session_state.simulation_language
                with st.spinner(f"Running {sim_lang} simulation..."):
                    if sim_lang == "Python":
                        result = run_python_simulation(st.session_state.simulation_code)
                    else:  # Julia
                        result = run_julia_simulation(st.session_state.simulation_code)
                    
                    if result["success"]:
                        st.success("Simulation completed successfully!")
                    else:
                        st.error(f"Simulation encountered an error:\n{result['error']}")
                        
                    # Store the result for display
                    st.session_state.simulation_result = result
            else:
                st.warning("Please generate simulation code first.")
    
    # Display code and results
    if 'simulation_code' in st.session_state:
        st.subheader(f"{st.session_state.simulation_language} Simulation Code")
        st.code(st.session_state.simulation_code, language=st.session_state.simulation_language.lower())
        
        # Add download button
        st.markdown(
            get_download_link(
                st.session_state.simulation_code,
                "physics_simulation",
                st.session_state.simulation_language
            ),
            unsafe_allow_html=True
        )
    
    if 'simulation_result' in st.session_state:
        st.subheader("Simulation Output")
        
        # Display any text output
        if st.session_state.simulation_result["output"]:
            st.text(st.session_state.simulation_result["output"])
        
        # Display any figures for Python simulations
        if st.session_state.simulation_language == "Python":
            for fig in st.session_state.simulation_result["figures"]:
                st.pyplot(fig)
        
        # For Julia simulations, we'd need to handle differently
        # This is placeholder for future implementation
        if st.session_state.simulation_language == "Julia" and st.session_state.simulation_result["success"]:
            st.info("Julia visualization integration is under development. Check the output for generated image paths.")

# Tab 4: About
with tab4:
    st.header("About Physics Paper Assistant")
    st.markdown("""
    ## Vision
    
    This tool is designed to bridge the gap between complex physics research literature and actionable insight. 
    It's built for students, researchers, and physics enthusiasts who often find themselves overwhelmed by 
    dense academic texts, yet deeply curious about the science beneath.
    
    ## Features
    
    - **Understand** physics paper paragraphs or equations in plain language
    - **Generate** Python or Julia code to simulate physical systems
    - **Explain** how equations map to simulation code
    - **Visualize** results through plots and animations
    
    ## Technologies
    
    - **Frontend**: Streamlit
    - **LLM Integration**: LangChain + OpenRouter API
    - **Code Generation**: Python and Julia (DiffEqFlux.jl)
    - **Prompt Engineering**: Custom templates for different tasks
    
    ## Future Plans
    
    - Upload paragraph from PDF
    - Equation extraction + improved LaTeX rendering
    - Run Julia code in-browser (via Pluto.jl)
    - Physics-Informed Neural Networks (PINN) integration
    - Universal Differential Equations (UDE) support
    - Symbolic math explanations (SymPy)
    
    ## Creator
    
    Pallavi, Astrophysicist + AI Engineer-in-training
    
    GitHub: [https://github.com/pallavi-2000](https://github.com/pallavi-2000)
    """)