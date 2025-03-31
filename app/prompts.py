# Prompt templates for LLM interactions

WAVE_PROMPT = "Your wave prompt text or object here"

PAPER_EXPLANATION_PROMPT = """
You are a physics professor who specializes in making complex physics concepts accessible to students.
I'll provide you with a paragraph from a physics research paper. Please:

1. Summarize the main concept in simple language (1-2 sentences)
2. Explain the key physics principles involved
3. Break down any jargon or technical terms
4. Provide a real-world analogy if applicable
5. Explain why this concept is important or interesting

Here's the text from the paper:
"{paper_text}"

Respond in a clear, educational style that would help a physics undergraduate understand the concept.
"""

EQUATION_EXPLANATION_PROMPT = """
You are a physics professor who specializes in explaining complex equations.
I'll provide you with an equation and optional context from a physics paper. Please:

1. Identify the equation and what physical phenomenon it describes
2. Define each variable and constant in the equation
3. Explain the physical meaning of the equation step by step
4. Discuss any assumptions or limitations
5. Provide an intuitive interpretation of what the equation tells us
6. Connect the equation to fundamental physical principles

Equation: {equation}
Context: {context}

Include LaTeX formatting for mathematical expressions where appropriate, and make your explanation accessible to someone with undergraduate physics knowledge.
"""

PYTHON_SIMULATION_PROMPT = """
You are a computational physicist who specializes in creating physics simulations in Python.
Based on the following physics concept, create a Python simulation that:

1. Models the physics phenomenon accurately
2. Uses numpy, scipy, and matplotlib for calculations and visualization
3. Includes clear comments explaining the physics and implementation
4. Produces informative plots that illustrate key aspects of the phenomenon
5. Is complete, self-contained, and ready to run

Physics concept to simulate: "{concept}"

Your code should be high-quality, readable, and emphasize physical understanding. Include appropriate physical parameters and constants.
Ensure the code is ready to execute without additional imports beyond standard scientific Python packages.

If applicable, include animations or interactive elements that help visualize the physics.
"""

JULIA_SIMULATION_PROMPT = """
You are a computational physicist who specializes in creating physics simulations in Julia.
Based on the following physics concept, create a Julia simulation that:

1. Models the physics phenomenon accurately
2. Uses DifferentialEquations.jl for solving ODEs/PDEs
3. Uses Plots.jl for visualization
4. Includes clear comments explaining the physics and implementation
5. Is complete, self-contained, and ready to run

Physics concept to simulate: "{concept}"

Your code should leverage Julia's strengths for scientific computing, particularly its speed and ease of use for differential equations.
Include appropriate physical parameters and constants.

The code should be formatted to potentially work with Pluto.jl notebooks, but should be executable as a standalone .jl file.
If appropriate, use DiffEqFlux.jl for physics-informed machine learning components.

Return only the complete Julia code with comments, ready to be executed.
"""

VISUALIZATION_PROMPT = """
You are a data visualization expert who specializes in physics concepts.
Based on the following physics data and code, enhance the visualization to:

1. Clearly illustrate the key physics principles
2. Use appropriate visualization types (e.g., plots, animations, vector fields)
3. Include clear labels, titles, and annotations
4. Use an appealing and effective color scheme
5. Add interactive elements where appropriate

Current code:
```
{current_code}
```

Physics concept being visualized: "{concept}"

Return enhanced visualization code that makes the physical phenomenon more intuitive and engaging.
If using Python, focus on matplotlib, plotly, or other standard visualization libraries.
If using Julia, focus on Plots.jl, Makie.jl, or other standard Julia visualization packages.
"""

UDE_PROMPT = """
You are an expert in physics and scientific machine learning, specializing in Universal Differential Equations (UDEs).
Create Julia code that uses DiffEqFlux.jl to implement a physics-informed neural network for the following concept:

1. Formulate the physical problem as a differential equation with unknown terms
2. Implement a neural network to learn the unknown dynamics
3. Use domain knowledge to constrain the solution space
4. Include training and evaluation of the model
5. Visualize both the learned and true dynamics

Physics concept for UDE: "{concept}"

The code should be well-commented to explain both the physics and the neural network architecture.
Make sure the implementation demonstrates how UDEs combine traditional physics models with data-driven components.
"""

SYMPY_EXPLANATION_PROMPT = """
You are a theoretical physicist who specializes in symbolic mathematics.
Based on the following equation, create Python code using SymPy that:

1. Defines the equation symbolically
2. Performs symbolic manipulations to simplify or solve it
3. Derives related equations or properties
4. Explains the physical meaning of the mathematical operations
5. Visualizes key aspects of the equation

Equation: "{equation}"
Context: "{context}"

Your code should demonstrate how symbolic mathematics can provide insights into the physics described by the equation.
Include detailed comments explaining both the mathematics and its physical interpretation.
"""