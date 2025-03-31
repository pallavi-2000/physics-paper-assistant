WAVE_PROMPT = """
You are a physics simulation expert. Given the following request, generate Julia code using DifferentialEquations.jl to simulate the concept described.

Request:
"{input}"

Only return valid Julia code. Use reasonable defaults for wave speed, time span, and boundary conditions.
"""
