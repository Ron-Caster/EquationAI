import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Load the Groq API key from the environment

# System prompt to help Groq understand the task
system_prompt = """
You are a helpful assistant designed to convert spoken equations into LaTeX code. 
When a user inputs a mathematical equation in text form, convert it into a valid LaTeX equation. 
You shall only answer "LaTeX code: whatever code here", don't reply anything else or
any explanations.

For example:
If the user inputs "A is equal to tan inverse x", your task is to:
1. Convert the text into a valid LaTeX equation: "A = \tan^{-1}(x)"
2. Provide this LaTeX code as output.

The LaTeX code should be formatted in the standard LaTeX form.
"""

# Function to convert text to LaTeX using Groq
def text_to_latex(text):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    chat_completion = client.chat.completions.create(
        messages=conversation,
        model="llama-3.3-70b-versatile",  # Or replace with the appropriate model ID
    )

    latex_code = chat_completion.choices[0].message.content
    if latex_code.startswith("LaTeX code: "):
        latex_code = latex_code[len("LaTeX code: "):]  # Remove prefix
    return latex_code

# Streamlit interface function
def streamlit_interface():
    st.title("Equation AI - Text to LaTeX Converter")
    
    user_input = st.text_area("Enter mathematical equation in text form:")
    
    if st.button("Convert to LaTeX"):
        if user_input.strip():
            with st.spinner("Generating LaTeX..."):
                try:
                    latex_code = text_to_latex(user_input)
                    st.subheader("Generated LaTeX:")
                    st.code(latex_code)
                except Exception as e:
                    st.error(f"Error generating LaTeX: {str(e)}")
        else:
            st.warning("Please enter some text to convert.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_interface()
