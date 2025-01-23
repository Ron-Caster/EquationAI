import os
import streamlit as st
import sys

# Check Python version compatibility
if sys.version_info >= (3, 10):
    st.error("This application requires Python version 3.6-3.9. Please use a compatible Python version.")
    st.stop()

# Global variables to store model and client
whisper_model = None
groq_client = None

@st.cache_resource(show_spinner=False)
def init_dependencies():
    global whisper_model, groq_client
    try:
        import whisper
        import sounddevice as sd
        from groq import Groq
        from dotenv import load_dotenv
        
        load_dotenv()
        
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        whisper_model = whisper.load_model("small.en")
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        return True, None
    except ImportError as e:
        return False, f"Failed to import required packages: {str(e)}"
    except Exception as e:
        return False, str(e)

import numpy as np
import tempfile
import wave
import queue
import time

# System prompt to help Groq understand the task`25`
system_prompt = """
You are a helpful assistant designed to convert spoken equations into LaTeX code. 
When a user speaks a mathematical equation, transcribe it into a clean text form, 
then convert the text into a valid LaTeX equation. 
You shall only answer "LaTeX code: whatever code here", don't reply anything else or
any explanations.

For example:
If the user says "A is equal to tan inverse x", your task is to:
1. Transcribe the equation into a clean format: "A = tan^{-1}(x)"
2. Convert the transcribed equation into LaTeX code: "A = \\tan^{-1}(x)"
3. Provide this LaTeX code as output.

The LaTeX code should be formatted in the standard Latex form.
"""

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recording = False
        self.audio_data = []
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.extend(indata.copy())
            
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            callback=self.callback,
            channels=1,
            samplerate=16000,
            dtype='int16'
        )
        self.stream.start()
        
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        return np.array(self.audio_data)

@st.cache_resource
def get_whisper_model():
    global whisper_model
    return whisper_model

# Function to convert text to LaTeX using Groq
def text_to_latex(text):
    global groq_client
    # Create a conversation with the system prompt
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    # Send the conversation to Groq for LaTeX code generation
    chat_completion = groq_client.chat.completions.create(
        messages=conversation,
        model="llama-3.3-70b-versatile",  # Or replace with the appropriate model ID
    )

    latex_code = chat_completion.choices[0].message.content
    # Strip the "LaTeX code: " prefix
    if latex_code.startswith("LaTeX code: "):
        latex_code = latex_code[len("LaTeX code: "):]
    return latex_code

# Streamlit interface function
def streamlit_interface():
    st.title("Equation AI - Speech to LaTeX Converter")
    
    with st.spinner("Initializing dependencies..."):
        success, error = init_dependencies()
        if not success:
            st.error("Failed to initialize required dependencies")
            st.error(f"Error details: {error}")
            st.stop()
    
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
        st.session_state.whisper_model = get_whisper_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Recording"):
            st.session_state.recorder.start_recording()
            st.session_state.recording_start_time = time.time()
            
    with col2:
        if st.button("Stop Recording"):
            if hasattr(st.session_state.recorder, 'recording') and st.session_state.recorder.recording:
                audio_data = st.session_state.recorder.stop_recording()
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                    wavfile = temp_wav.name
                    with wave.open(wavfile, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_data.tobytes())
                
                with st.spinner('Transcribing...'):
                    try:
                        result = st.session_state.whisper_model.transcribe(wavfile)
                        transcribed_text = result['text']
                        st.text(f"Transcribed: {transcribed_text}")
                        
                        with st.spinner('Generating LaTeX...'):
                            latex_code = text_to_latex(transcribed_text)
                            st.subheader("Generated LaTeX:")
                            st.code(latex_code)
                            # Removed the rendered equation part
                            # st.subheader("Rendered Equation:")
                            # st.latex(latex_code)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        os.unlink(wavfile)
    
    # Display recording duration
    if hasattr(st.session_state, 'recording_start_time') and \
       hasattr(st.session_state.recorder, 'recording') and \
       st.session_state.recorder.recording:
        duration = int(time.time() - st.session_state.recording_start_time)
        st.write(f"Recording... {duration} seconds")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_interface()
