"""
Streaming Streamlit app for reverse GPT-2 text generation
"""

import os
import pickle
import torch
import tiktoken
import streamlit as st
from contextlib import nullcontext
from model import GPTConfig, GPT
import time

# Set page config
st.set_page_config(
    page_title="Reverse GPT-2 Generator",
    page_icon="🔄",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the reverse GPT-2 model"""
    
    # Model configuration
    init_from = "resume"
    out_dir = "out-rev-openwebtext"
    device = "cpu"
    
    # Load model
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Setup encoding
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    return model, encode, decode, device

def generate_story_streaming(model, encode, decode, device, prompt, max_tokens=500, temperature=0.8, top_k=200):
    """Generate the story and yield it progressively"""
    
    # First generate the complete story
    start = f"{prompt}<|endoftext|>"
    start_ids = encode(start)
    start_ids = start_ids[::-1]  # Reverse for the reverse model
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate using model's generate method
    with torch.no_grad():
        with nullcontext():
            y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
            
            # Reverse the output to get the final result
            reversed_output = decode(y[0].tolist()[::-1])
            
            # Extract the generated part (everything before our prompt)
            if f"{prompt}<|endoftext|>" in reversed_output:
                before_prompt = reversed_output.split(f"{prompt}<|endoftext|>")[0]
                full_story = before_prompt + prompt
            else:
                full_story = reversed_output
            
            # Now yield the story progressively for streaming effect
            current_text = ""
            for i, char in enumerate(full_story):
                current_text += char
                yield current_text
                
                # Adjust timing based on character type
                if char in ['.', '!', '?']:
                    time.sleep(0.1)  # Pause at sentence endings
                elif char in [',', ';', ':']:
                    time.sleep(0.05)  # Brief pause at punctuation
                elif char == ' ':
                    time.sleep(0.02)  # Quick pause at spaces
                else:
                    time.sleep(0.01)  # Fast for regular characters

# Streamlit UI
st.title("🔄 Reverse GPT-2 Generator")
st.markdown("Enter a prompt and watch as the model generates text that would come *before* your prompt!")

# Load model
try:
    model, encode, decode, device = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Input section
st.header("Input")
prompt = st.text_area(
    "Enter your prompt:",
    value="And they lived happily ever after.",
    height=100,
    help="The model will generate text that would logically come before this prompt."
)

# Parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_tokens = st.slider("Max tokens", 50, 1000, 500)
with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
with col3:
    top_k = st.slider("Top-k", 1, 1000, 200)

# Generate button
if st.button("🚀 Generate", type="primary"):
    if prompt.strip():
        st.header("Generated Story")
        
        # Create container for streaming output
        story_container = st.empty()
        
        try:
            # Stream the generation
            for current_story in generate_story_streaming(model, encode, decode, device, prompt, max_tokens, temperature, top_k):
                story_container.text_area(
                    "Story in progress...",
                    value=current_story,
                    height=400,
                    disabled=True
                )
            
            st.success("✅ Story generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter a prompt first!")

# Instructions
st.sidebar.header("How it works")
st.sidebar.markdown("""
This app uses a reverse-trained GPT-2 model that predicts what text would come **before** your prompt.

1. Enter a prompt (e.g., "And they lived happily ever after")
2. Click Generate
3. Watch as the model creates text in real-time!

**Parameters:**
- **Max tokens**: Maximum number of tokens to generate
- **Temperature**: Controls randomness (higher = more random)
- **Top-k**: Limits token choices to top k most likely tokens
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit and PyTorch")