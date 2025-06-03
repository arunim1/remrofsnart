"""
Streamlit app for reverse GPT-2 text generation
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
    dtype = "float32"  # Use float32 for CPU compatibility
    
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
    load_meta = False
    if (
        init_from == "resume"
        and "config" in checkpoint
        and "dataset" in checkpoint["config"]
    ):
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    return model, encode, decode, device

def generate_tokens_streaming(model, encode, decode, device, prompt, max_tokens=500, temperature=0.8, top_k=200):
    """Generate tokens one by one for streaming"""
    
    # Prepare input exactly like sample.py
    start = f"{prompt}<|endoftext|>"
    start_ids = encode(start)
    start_ids = start_ids[::-1]  # Reverse for the reverse model
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Use the model's generate method like in sample.py
    with torch.no_grad():
        with nullcontext():
            # Generate all tokens at once using the model's generate method
            y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
            
            # Get the full generated sequence
            full_generated = y[0].tolist()
            
            # Decode the full sequence to get the raw output
            raw_text = decode(full_generated)
            
            # Now reverse it to get the final result (like line 118 in sample.py)
            reversed_full = decode(full_generated[::-1])
            
            # The final text should have the original prompt removed and be properly ordered
            # Find where our original prompt starts in the reversed text
            prompt_with_token = f"{prompt}<|endoftext|>"
            if prompt_with_token in reversed_full:
                # Split and take the part before our prompt
                before_prompt = reversed_full.split(prompt_with_token)[0]
                
                # Stream character by character for effect
                for i, char in enumerate(before_prompt):
                    yield char
                    time.sleep(0.02)
            else:
                # Fallback: just stream the reversed text
                for i, char in enumerate(reversed_full):
                    yield char
                    time.sleep(0.02)

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
    value="And so, they lived happily ever after.",
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
        st.header("Generated Text")
        
        # Create containers for the output
        full_text_container = st.empty()
        progress_container = st.empty()
        
        generated_text = ""
        token_count = 0
        
        try:
            # Stream the generation
            for char in generate_tokens_streaming(model, encode, decode, device, prompt, max_tokens, temperature, top_k):
                generated_text += char
                token_count += 1
                
                # Update the display - show generated text + original prompt
                full_text_container.text_area(
                    f"Streaming... ({len(generated_text)} characters):",
                    value=generated_text + prompt,
                    height=300,
                    disabled=True
                )
                
                # Small delay for better UX
                time.sleep(0.01)
            
            # Final output
            st.header("Complete Story")
            final_text = generated_text + prompt
            
            st.text_area(
                "Full generated text:",
                value=final_text,
                height=400,
                disabled=True
            )
            
            st.success(f"✅ Generated {len(generated_text)} characters!")
            
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
    else:
        st.warning("⚠️ Please enter a prompt first!")

# Instructions
st.sidebar.header("How it works")
st.sidebar.markdown("""
This app uses a reverse-trained GPT-2 model that predicts what text would come **before** your prompt.

1. Enter a prompt (e.g., "And they lived happily ever after")
2. The model generates text that would logically precede your prompt
3. Watch as tokens are generated in real-time!

**Parameters:**
- **Max tokens**: Maximum number of tokens to generate
- **Temperature**: Controls randomness (higher = more random)
- **Top-k**: Limits token choices to top k most likely tokens
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit and PyTorch")