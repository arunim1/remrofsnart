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
    """Generate tokens one by one and prepend them to create streaming effect"""
    
    # Prepare input exactly like sample.py
    start = f"{prompt}<|endoftext|>"
    start_ids = encode(start)
    start_ids = start_ids[::-1]  # Reverse for the reverse model
    current_input = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    generated_text = ""
    
    # Generate tokens one by one
    with torch.no_grad():
        with nullcontext():
            for i in range(max_tokens):
                # Get logits for next token
                outputs = model(current_input)
                # Handle case where model returns tuple (logits, loss) or just logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample from the distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode the new token
                try:
                    new_token_text = decode([next_token.item()])
                    
                    # PREPEND the new token (since we're generating in reverse)
                    generated_text = new_token_text + generated_text
                    
                    # Yield the current state: generated_text + original_prompt
                    yield generated_text + prompt
                    
                    # Update input for next iteration
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # Check for end of text token
                    if next_token.item() == encode("<|endoftext|>")[0]:
                        break
                        
                except Exception as e:
                    # Skip problematic tokens
                    continue

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
            token_count = 0
            # Stream the generation - tokens are prepended in real-time
            for current_story in generate_story_streaming(model, encode, decode, device, prompt, max_tokens, temperature, top_k):
                token_count += 1
                story_container.text_area(
                    f"Generating... (Token {token_count}) - Watch text appear before your prompt!",
                    value=current_story,
                    height=400,
                    disabled=True
                )
                
                # Small delay for better visual effect
                time.sleep(0.05)
            
            st.success(f"✅ Generated {token_count} tokens!")
            
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