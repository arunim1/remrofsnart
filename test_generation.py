"""
Test the generation function directly
"""
import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

def test_generation():
    # Load model
    init_from = "resume"
    out_dir = "out-rev-openwebtext"
    device = "cpu"
    reverse = True
    
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
    
    # Test generation
    prompt = "And they lived happily ever after."
    start = f"{prompt}<|endoftext|>"
    start_ids = encode(start)
    
    print(f"Original prompt: '{prompt}'")
    print(f"Start with token: '{start}'")
    print(f"Encoded: {start_ids}")
    
    if reverse:
        start_ids = start_ids[::-1]
        print(f"Reversed: {start_ids}")
    
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    print(f"Input tensor shape: {x.shape}")
    
    # Generate
    with torch.no_grad():
        with nullcontext():
            try:
                y = model.generate(x, 100, temperature=0.8, top_k=200)
                print(f"Generated tensor shape: {y.shape}")
                
                # Decode the raw output
                raw_output = decode(y[0].tolist())
                print(f"\nRaw output:\n{raw_output}")
                
                if reverse:
                    # Reverse the output like in sample.py
                    reversed_output = decode(y[0].tolist()[::-1])
                    print(f"\nReversed output:\n{reversed_output}")
                    
                    # Extract just the generated part
                    if prompt in reversed_output:
                        before_prompt = reversed_output.split(f"{prompt}<|endoftext|>")[0]
                        print(f"\nGenerated story:\n{before_prompt}{prompt}")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_generation()