"""
Quick test to verify model loading works
"""
import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT

def test_model_loading():
    # Test model loading
    init_from = "resume"
    out_dir = "out-rev-openwebtext"
    device = "cpu"
    
    try:
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
        
        print("✅ Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test encoding
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        
        test_text = "Hello world<|endoftext|>"
        encoded = encode(test_text)
        decoded = decode(encoded)
        print(f"✅ Encoding test: '{test_text}' -> {encoded} -> '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()