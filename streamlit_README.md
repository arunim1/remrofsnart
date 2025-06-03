# Reverse GPT-2 Streamlit App

A web interface for the reverse-trained GPT-2 model that generates text that would logically come **before** your prompt.

## Features

- 🔄 Reverse text generation (predicts what comes before your prompt)
- 🌊 Real-time token streaming
- 🎛️ Adjustable parameters (temperature, top-k, max tokens)
- 🖥️ Clean, intuitive web interface

## Local Development

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install streamlit torch tiktoken numpy
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to http://localhost:8501

## Deployment to Streamlit Community Cloud

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Deploy!

## How it Works

The app loads a reverse-trained GPT-2 model from `out-rev-openwebtext/ckpt.pt` and:

1. Takes user input prompt
2. Appends `<|endoftext|>` token
3. Reverses the token sequence (for the reverse model)
4. Generates tokens one by one
5. Streams the output back to the user
6. Shows the complete generated text + original prompt

## Model Requirements

The app expects:
- `out-rev-openwebtext/ckpt.pt` - PyTorch model checkpoint
- `model.py` - Model architecture definition
- Standard GPT-2 tokenizer (tiktoken)

## Parameters

- **Temperature**: Controls randomness (0.1 = deterministic, 2.0 = very random)
- **Top-k**: Limits token choices to top k most likely (lower = more focused)
- **Max tokens**: Maximum number of tokens to generate

## Example

Input: "And they lived happily ever after."
Output: "Once upon a time, there was a brave knight who saved the princess from the dragon. And they lived happily ever after."