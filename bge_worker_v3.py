# FILE: bge_worker.py (Optimized Version)
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import AutoModel
from PIL import Image
from io import BytesIO
import time
from contextlib import nullcontext # --- MODIFICATION: Import for creating a context manager

# --- Configuration ---
MODEL_NAME = "BAAI/BGE-VL-large"
DEVICE = os.getenv("BGE_DEVICE", "cuda:0")

app = FastAPI()
model_data = {} # Will store model, device, and amp_context

@app.on_event("startup")
def load_model():
    """
    Load and optimize the BGE-VL-Large model on startup.
    """
    print(f"--- BGE Worker: Loading model '{MODEL_NAME}' onto {DEVICE}... ---")
    st_load = time.time()
    
    device = torch.device(DEVICE)
    
    # --- MODIFICATION START: Improved model loading for mixed precision ---
    # Load the model in its default precision first (usually float32).
    # We will apply mixed precision dynamically during inference using autocast for better stability.
    # The `trust_remote_code=True` is required for this model architecture.
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(device).eval()
    # --- MODIFICATION END ---
    
    # The set_processor call is crucial for this model's internal tokenizer and image processor.
    model.set_processor(MODEL_NAME)

    # torch.compile() is a major optimization that JIT-compiles the model for significant speedup.
    if hasattr(torch, 'compile') and 'cuda' in DEVICE:
        print("--- BGE Worker: Compiling model with torch.compile()... (this may take a minute) ---")
        model = torch.compile(model)

    model_data['model'] = model
    model_data['device'] = device
    
    # --- MODIFICATION START: Set up Automatic Mixed Precision (AMP) context ---
    # This is the standard and most robust way to get FP16 speed benefits.
    # It automatically casts only the safe operations to half-precision.
    if 'cuda' in DEVICE:
        print("--- BGE Worker: Enabling Automatic Mixed Precision (AMP). ---")
        # For PyTorch versions >= 1.10, this is the recommended syntax.
        amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        print("--- BGE Worker: CPU device detected. AMP will not be used. ---")
        amp_context = nullcontext()
    model_data['amp_context'] = amp_context
    # --- MODIFICATION END ---
    
    print(f"--- BGE Worker: Model loaded and optimized in {time.time() - st_load:.2f}s. Ready. ---")

# --- MODIFICATION START: Changed endpoint from `async def` to `def` ---
# This is the MOST IMPORTANT change for server throughput. FastAPI runs `def` routes
# in a thread pool, preventing a single long model inference from blocking the
# entire server. This allows for true concurrent request handling.
@app.post("/embed")
def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
# --- MODIFICATION END ---
    """
    Generate an embedding for a text query, an uploaded image, or both (multimodal).
    """
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Please provide 'text_query' or 'image_file'")

    model = model_data.get('model')
    if not model:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    # --- MODIFICATION START: Refactored and simplified logic ---
    # We build a dictionary of arguments for the model's encode function.
    # This is cleaner than multiple if/elif branches.
    encode_kwargs = {}
    if text_query:
        encode_kwargs['text'] = [text_query]
    
    if image_file:
        # Since this is now a synchronous function, we read the file directly.
        image_bytes = image_file.file.read()
        encode_kwargs['images'] = [BytesIO(image_bytes)]
    
    vec_list = None
    # `torch.no_grad()` is essential for inference speed and memory.
    with torch.no_grad():
        # Apply the AMP context for mixed-precision speedup on GPU.
        with model_data['amp_context']:
            # Use the ** operator to unpack our arguments dictionary into the function call.
            vec_tensor = model.encode(**encode_kwargs)
            vec_list = vec_tensor.cpu().numpy().tolist()
    # --- MODIFICATION END ---

    if not vec_list:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

    # The model returns a list of embeddings. For a single request, we return the first one.
    return {"embedding": vec_list[0]}