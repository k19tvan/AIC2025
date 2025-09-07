# FILE: beit3_worker.py (Optimized Version)
import os, sys, torch
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from contextlib import nullcontext # --- MODIFICATION: Import for creating a context manager

# --- BEiT-3 Setup ---
# Assumes this script is run from the project root
try:
    from modeling_finetune import BEiT3ForRetrieval, _get_large_config
    from datasets import get_sentencepiece_model_for_beit3
except ImportError as e:
    sys.exit(f"FATAL: BEiT-3 source files not found. Make sure they are in the 'scripts' directory. Error: {e}")

DEVICE = os.getenv("BEIT3_DEVICE", "cuda:0")
MODEL_PATH = "./beit3_large_patch16_384_coco_retrieval.pth"
SPM_PATH = "./beit3.spm"

app = FastAPI()
model_data = {} # Will store model, tokenizer, transform, and amp_context

# --- MODIFICATION START: Check for PyTorch 2.0+ for torch.compile() ---
IS_PYTORCH_2 = int(torch.__version__.split('.')[0]) >= 2
# --- MODIFICATION END ---

# Define the transform globally to avoid re-creating it on every request
image_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.on_event("startup")
def load_model():
    print(f"--- BEiT-3 Worker: Loading model onto {DEVICE}... ---")
    model_config = _get_large_config(img_size=384)
    model = BEiT3ForRetrieval(model_config)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt.get('module')), strict=False)

    model = model.to(DEVICE).eval()

    # --- MODIFICATION START: Add torch.compile() for PyTorch 2.x for huge speedup ---
    if IS_PYTORCH_2 and 'cuda' in DEVICE:
        print("PyTorch 2.x detected. Compiling model... (this may take a minute)")
        # This uses a JIT compiler to fuse operations and optimize the model graph
        # for your specific hardware, resulting in a significant performance boost.
        model = torch.compile(model)
        print("Model compiled successfully.")
    # --- MODIFICATION END ---

    model_data['model'] = model
    model_data['tokenizer'] = get_sentencepiece_model_for_beit3(SimpleNamespace(sentencepiece_model=SPM_PATH))
    model_data['transform'] = image_transform

    # --- MODIFICATION START: Set up Automatic Mixed Precision (AMP) for GPU speedup ---
    # AMP allows the model to use faster half-precision (FP16) math for many operations
    # without sacrificing significant accuracy. It only works on CUDA devices.
    if 'cuda' in DEVICE:
        print("CUDA device detected. Enabling Automatic Mixed Precision (AMP).")
        amp_context = torch.cuda.amp.autocast()
    else:
        print("CPU device detected. AMP will not be used.")
        amp_context = nullcontext()
    model_data['amp_context'] = amp_context
    # --- MODIFICATION END ---

    print("--- BEiT-3 Worker: Model loaded and ready. ---")

# --- MODIFICATION START: Changed endpoint from `async def` to `def` ---
# This is CRITICAL for performance. FastAPI runs normal `def` routes in a
# separate thread pool. This prevents the single, blocking PyTorch model call
# from freezing the entire server's event loop. This allows the server to
# handle many concurrent requests smoothly instead of processing them one by one.
@app.post("/embed")
def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
# --- MODIFICATION END ---
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Provide text_query or image_file")
    
    vec = None
    # torch.no_grad() is essential for inference as it disables gradient calculations,
    # reducing memory consumption and speeding up computations.
    with torch.no_grad():
        # --- MODIFICATION START: Use the amp_context for inference ---
        # This `with` block applies AMP (if on CUDA) or does nothing (if on CPU).
        with model_data['amp_context']():
            if image_file:
                # The way to read the file changes slightly from async to sync
                image = Image.open(BytesIO(image_file.file.read())).convert("RGB")
                tensor = model_data['transform'](image).unsqueeze(0).to(DEVICE)
                
                # Model call is the same for images
                vec_tensor, _ = model_data['model'](image=tensor, only_infer=True)
                vec = vec_tensor.cpu().numpy().tolist()
            
            else: # text_query
                tokenizer = model_data['tokenizer']
                tokens = tokenizer.encode(text_query, out_type=int)[:64]
                padded = tokens + [tokenizer.pad_token_id] * (64 - len(tokens))
                text_input = torch.tensor([padded], device=DEVICE)
                padding_mask = (text_input == tokenizer.pad_token_id)
                
                # --- MODIFICATION START: The MOST IMPORTANT fix for text embedding ---
                # REMOVED: dummy_image = torch.zeros(1, 3, 384, 384, device=DEVICE)
                #
                # We now call the model WITHOUT the `image` argument. This tells the model
                # to *only* run the text encoder part of the network. The previous version
                # was wastefully running the entire, expensive vision transformer on a
                # blank image for every single text query. This change avoids that waste.
                _, vec_tensor = model_data['model'](
                    text_description=text_input,
                    padding_mask=padding_mask,
                    only_infer=True
                )
                # --- MODIFICATION END ---
                vec = vec_tensor.cpu().numpy().tolist()
        # --- MODIFICATION END ---

    return {"embedding": vec}