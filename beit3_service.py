import grpc.aio as grpc # MODIFICATION: Import the async version
from concurrent import futures
import torch
import time
from io import BytesIO
from typing import Dict, Any
from contextlib import nullcontext
import asyncio # MODIFICATION: Import asyncio

# --- gRPC Imports (from generated files) ---
try:
    import embedding_pb2
    import embedding_pb2_grpc
except ImportError:
    print("Fatal: Could not import generated gRPC files.")
    exit()

# --- VBS Imports ---
try:
    from beit3_utils import load_model, embed_text, embed_image
except ImportError:
    print("Fatal: Could not find 'beit3_utils.py'.")
    exit()

# --- Configuration ---
class Settings:
    MODEL_DIR = "models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GRPC_PORT = 50051

settings = Settings()

# --- Global objects ---
model_pack: Dict[str, Any] = {}
amp_context = nullcontext()
IS_PYTORCH_2 = int(torch.__version__.split('.')[0]) >= 2

# --- gRPC Servicer Implementation ---
class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
    
    # MODIFICATION: Make the method async
    async def EmbedText(self, request, context):
        if not model_pack.get("model"):
            await context.set_code(grpc.StatusCode.UNAVAILABLE)
            await context.set_details("Worker model not ready.")
            return embedding_pb2.EmbeddingResponse()
            
        try:
            # MODIFICATION: Run blocking PyTorch code in a thread to not block the event loop
            def _blocking_embed():
                with amp_context:
                    text_model_pack = {"model": model_pack["model"], "tokenizer": model_pack["tokenizer"], "config": model_pack["config"]}
                    embedding_tensor = embed_text(text=request.text, **text_model_pack)
                
                query_vectors = embedding_tensor.cpu().numpy().tolist()
                if query_vectors and isinstance(query_vectors[0], list):
                    return query_vectors[0]
                return query_vectors

            embedding_list = await asyncio.to_thread(_blocking_embed)
            return embedding_pb2.EmbeddingResponse(embedding=embedding_list)
        except Exception as e:
            print(f"Error during text embedding: {e}")
            await context.set_code(grpc.StatusCode.INTERNAL)
            await context.set_details(f"An error occurred: {e}")
            return embedding_pb2.EmbeddingResponse()

    # MODIFICATION: Make the method async
    async def EmbedImage(self, request, context):
        if not model_pack.get("model"):
            await context.set_code(grpc.StatusCode.UNAVAILABLE)
            await context.set_details("Worker model not ready.")
            return embedding_pb2.EmbeddingResponse()
            
        try:
            # MODIFICATION: Run blocking PyTorch code in a thread
            def _blocking_embed():
                image_buffer = BytesIO(request.image_data)
                with amp_context:
                    image_model_pack = {"model": model_pack["model"], "transform": model_pack["transform"], "config": model_pack["config"]}
                    embedding_tensor = embed_image(image=image_buffer, **image_model_pack)
                
                query_vectors = embedding_tensor.cpu().numpy().tolist()
                if query_vectors and isinstance(query_vectors[0], list):
                    return query_vectors[0]
                return query_vectors

            embedding_list = await asyncio.to_thread(_blocking_embed)
            return embedding_pb2.EmbeddingResponse(embedding=embedding_list)
        except Exception as e:
            print(f"Error during image embedding: {e}")
            await context.set_code(grpc.StatusCode.INTERNAL)
            await context.set_details(f"An error occurred: {e}")
            return embedding_pb2.EmbeddingResponse()

# --- Server Startup Logic ---
# MODIFICATION: Convert serve() to an async function
async def serve():
    global model_pack, amp_context
    print("--- [BEiT-3 gRPC Worker] Loading model ---")
    
    model, tokenizer, transform, config = load_model(settings.MODEL_DIR, settings.DEVICE)
    
    if IS_PYTORCH_2 and settings.DEVICE != 'cpu':
        print("PyTorch 2.x detected. Compiling model...")
        model = torch.compile(model)
        print("Model compiled.")

    if settings.DEVICE == 'cuda':
        print("Enabling Automatic Mixed Precision (AMP).")
        amp_context = torch.cuda.amp.autocast()
        
    model_pack.update({"model": model, "tokenizer": tokenizer, "transform": transform, "config": config})
    print("--- [BEiT-3 gRPC Worker] Model loaded. ---")

    # --- Start gRPC Server (Async Version) ---
    server = grpc.server()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingService(), server)
    
    server.add_insecure_port(f'[::]:{settings.GRPC_PORT}')
    
    await server.start()
    print(f"--- [BEiT-3 gRPC Worker] Async Server started on port {settings.GRPC_PORT}. ---")
    
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        await server.stop(0)

# MODIFICATION: Run the async serve function
if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("Server shutdown gracefully.")