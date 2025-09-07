import os
import sys
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import time
import traceback
from contextlib import asynccontextmanager
import torch

# --- Thêm thư mục của script vào Python path ---
WORKER_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKER_FILE_DIR not in sys.path:
    sys.path.insert(0, WORKER_FILE_DIR)

# --- Import class model Ops-MM-embedding ---
try:
    from ops_mm_embedding_v1 import OpsMMEmbeddingV1
    print(f"--- Ops-MM Worker: Đã import thành công 'OpsMMEmbeddingV1' từ '{WORKER_FILE_DIR}'. ---")
except ImportError:
    print("!!! LỖI IMPORT NGHIÊM TRỌNG !!!")
    print("Không thể tìm thấy file 'ops_mm_embedding_v1.py'.")
    print("Vui lòng đảm bảo file này nằm trong cùng thư mục với worker.")
    sys.exit(1)

Image.MAX_IMAGE_PIPIXELS = None

# --- Cấu hình ---
MODEL_NAME = "OpenSearch-AI/Ops-MM-embedding-v1-2B"
DEVICE = "cuda"
# Thêm kích thước tối đa cho ảnh để tránh lỗi OOM.
MAX_IMAGE_DIMENSION = 768

# --- Lifespan Event Handler ---
model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code khởi động
    print(f"--- Ops-MM Worker: Đang tải model '{MODEL_NAME}' lên {DEVICE}... ---")
    st_load = time.time()
    
    try:
        # === PHẦN SỬA LỖI NẰM Ở ĐÂY ===
        # Khởi tạo model theo đúng cách gọi trong file demo của bạn:
        # - MODEL_NAME là tham số vị trí (không có tên)
        # - device là tham số từ khóa
        model = OpsMMEmbeddingV1(
            MODEL_NAME,
            device=DEVICE
        )
        # ================================

        model_data['model'] = torch.compile(model.eval())
        print(f"--- Ops-MM Worker: Model đã tải xong trong {time.time() - st_load:.2f}s. Sẵn sàng hoạt động. ---")
    except Exception as e:
        print(f"!!! LỖI KHỞI ĐỘNG MODEL !!!")
        print(f"Không thể tải model '{MODEL_NAME}'. Lỗi: {e}")
        traceback.print_exc()
        
    yield
    
    # Code tắt
    print("--- Ops-MM Worker: Đang tắt. Dọn dẹp dữ liệu model. ---")
    model_data.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- Các hàm Helper cho Ops-MM-embedding (Không thay đổi) ---
def embed_image_func(image: Image.Image, model: OpsMMEmbeddingV1):
    """Tạo embedding cho một ảnh."""
    return model.get_image_embeddings([image], show_progress=False)

def embed_text_func(text: str, model: OpsMMEmbeddingV1):
    """Tạo embedding cho một đoạn văn bản."""
    return model.get_text_embeddings([text], show_progress=False)

t2i_prompt = "Find an image that matches the given text."
def embed_fusion_func(image: Image.Image, text: str, model: OpsMMEmbeddingV1):
    """Tạo embedding kết hợp (fusion) cho cả ảnh và văn bản."""
    return model.get_fused_embeddings(text=[text], images=[image], instruction=t2i_prompt, show_progress=False)

@app.post("/embed")
async def get_embedding(text_query: str = Form(None), image_file: UploadFile = File(None)):
    if not text_query and not image_file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp 'text_query' hoặc 'image_file' hoặc cả hai.")

    model = model_data.get('model')

    if not model:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng hoặc đã gặp lỗi khi tải.")

    vec = None
    try:
        with torch.no_grad():
            image = None
            if image_file:
                image_bytes = await image_file.read()
                try:
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    
                    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                        print(f"--- OPS-MM WORKER: Resize ảnh từ {img.size} về tối đa {MAX_IMAGE_DIMENSION}px ---")
                        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
                    image = img
                    
                except UnidentifiedImageError:
                    raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ.")

            # Logic để gọi hàm embedding tương ứng
            if image and text_query:
                print("--- OPS-MM WORKER: Chế độ Fusion (Ảnh + Text) ---")
                vec_tensor = embed_fusion_func(image, text_query, model)
            elif image:
                print("--- OPS-MM WORKER: Chế độ Image-Only ---")
                vec_tensor = embed_image_func(image, model)
            elif text_query:
                print("--- OPS-MM WORKER: Chế độ Text-Only ---")
                vec_tensor = embed_text_func(text_query, model)

            vec = vec_tensor[0].cpu().float().numpy().tolist()

    except Exception as e:
        print(f"!!! OPS-MM WORKER GẶP SỰ CỐ !!!")
        print(f"Loại lỗi: {type(e).__name__}")
        print(f"Chi tiết: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if vec is None:
        raise HTTPException(status_code=500, detail="Không thể tạo embedding vì một lý do không xác định.")

    return {"embedding": [vec]}

# Để chạy thử nghiệm trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mm_embedding_worker:app", host="0.0.0.0", port=8004, reload=True)