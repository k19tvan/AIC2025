import os
import sys
import time
import traceback
import base64
import asyncio
import operator
import json
import re
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, ORJSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from pymilvus import Collection, connections, utility
import requests
try:
    from pymilvus import LoadState as _MilvusLoadState
except ImportError:
    try:
        from pymilvus.client.constants import LoadState as _MilvusLoadState
    except Exception:
        _MilvusLoadState = None
from elasticsearch import Elasticsearch, NotFoundError
import polars as pl
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote

import cv2

# --- PERFORMANCE OPTIMIZATION: Caches ---
embedding_cache = {}
processed_query_cache = {}
CACHE_MAX_SIZE = 500
# -----------------------------------------

# ## START: GOOGLE IMAGE SEARCH INTEGRATION (HELPERS & MODELS) ##
google_search_session = requests.Session()
google_search_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
})

def get_google_images(keyword: str, k: int = 15):
    try:
        url = f"https://www.google.com/search?q={quote(keyword)}&tbm=isch"
        html = google_search_session.get(url, timeout=15).text
        start = html.find('["https://')
        if start == -1:
            return []
        html = html[start:]
        image_links = re.findall(r'\["(https?://[^"]+)",\d+,\d+]', html)
        seen = set()
        results = []
        for link in image_links:
            if not link.startswith("https://encrypted-tbn0.gstatic.com") and link not in seen:
                seen.add(link)
                results.append(link)
                if len(results) >= k:
                    break
        return results
    except Exception as e:
        print(f"Error during Google Image Search: {e}")
        return []

class GoogleImageSearchRequest(BaseModel):
    query: str

class DownloadImageRequest(BaseModel):
    url: str
# ## END: GOOGLE IMAGE SEARCH INTEGRATION (HELPERS & MODELS) ##


# --- Thiết lập & Cấu hình ---
app = FastAPI(default_response_class=ORJSONResponse)

TEMP_UPLOAD_DIR = Path("/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_BASE_DIR = "/workspace/mlcv2"

# ## START: GOOGLE IMAGE SEARCH API ENDPOINTS ##
@app.post("/google_image_search")
async def google_image_search(request_data: GoogleImageSearchRequest):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        image_urls = await asyncio.to_thread(get_google_images, request_data.query)
        return {"image_urls": image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search Google Images: {e}")

@app.post("/download_external_image")
async def download_external_image(request_data: DownloadImageRequest):
    try:
        response = requests.get(request_data.url, stream=True, timeout=20, headers=google_search_session.headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', 'image/jpeg')
        if 'image' not in content_type:
            raise HTTPException(status_code=400, detail="URL does not point to a valid image.")
            
        extension = ".jpg"
        if 'png' in content_type: extension = '.png'
        elif 'webp' in content_type: extension = '.webp'
        elif 'gif' in content_type: extension = '.gif'

        temp_filename = f"g-search-{uuid.uuid4()}{extension}"
        temp_filepath = TEMP_UPLOAD_DIR / temp_filename

        with temp_filepath.open("wb") as buffer:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)

        full_path_str = str(temp_filepath.resolve())
        
        return {
            "temp_image_name": temp_filename,
            "filepath": full_path_str,
            "url": f"/images/{base64.urlsafe_b64encode(full_path_str.encode('utf-8')).decode('utf-8')}"
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download image from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during image download: {e}")
# ## END: GOOGLE IMAGE SEARCH API ENDPOINTS ##

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory="."), name="static")


_CURRENT_DIR_PARENT = os.path.dirname(os.path.abspath(__file__))
COMMON_PARENT_DIR = os.path.dirname(_CURRENT_DIR_PARENT)
if COMMON_PARENT_DIR not in sys.path:
    sys.path.insert(0, COMMON_PARENT_DIR)

try:
    from function import translate_query, enhance_query, expand_query_parallel
    print("--- Gateway Server: Đã import thành công các hàm xử lý truy vấn. ---")
except ImportError:
    print("!!! CẢNH BÁO: Không thể import các hàm xử lý truy vấn. Sử dụng hàm DUMMY. !!!")
    def enhance_query(q: str) -> str: return q
    def expand_query_parallel(q: str) -> list[str]: return [q]
    async def translate_query(q: str) -> str: return q

# --- Cấu hình DRES và hệ thống ---
DRES_BASE_URL = "http://192.168.28.151:5000"
VIDEO_BASE_DIR = "/workspace/mlcv1/Datasets/HCMAI25/batch1/video"
IMAGE_BASE_PATH = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Retrieval/Keyframes/webp_keyframes"

BEIT3_WORKER_URL = "http://model-workers2:8001/embed"
BGE_WORKER_URL = "http://model-workers:8002/embed"
UNITE_WORKER_URL = "http://model-workers:8005/embed"
OPS_MM_WORKER_URL = "http://model-workers:8004/embed"
IMAGE_GEN_WORKER_URL = "http://localhost:8004/generate"

ELASTICSEARCH_HOST = "http://elasticsearch2:9200"
OCR_ASR_INDEX_NAME = "vongsotuyen_batch1"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

BEIT3_COLLECTION_NAME = "beit3_batch1_filter_2"
BGE_COLLECTION_NAME = "bge_batch1_filter_2"
UNITE_COLLECTION_NAME = "Unite_Batch1_with_filepath_filter"
UNITE_FUSION_COLLECTION_NAME = "Unite_Fusion_Batch1_with_filepath_filter"
OPS_MM_COLLECTION_NAME = "Mm_embed_Batch1_with_filepath_filter"

MODEL_WEIGHTS = {"beit3": 0.3, "bge": 0.1, "unite": 0.3, "ops_mm": 0.3}
SEARCH_DEPTH = 1000
TOP_K_RESULTS = 1000
MAX_SEQUENCES_TO_RETURN = 500
SEARCH_DEPTH_PER_STAGE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720

# --- PERFORMANCE OPTIMIZATION: Tuned Milvus search parameters ---
# Lowering 'ef' provides a significant speed boost with minimal accuracy loss for HNSW.
SEARCH_PARAMS = {
    "HNSW": {"metric_type": "IP", "params": {"ef": 1024}}, # Lowered from 1024
    "IVF_FLAT": {"metric_type": "COSINE", "params": {"nprobe": 24}},
    "SCANN": {"metric_type": "COSINE", "params": {"nprobe": 128}},
    "DEFAULT": {"metric_type": "IP", "params": {}}
}
# ---------------------------------------------------------------

COLLECTION_TO_INDEX_TYPE = {
    BEIT3_COLLECTION_NAME: "HNSW",
    BGE_COLLECTION_NAME: "HNSW",
    UNITE_COLLECTION_NAME: "HNSW",
    UNITE_FUSION_COLLECTION_NAME: "HNSW",
    OPS_MM_COLLECTION_NAME: "HNSW"
}

es = None
OBJECT_COUNTS_DF: Optional[pl.DataFrame] = None
OBJECT_POSITIONS_DF: Optional[pl.DataFrame] = None
beit3_collection: Optional[Collection] = None
bge_collection: Optional[Collection] = None
unite_collection: Optional[Collection] = None
unite_collection_fusion: Optional[Collection] = None
ops_mm_collection: Optional[Collection] = None

FRAME_CONTEXT_CACHE_FILE = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/DataPreprocessing/KF/frame_context_cache.json"
FRAME_CONTEXT_CACHE: Optional[Dict[str, List[str]]] = None

# ## TEAMWORK: Connection Manager for WebSockets ##
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
# ## END TEAMWORK ##
trake_panel_state: List[Dict[str, Any]] = [] 

# --- Pydantic Models ---
class ObjectCountFilter(BaseModel): conditions: Dict[str, str] = {}
class PositionBox(BaseModel): label: str; box: List[float]
class ObjectPositionFilter(BaseModel): boxes: List[PositionBox] = []
class ObjectFilters(BaseModel): counting: Optional[ObjectCountFilter] = None; positioning: Optional[ObjectPositionFilter] = None
class StageData(BaseModel):
    query: str
    enhance: bool
    expand: bool
    use_unite_fusion: bool = False
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    query_image_name: Optional[str] = None
    generated_image_name: Optional[str] = None

class TemporalSearchRequest(BaseModel):
    stages: list[StageData]
    models: List[str] = ["beit3", "bge", "unite"]
    cluster: bool = False
    filters: Optional[ObjectFilters] = None
    ambiguous: bool = False
    page: int = 1
    page_size: int = 30

class ProcessQueryRequest(BaseModel):
    query: str
    enhance: bool = False
    expand: bool = False

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_name: Optional[str] = None
    image_search_text: Optional[str] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    models: List[str] = ["beit3", "bge", "unite"]
    filters: Optional[ObjectFilters] = None
    enhance: bool = False
    expand: bool = False
    use_unite_fusion: bool = False
    generated_image_name: Optional[str] = None
    page: int = 1
    page_size: int = 30
class CheckFramesRequest(BaseModel): base_filepath: str
class DRESLoginRequest(BaseModel): username: str; password: str
class DRESSubmitRequest(BaseModel):
    sessionId: str
    evaluationId: str
    video_id: str
    filepath: str
    frame_id: Optional[int] = None

class VideoInfoResponse(BaseModel):
    fps: float

@app.on_event("startup")
def startup_event():
    global es, OBJECT_COUNTS_DF, OBJECT_POSITIONS_DF, beit3_collection, bge_collection, unite_collection, unite_collection_fusion, ops_mm_collection
    try:
        print("--- Loading cache json ---")
        load_frame_context_cache_from_json()
        print("--- Loading cache json successfully ---")
    
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("--- Milvus connection successful. ---")
        print("--- Loading Milvus collections into memory... ---")
        
        collections_to_load = {
            "BEiT3": (BEIT3_COLLECTION_NAME, "beit3"),
            "BGE": (BGE_COLLECTION_NAME, "bge"),
            "Unite": (UNITE_COLLECTION_NAME, "unite"),
            "UniteFusion": (UNITE_FUSION_COLLECTION_NAME, "unite_fusion"),
            "OpsMM": (OPS_MM_COLLECTION_NAME, "ops_mm")
        }

        for name, (col_name, var_name) in collections_to_load.items():
            if utility.has_collection(col_name):
                collection = Collection(col_name)
                collection.load()
                
                if var_name == "beit3":
                    beit3_collection = collection
                elif var_name == "bge":
                    bge_collection = collection
                elif var_name == "unite":
                    unite_collection = collection
                elif var_name == "unite_fusion":
                    unite_collection_fusion = collection
                elif var_name == "ops_mm": # <-- THÊM KHỐI LỆNH NÀY
                    ops_mm_collection = collection
                    
                print(f"--- Collection '{col_name}' (for {name}) loaded successfully. ---")
            else:
                print(f"!!! WARNING: Collection '{col_name}' (for {name}) not found. !!!")

    except Exception as e:
        print(f"FATAL: Could not connect to or load from Milvus. Error: {e}")
        traceback.print_exc()
        
    try:
        es = Elasticsearch(ELASTICSEARCH_HOST)
        if es.ping():
            print("--- Elasticsearch connection successful. ---")
        else:
            print("FATAL: Could not connect to Elasticsearch.")
            es = None
    except Exception as e:
        print(f"FATAL: Could not connect to Elasticsearch. Error: {e}")
        es = None
        
    try:
        print("--- Loading object detection data... ---")
        counts_path = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Object/object_counts.parquet"
        positions_path = "/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/Dataset/Object/object_positions.parquet"
        counts_df = pl.read_parquet(counts_path)
        OBJECT_COUNTS_DF = counts_df.with_columns(pl.col("image_name").str.split(".").list.first().alias("name_stem"))
        positions_df = pl.read_parquet(positions_path)
        OBJECT_POSITIONS_DF = positions_df.with_columns([((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("bbox_area"), pl.col("image_name").str.split(".").list.first().alias("name_stem")])
        print(f"--- Object data loaded. ---")
    except Exception as e:
        print(f"!!! WARNING: Could not load object parquet files. Filtering disabled. Error: {e} !!!")
        OBJECT_COUNTS_DF = None
        OBJECT_POSITIONS_DF = None

# --- Helper Functions ---

def load_frame_context_cache_from_json():
    """
    Tải cache tra cứu context từ file JSON đã được tính toán trước.
    """
    global FRAME_CONTEXT_CACHE
    if not os.path.exists(FRAME_CONTEXT_CACHE_FILE):
        print("\n" + "="*80)
        print(f"!!! CẢNH BÁO QUAN TRỌNG: Không tìm thấy file cache '{FRAME_CONTEXT_CACHE_FILE}'.")
        print("!!! Tính năng 'Xem Context' sẽ không hoạt động tối ưu.")
        print(f"!!! Hãy chạy script 'create_context_cache.py' để tạo file này.")
        print("="*80 + "\n")
        FRAME_CONTEXT_CACHE = {} # Khởi tạo rỗng để tránh lỗi khi server chạy
        return

    print(f"--- Đang tải Frame Context Cache từ '{FRAME_CONTEXT_CACHE_FILE}'... ---")
    start_time = time.time()
    try:
        with open(FRAME_CONTEXT_CACHE_FILE, 'r', encoding='utf-8') as f:
            FRAME_CONTEXT_CACHE = json.load(f)
        end_time = time.time()
        print(f"--- Tải Frame Context Cache thành công sau {end_time - start_time:.2f}s. ---")
        print(f"--- Đã cache context cho {len(FRAME_CONTEXT_CACHE)} frames. ---")
    except Exception as e:
        print(f"!!! LỖI NGHIÊM TRỌNG: Không thể đọc hoặc phân tích file cache '{FRAME_CONTEXT_CACHE_FILE}'. Lỗi: {e}")
        FRAME_CONTEXT_CACHE = {}

# ## END: TỐI ƯU HÓA CONTEXT VIEW ##

def get_video_fps(video_path: str) -> float:
    """Gets the FPS of a video file using OpenCV. Returns a default if it fails."""
    if not os.path.exists(video_path):
        print(f"Warning: Video path does not exist for FPS check: {video_path}. Returning default.")
        return 30.0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file to get FPS: {video_path}. Returning default.")
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 0 else 30.0
    except Exception as e:
        print(f"Error getting FPS for {video_path}: {e}. Returning default.")
        return 30.0

def process_and_cluster_results_optimized(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    shots_by_video = defaultdict(list)
    for res in results:
        if not all(k in res for k in ['video_id', 'shot_id']):
            continue
        try:
            res['shot_id_int'] = int(str(res['shot_id']))
            shots_by_video[res['video_id']].append(res)
        except (ValueError, TypeError):
            continue
    all_clusters = []
    for video_id, shots_in_video in shots_by_video.items():
        if not shots_in_video:
            continue
        sorted_shots = sorted(shots_in_video, key=lambda x: x['shot_id_int'])
        if not sorted_shots:
            continue
        current_cluster = [sorted_shots[0]]
        for i in range(1, len(sorted_shots)):
            current_shot = sorted_shots[i]
            last_shot_in_cluster = current_cluster[-1]
            if current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int'] or current_shot['shot_id_int'] == last_shot_in_cluster['shot_id_int'] + 1:
                current_cluster.append(current_shot)
            else:
                all_clusters.append(current_cluster)
                current_cluster = [current_shot]
        if current_cluster:
            all_clusters.append(current_cluster)
    if not all_clusters:
        return []
    processed_clusters = []
    for cluster_shots in all_clusters:
        if not cluster_shots:
            continue
        sorted_cluster_shots = sorted(
            cluster_shots,
            key=lambda x: x.get('rrf_score', x.get('score', 0)),
            reverse=True
        )
        best_shot = sorted_cluster_shots[0]
        max_score = best_shot.get('rrf_score', best_shot.get('score', 0))
        processed_clusters.append({
            "cluster_score": max_score,
            "shots": sorted_cluster_shots,
            "best_shot": best_shot
        })
    return sorted(processed_clusters, key=lambda x: x['cluster_score'], reverse=True)

def get_filename_stem(filepath: str) -> Optional[str]:
    if not filepath: return None
    try: return os.path.splitext(os.path.basename(filepath))[0]
    except Exception: return None

# --- FUNCTION 2 TO REPLACE ---
# This global variable will cache the results of our expensive vectorized checks
temporal_filter_cache = {}

def is_temporal_sequence_valid(sequence: Dict, filters: ObjectFilters) -> bool:
    print("\n--- [DEBUG] Checking Temporal Sequence (Corrected Logic) ---")
    
    # Create a checklist of all conditions that this sequence must satisfy.
    checklist = set()
    if filters.counting and filters.counting.conditions:
        for obj, cond in filters.counting.conditions.items(): checklist.add(f"count_{obj}_{cond}")
    if filters.positioning and filters.positioning.boxes:
        for i, pbox in enumerate(filters.positioning.boxes): checklist.add(f"pos_{i}_{pbox.label}")
    if not checklist: return True # If there are no filters, the sequence is always valid.

    # Get all the unique frame stems that belong ONLY to this sequence.
    sequence_filepaths = {s['filepath'] for s in sequence.get('shots', []) if 'filepath' in s}
    for cluster in sequence.get('clusters', []):
        for shot in cluster.get('shots', []):
            if 'filepath' in shot: sequence_filepaths.add(shot['filepath'])
    if not sequence_filepaths: return False
    
    sequence_stems = {get_filename_stem(p) for p in sequence_filepaths if p}
    print(f"Sequence stems to check: {sequence_stems}")

    # Check each frame in this sequence to see if it can satisfy any of the remaining conditions.
    for stem in sequence_stems:
        if not checklist: break # Optimization: If the checklist is empty, we are done.
        
        # Check counting conditions for this one frame
        if filters.counting and OBJECT_COUNTS_DF is not None:
            frame_counts = OBJECT_COUNTS_DF.filter(pl.col("name_stem") == stem)
            if not frame_counts.is_empty():
                for obj, cond_str in filters.counting.conditions.items():
                    key = f"count_{obj}_{cond_str}"
                    if key in checklist:
                        op, val = parse_condition(cond_str)
                        actual_value = frame_counts.row(0, named=True)[obj]
                        print(f"  - Checking count for '{stem}': Is {obj} ({actual_value}) {cond_str}? ", end="")
                        if op and val is not None and obj in frame_counts.columns and op(actual_value, val):
                            print("YES. Condition met.")
                            checklist.remove(key)
                        else:
                            print("NO.")

        # Check positioning conditions for this one frame
        if filters.positioning and OBJECT_POSITIONS_DF is not None:
            frame_positions = OBJECT_POSITIONS_DF.filter(pl.col("name_stem") == stem)
            if not frame_positions.is_empty():
                for i, p_box in enumerate(filters.positioning.boxes):
                    key = f"pos_{i}_{p_box.label}"
                    if key in checklist and p_box.label in frame_positions['object']:
                        # --- FIX: Reverted to older, more compatible Polars syntax for min/max ---
                        intersect_area = (pl.min([pl.col("x_max"), pl.lit(p_box.box[2] * IMAGE_WIDTH)]) - pl.max([pl.col("x_min"), pl.lit(p_box.box[0] * IMAGE_WIDTH)])).clip_min(0) * \
                                       (pl.min([pl.col("y_max"), pl.lit(p_box.box[3] * IMAGE_HEIGHT)]) - pl.max([pl.col("y_min"), pl.lit(p_box.box[1] * IMAGE_HEIGHT)])).clip_min(0)
                        
                        match_df = frame_positions.filter(pl.col("object") == p_box.label).with_columns(
                            overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
                        ).filter(pl.col("overlap_ratio") >= 0.75)
                        
                        print(f"  - Checking position for '{stem}': Does label '{p_box.label}' match? ", end="")
                        if not match_df.is_empty():
                            print("YES. Condition met.")
                            checklist.remove(key)
                        else:
                            print("NO.")

    is_valid = not checklist
    print(f"Final Sequence Result: {'VALID' if is_valid else 'INVALID'}. Remaining checklist: {checklist if not is_valid else 'None'}")
    return is_valid


def parse_condition(condition_str: str) -> tuple[Any, int]:
    try: return operator.ge, int(condition_str)
    except ValueError:
        op_map = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "=": operator.eq}
        for op_str in [">=", "<=", "==", ">", "<", "="]:
            if condition_str.startswith(op_str):
                try: return op_map[op_str], int(condition_str[len(op_str):])
                except (ValueError, TypeError): return None, None
    return None, None

def get_valid_filepaths_for_strict_search(all_filepaths: set, filters: ObjectFilters) -> set:
    print("\n--- [DEBUG] Applying OPTIMIZED STRICT Object Filters ---")
    print(f"Filter Conditions: {filters.dict()}")
    
    candidate_stems = {get_filename_stem(p) for p in all_filepaths if p}
    if not candidate_stems: return set()

    print(f"Initial candidates: {len(all_filepaths)} (Unique stems: {len(candidate_stems)})")

    # Start with a DataFrame of all possible valid stems
    valid_stems_df = pl.DataFrame({"name_stem": list(candidate_stems)})

    # --- OPTIMIZED COUNTING FILTER ---
    if filters.counting and OBJECT_COUNTS_DF is not None and filters.counting.conditions:
        expressions = []
        for obj, cond_str in filters.counting.conditions.items():
            op, val = parse_condition(cond_str)
            if op and val is not None and obj in OBJECT_COUNTS_DF.columns:
                expressions.append(op(pl.col(obj), val))
        
        if expressions:
            # OPTIMIZATION: Join once, filter once. This is highly vectorized.
            count_matches_df = OBJECT_COUNTS_DF.lazy().join(
                valid_stems_df.lazy(), on="name_stem", how="inner"
            ).filter(
                pl.all_horizontal(expressions)
            ).select("name_stem").collect()
            
            valid_stems_df = count_matches_df
            print(f"After COUNTING filter: {len(valid_stems_df)} candidates remain.")

    # --- OPTIMIZED POSITIONING FILTER ---
    if filters.positioning and OBJECT_POSITIONS_DF is not None and filters.positioning.boxes:
        # If there are no more valid stems, we can stop early.
        if valid_stems_df.is_empty():
            print("--- [DEBUG] Strict Filtering Complete (no candidates left for positioning) ---")
            return set()

        # OPTIMIZATION: Filter the huge positions table only ONCE.
        positions_subset_df = OBJECT_POSITIONS_DF.lazy().join(
            valid_stems_df.lazy(), on="name_stem", how="inner"
        ).collect()

        stems_satisfying_all_boxes = set(valid_stems_df["name_stem"])

        for p_box in filters.positioning.boxes:
            # This logic is now vectorized across the entire positions_subset_df
            user_x_min_lit, user_y_min_lit, user_x_max_lit, user_y_max_lit = [pl.lit(v) for v in [p_box.box[0] * IMAGE_WIDTH, p_box.box[1] * IMAGE_HEIGHT, p_box.box[2] * IMAGE_WIDTH, p_box.box[3] * IMAGE_HEIGHT]]
            
            # FIX: Use pl.min/max for compatibility with older Polars versions
            intersect_area = (pl.min([pl.col("x_max"), user_x_max_lit]) - pl.max([pl.col("x_min"), user_x_min_lit])).clip_min(0) * \
                           (pl.min([pl.col("y_max"), user_y_max_lit]) - pl.max([pl.col("y_min"), user_y_min_lit])).clip_min(0)
            
            stems_with_match_for_this_box = positions_subset_df.filter(
                pl.col("object") == p_box.label
            ).with_columns(
                overlap_ratio=(intersect_area / pl.col("bbox_area")).fill_null(0)
            ).filter(
                pl.col("overlap_ratio") >= 0.75
            ).select("name_stem").unique()["name_stem"].to_list()
            
            stems_satisfying_all_boxes.intersection_update(stems_with_match_for_this_box)

        valid_stems_df = pl.DataFrame({"name_stem": list(stems_satisfying_all_boxes)})
        print(f"After POSITIONING filter: {len(valid_stems_df)} candidates remain.")

    final_valid_stems = set(valid_stems_df["name_stem"])
    print(f"--- [DEBUG] Strict Filtering Complete. Returning {len(final_valid_stems)} valid filepaths. ---")
    return {fp for fp in all_filepaths if get_filename_stem(fp) in final_valid_stems}


def search_milvus_sync(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    try:
        # print(collection)
        if not collection or not query_vectors:
            return []
        
        # --- Phần kiểm tra collection có được load hay chưa ---
        need_load = False
        try:
            state = utility.load_state(collection.name)
            if _MilvusLoadState and isinstance(state, _MilvusLoadState):
                need_load = (state != _MilvusLoadState.Loaded)
            else:
                state_name = getattr(state, "name", str(state))
                if str(state_name).lower() != "loaded" and str(state) != "2":
                    need_load = True
        except Exception as e:
            print(f"Could not determine load state for '{collection.name}' ({e}); will attempt load.")
            need_load = True
        
        if need_load:
            print(f"--- Collection '{collection.name}' not loaded. Loading... ---")
            collection.load()
            print(f"--- Collection '{collection.name}' loaded. ---")

        # --- Cấu hình và thực hiện search ---
        index_type = COLLECTION_TO_INDEX_TYPE.get(collection_name, "HNSW")
        search_params = SEARCH_PARAMS.get(index_type, SEARCH_PARAMS["HNSW"])
        
        anns_field = "vector_embedding"
        output_fields = ["frame_name", "video_id", "shot_id", "frame_id"]

        results = collection.search(
            data=query_vectors,
            anns_field=anns_field,
            param=search_params,
            limit=limit,
            output_fields=output_fields,
            expr=expr
        )
        
        print(results)
        
        # --- Xử lý kết quả trả về (ĐÃ SỬA LỖI THEO YÊU CẦU MỚI) ---
        final_results = []
        for one_query_hits in results:
            for hit in one_query_hits:
                entity = hit.entity
                
                # ## START: LOGIC SỬA LỖI - ÁP DỤNG QUY TẮC NHẤT QUÁN ##
                frame_name = entity.get("frame_name")
                
                # Bỏ qua nếu bản ghi không hợp lệ (không có frame_name)
                if not frame_name:
                    continue

                # Quy tắc nhất quán: Mọi frame_name từ Milvus đều là TÊN GỐC KHÔNG CÓ ĐUÔI.
                # Luôn luôn ghép nó với IMAGE_BASE_PATH và thêm đuôi .webp.
                # Logic này giờ đây áp dụng cho TẤT CẢ các collection.
                if frame_name.endswith(".webp"):
                    frame_name = frame_name[:-5]
                    
                filepath = os.path.join(IMAGE_BASE_PATH, f"{frame_name}.webp")
                # ## END: LOGIC SỬA LỖI ##

                final_results.append({
                    "filepath": filepath,
                    "score": hit.distance,
                    "video_id": entity.get("video_id"),
                    "frame_id": entity.get("frame_id"),
                    "shot_id": str(entity.get("shot_id"))
                })
        return final_results
        
    except Exception as e:
        print(f"ERROR during Milvus search on '{collection_name}': {e}")
        traceback.print_exc()
        return []

def search_ocr_on_elasticsearch_sync(keyword: str, limit: int = 500):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["ocr_text"]}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        return [{"filepath": hit['_source']['file_path'], "score": hit['_score'], "video_id": hit['_source']['video_id'], "shot_id": str(hit['_source']['shot_id']), "frame_id": hit['_source']['frame_id']} for hit in response["hits"]["hits"] if all(k in hit['_source'] for k in ['file_path', 'video_id', 'shot_id', 'frame_id'])]
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch OCR: {e}"); return []

def search_asr_on_elasticsearch_sync(keyword: str, limit: int = 500):
    if not es: return []
    query = {"query": {"multi_match": {"query": keyword, "fields": ["asr_text^3", "text^1"], "type": "best_fields", "fuzziness": "AUTO"}}}
    try:
        response = es.search(index=OCR_ASR_INDEX_NAME, body=query, size=limit)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit['_source']
            filepath = source.get('file_path') or source.get('filepath')
            if filepath and all(k in source for k in ['video_id', 'shot_id', 'frame_id']):
                results.append({"filepath": filepath, "score": hit['_score'], "video_id": source['video_id'], "shot_id": str(source['shot_id']), "frame_id": source['frame_id']})
        return results
    except NotFoundError: return []
    except Exception as e: print(f"Lỗi Elasticsearch ASR: {e}"); return []

async def search_milvus_async(collection: Collection, collection_name: str, query_vectors: list, limit: int, expr: str = None):
    return await asyncio.to_thread(search_milvus_sync, collection, collection_name, query_vectors, limit, expr)

async def search_ocr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_ocr_on_elasticsearch_sync, keyword, limit)

async def search_asr_on_elasticsearch_async(keyword: str, limit: int = 500):
    return await asyncio.to_thread(search_asr_on_elasticsearch_sync, keyword, limit)

def reciprocal_rank_fusion(results_lists: dict, weights: dict, k_rrf: int = 60):
    master_data = defaultdict(lambda: {"raw_scores": {}})
    for model_name, results in results_lists.items():
        if not results:
            continue
        sorted_results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            filepath = result.get('filepath')
            if not filepath:
                continue
            if 'metadata' not in master_data[filepath]:
                master_data[filepath]['metadata'] = result
            master_data[filepath]['raw_scores'][model_name] = {
                "score": result.get('score', 0.0),
                "rank": rank
            }
            
    if not master_data:
        return []
        
    final_results = []
    for filepath, data in master_data.items():
        rrf_score = 0.0
        for model_name, score_info in data['raw_scores'].items():
            rrf_score += weights.get(model_name, 1.0) * (1.0 / (k_rrf + score_info['rank']))
        
        final_item = data['metadata']
        final_item['rrf_score'] = rrf_score
        final_item.pop('score', None)
        final_results.append(final_item)
        
    return sorted(final_results, key=lambda x: x['rrf_score'], reverse=True)

def package_response_with_urls(data: List[Dict[str, Any]], base_url: str):
    for item in data:
        if not isinstance(item, dict): continue
        def process_shot(shot_dict):
            if isinstance(shot_dict, dict) and 'filepath' in shot_dict:
                current_path = shot_dict.get('filepath')
                if current_path:
                    if not current_path.startswith(ALLOWED_BASE_DIR):
                        current_path = os.path.join(IMAGE_BASE_PATH, os.path.basename(current_path))
                    
                    shot_dict['filepath'] = current_path
                    if 'url' not in shot_dict:
                         shot_dict['url'] = f"{base_url}images/{base64.urlsafe_b64encode(current_path.encode('utf-8')).decode('utf-8')}"
        if 'shots' in item and isinstance(item['shots'], list):
            for shot in item['shots']: process_shot(shot)
        if 'best_shot' in item: process_shot(item['best_shot'])
        if 'clusters' in item and isinstance(item['clusters'], list):
            for cluster in item['clusters']:
                 if isinstance(cluster, dict):
                    if 'shots' in cluster and isinstance(cluster['shots'], list):
                        for shot in cluster['shots']: process_shot(shot)
                    if 'best_shot' in cluster: process_shot(cluster['best_shot'])
    return ORJSONResponse(content={"results": data})

async def get_embeddings_for_query(
    client: httpx.AsyncClient,
    text_queries: List[str],
    image_content: Optional[bytes],
    models: List[str],
    query_image_info: Optional[Dict] = None,
    is_fusion: bool = False
) -> Dict[str, List[List[float]]]:
    if image_content:
        return await get_embeddings_for_query_from_worker(client, text_queries, image_content, models, query_image_info, is_fusion)

    global embedding_cache
    if len(embedding_cache) > CACHE_MAX_SIZE:
        embedding_cache.clear()

    cache_key = f"{'|'.join(sorted(models))}:{'|'.join(text_queries)}"
    if cache_key in embedding_cache:
        print(f"--- EMBEDDING CACHE HIT for key: {cache_key[:80]}... ---")
        return embedding_cache[cache_key]
    
    print(f"--- EMBEDDING CACHE MISS for key: {cache_key[:80]}... ---")
    results = await get_embeddings_for_query_from_worker(client, text_queries, image_content, models, query_image_info, is_fusion)
    embedding_cache[cache_key] = results
    return results

async def get_embeddings_for_query_from_worker(
    client: httpx.AsyncClient,
    text_queries: List[str],
    image_content: Optional[bytes],
    models: List[str],
    query_image_info: Optional[Dict] = None,
    is_fusion: bool = False
) -> Dict[str, List[List[float]]]:
    tasks = []
    model_url_map = {"beit3": BEIT3_WORKER_URL, "bge": BGE_WORKER_URL, "unite": UNITE_WORKER_URL, "ops_mm": OPS_MM_WORKER_URL}
    async def get_model_embedding(model_name: str) -> tuple[str, list]:
        url = model_url_map.get(model_name)
        if not url: return model_name, []
        try:
            embeddings = []
            if model_name == 'unite' and is_fusion and image_content and text_queries:
                print("--- GATEWAY: Preparing FUSION request for Unite worker ---")
                data = {'text_query': text_queries[0]}
                files = {'image_file': (query_image_info['filename'], image_content, query_image_info['content_type'])}
                resp = await client.post(url, files=files, data=data, timeout=30.0)
                if resp.status_code == 200:
                    embeddings.extend(resp.json().get('embedding', []))
                else:
                    print(f"ERROR: Unite worker returned status {resp.status_code} with text: {resp.text}")
            else:
                queries = text_queries or [""]
                for q in queries:
                    data = {'text_query': q} if q else {}
                    files = None
                    if image_content:
                         files = {'image_file': (query_image_info['filename'], image_content, query_image_info['content_type'])}
                    resp = await client.post(url, files=files, data=data, timeout=20.0)
                    if resp.status_code == 200:
                        embeddings.extend(resp.json().get('embedding', []))
            return model_name, embeddings
        except Exception as e:
            print(f"Error getting embedding for {model_name}: {e}")
            return model_name, []
    for model in models:
        if model in model_url_map:
            tasks.append(get_model_embedding(model))
    results = await asyncio.gather(*tasks)
    return {model_name: vecs for model_name, vecs in results}

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    ui_path = os.path.join(BASE_DIR, "ui.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=500, detail="UI file not found")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global trake_panel_state # Declare we are using the global variable

    # When a new user connects, send them the current state of the Trake Panel
    try:
        await websocket.send_text(json.dumps({
            "type": "trake_sync",
            "data": trake_panel_state
        }))
    except Exception as e:
        print(f"Initial sync failed for a client: {e}")

    try:
        while True:
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            msg_type = message.get("type")

            if msg_type == "new_frame": # Teamwork Panel Push
                await manager.broadcast(raw_data)

            elif msg_type == "remove_frame": # Teamwork Panel Remove
                await manager.broadcast(raw_data)

            elif msg_type == "clear_panel": # Teamwork Panel Clear
                await manager.broadcast(raw_data)
            
            # --- START: NEW TRAKE PANEL LOGIC ---
            elif msg_type == "trake_add":
                shot_data = message.get("data", {}).get("shot")
                if shot_data:
                    # Prevent duplicates by filepath
                    if not any(item['filepath'] == shot_data.get('filepath') for item in trake_panel_state):
                        trake_panel_state.append(shot_data)
                        await manager.broadcast(json.dumps({
                            "type": "trake_add",
                            "data": {"shot": shot_data}
                        }))

            elif msg_type == "trake_remove":
                filepath = message.get("data", {}).get("filepath")
                if filepath:
                    trake_panel_state = [item for item in trake_panel_state if item.get('filepath') != filepath]
                    await manager.broadcast(raw_data) # Forward the original remove message

            elif msg_type == "trake_reorder":
                # The client sends the new order of filepaths
                new_order_filepaths = message.get("data", {}).get("order")
                if isinstance(new_order_filepaths, list):
                    # Create a map for quick lookups
                    current_items_map = {item['filepath']: item for item in trake_panel_state}
                    # Rebuild the list based on the new order
                    new_state = [current_items_map[fp] for fp in new_order_filepaths if fp in current_items_map]
                    trake_panel_state = new_state
                    await manager.broadcast(raw_data) # Forward the reorder message
            
            elif msg_type == "trake_replace":
                # Client sends the filepath of the item to replace and the new shot data
                data = message.get("data", {})
                filepath_to_replace = data.get("filepath")
                new_shot_data = data.get("newShot")
                if filepath_to_replace and new_shot_data:
                    for i, item in enumerate(trake_panel_state):
                        if item.get("filepath") == filepath_to_replace:
                            trake_panel_state[i] = new_shot_data
                            break
                    await manager.broadcast(raw_data) # Forward the replace message
            # --- END: NEW TRAKE PANEL LOGIC ---
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)

# ## PERFORMANCE OPTIMIZATION: Asynchronous Query Processing with Cache ##
@app.post("/process_query")
async def process_query(request_data: ProcessQueryRequest):
    """
    Xử lý một truy vấn văn bản gốc dựa trên các tùy chọn enhance và expand.
    - enhance=True: Dùng AI để dịch và tối ưu hóa (chất lượng cao).
    - expand=True: Dùng googletrans dịch rồi expand.
    - Mặc định: Dùng googletrans chỉ để dịch.
    """
    query = request_data.query
    if not query:
        return {"processed_query": ""}

    cache_key = f"{query}|{request_data.enhance}|{request_data.expand}"
    if cache_key in processed_query_cache:
        print(f"--- QUERY CACHE HIT for: {query[:50]}... ---")
        return {"processed_query": processed_query_cache[cache_key]}
    print(f"--- QUERY CACHE MISS for: {query[:50]}... ---")

    processed_query = ""
    
    if request_data.enhance:
        print(f"--- Action: Enhancing '{query[:50]}...' using AI (high quality translate + optimize)")
        processed_query = await asyncio.to_thread(enhance_query, query)
    
    elif request_data.expand:
        print(f"--- Action: Expanding '{query[:50]}...' using fast translate")
        translated_query = await translate_query(query)
        processed_query = await asyncio.to_thread(expand_query_parallel, translated_query)
    
    else:
        print(f"--- Action: Translating '{query[:50]}...' using fast translate")
        processed_query = await translate_query(query)

    if isinstance(processed_query, list):
        processed_query = "\n".join(processed_query)

    processed_query_cache[cache_key] = processed_query
    return {"processed_query": processed_query}

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    extension = Path(image.filename).suffix
    temp_filename = f"{uuid.uuid4()}{extension}"
    temp_filepath = TEMP_UPLOAD_DIR / temp_filename
    try:
        with temp_filepath.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    finally:
        image.file.close()
    return {"temp_image_name": temp_filename}

@app.post("/dres/login")
async def dres_login(login_data: DRESLoginRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{DRES_BASE_URL}/api/v2/login", json=login_data.dict())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES login failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

@app.get("/dres/list_evaluations")
async def dres_list_evaluations(session: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{DRES_BASE_URL}/api/v2/client/evaluation/list", params={"session": session})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"DRES list evaluations failed: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while contacting DRES: {e}")

@app.post("/dres/submit")
async def dres_submit(submit_data: DRESSubmitRequest):
    try:
        if submit_data.frame_id is None:
             raise ValueError("frame_id is missing from the submission data.")
        
        video_filename = submit_data.video_id
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_filename += '.mp4'
        
        video_full_path = os.path.join(VIDEO_BASE_DIR, video_filename)
        video_fps = await asyncio.to_thread(get_video_fps, video_full_path)
        
        time_ms = int((submit_data.frame_id / video_fps) * 1000)
        video_item_name = os.path.splitext(submit_data.video_id)[0]
        submission_body = {"answerSets": [{"answers": [{"mediaItemName": video_item_name, "start": time_ms, "end": time_ms}]}]}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DRES_BASE_URL}/api/v2/submit/{submit_data.evaluationId}",
                params={"session": submit_data.sessionId},
                json=submission_body
            )
            response.raise_for_status()
            
            try:
                submission_result = response.json()
                if submission_result.get("submission") == "CORRECT":
                    await manager.broadcast(json.dumps({"type": "clear_panel", "status": "success"}))
            except Exception as e:
                print(f"Error broadcasting clear panel message: {e}")
                
            return response.json()
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"DRES submission failed: {e.response.text}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while contacting DRES: {e}")

class ImageGenTextRequest(BaseModel):
    query: str
    enhance: bool = False
    expand: bool = False

@app.post("/generate_image_from_text")
async def generate_image_from_text(request_data: ImageGenTextRequest):
    if not request_data.query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    processed_text = await translate_query(request_data.query)
    if request_data.expand:
        expanded_list = await asyncio.to_thread(expand_query_parallel, processed_text)
        processed_text = " ".join(expanded_list)
    if request_data.enhance:
        processed_text = await asyncio.to_thread(enhance_query, processed_text)
    print(f"--- Image Gen: Using processed query for generation: '{processed_text}'")
    try:
        response = requests.post(IMAGE_GEN_WORKER_URL, json={"query": processed_text}, timeout=60.0)
        response.raise_for_status()
        image_content = response.content
        temp_filename = f"gen_{uuid.uuid4()}.png"
        temp_filepath = TEMP_UPLOAD_DIR / temp_filename
        with temp_filepath.open("wb") as f:
            f.write(image_content)
        encoded_path = base64.urlsafe_b64encode(str(temp_filepath).encode('utf-8')).decode('utf-8')
        image_url = f"/images/{encoded_path}"
        return {"temp_image_name": temp_filename, "image_url": image_url}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to contact image generation service: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image generation process: {e}")

# --- Search Endpoints ---
@app.post("/search")
async def search_unified(request: Request, search_data: str = Form(...), query_image: Optional[UploadFile] = File(None)):
    start_total_time = time.time()
    timings = {}
    try:
        search_data_model = UnifiedSearchRequest.parse_raw(search_data)
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid search data format: {e}")
    milvus_expr = None
    es_results_for_standalone_search = []
    is_filter_search = bool(search_data_model.ocr_query or search_data_model.asr_query)
    if is_filter_search:
        start_es = time.time()
        es_tasks = []
        if search_data_model.ocr_query:
            es_tasks.append(search_ocr_on_elasticsearch_async(search_data_model.ocr_query, limit=SEARCH_DEPTH * 5))
        if search_data_model.asr_query:
            es_tasks.append(search_asr_on_elasticsearch_async(search_data_model.asr_query, limit=SEARCH_DEPTH * 5))
        es_results_lists = await asyncio.gather(*es_tasks)
        es_res_map = {res['filepath']: res for res_list in es_results_lists for res in res_list}
        es_results_for_standalone_search = list(es_res_map.values())
        timings["ocr_asr_filtering_s"] = time.time() - start_es
        if not es_results_for_standalone_search:
            response = package_response_with_urls([], str(request.base_url))
            response_content = json.loads(response.body)
            response_content.update({"processed_query": "", "total_results": 0, "timing_info": {**timings, "total_request_s": time.time() - start_total_time}})
            return ORJSONResponse(content=response_content)
        
        candidate_frame_names = [os.path.splitext(os.path.basename(res['filepath']))[0] for res in es_results_for_standalone_search]
        if candidate_frame_names:
            formatted_names = [f'"{name}"' for name in candidate_frame_names]
            milvus_expr = f'frame_name in [{",".join(formatted_names)}]'

    start_query_proc = time.time()
    final_queries_to_embed = []
    processed_query_for_ui = ""
    image_content, query_image_info = None, None
    models_to_use = search_data_model.models
    is_image_search = bool(query_image or search_data_model.query_image_name)
    is_gen_image_search = bool(search_data_model.generated_image_name)
    
    if is_gen_image_search:
        models_to_use = ["unite"]
        base_query = await translate_query(search_data_model.query_text)
        final_queries_to_embed = [base_query]
        temp_filepath = TEMP_UPLOAD_DIR / search_data_model.generated_image_name
        if temp_filepath.is_file():
            image_content = temp_filepath.read_bytes()
            query_image_info = {"filename": search_data_model.generated_image_name, "content_type": "image/png"}
        else:
            print(f"WARNING: Generated image file not found: {temp_filepath}")
    elif is_image_search:
        models_to_use = ["bge"]
        if search_data_model.image_search_text:
            translated_image_text = await translate_query(search_data_model.image_search_text)
            final_queries_to_embed = [translated_image_text]
        if query_image:
            image_content = await query_image.read()
            query_image_info = {"filename": query_image.filename, "content_type": query_image.content_type}
        elif search_data_model.query_image_name:
            temp_filepath = TEMP_UPLOAD_DIR / search_data_model.query_image_name
            if temp_filepath.is_file():
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": search_data_model.query_image_name, "content_type": "image/jpeg"}
            else:
                print(f"WARNING: Temporary image file not found: {temp_filepath}")
    elif search_data_model.query_text:
        processed_query_response = await process_query(ProcessQueryRequest(query=search_data_model.query_text, enhance=search_data_model.enhance, expand=search_data_model.expand))
        processed_query_for_ui = processed_query_response["processed_query"]
        final_queries_to_embed = [processed_query_for_ui]

    timings["query_processing_s"] = time.time() - start_query_proc
    
    is_primary_search = bool(final_queries_to_embed or image_content)
    final_fused_results = []
    if is_primary_search:
        if is_filter_search and not milvus_expr:
            final_fused_results = []
        else:
            start_embed = time.time()
            async with httpx.AsyncClient() as client:
                results_by_model = await get_embeddings_for_query(client, final_queries_to_embed, image_content, models_to_use, query_image_info, is_fusion=is_gen_image_search )
            timings["embedding_generation_s"] = time.time() - start_embed
            if any(results_by_model.values()):
                start_milvus = time.time()
                milvus_tasks = []
                models_in_task_order = [] # <-- Biến mới để theo dõi thứ tự

                # Thêm task theo thứ tự nhất quán
                if "beit3" in models_to_use and results_by_model.get("beit3"):
                    milvus_tasks.append(search_milvus_async(beit3_collection, BEIT3_COLLECTION_NAME, results_by_model["beit3"], SEARCH_DEPTH, expr=milvus_expr))
                    models_in_task_order.append("beit3")

                if "bge" in models_to_use and results_by_model.get("bge"):
                    milvus_tasks.append(search_milvus_async(bge_collection, BGE_COLLECTION_NAME, results_by_model["bge"], SEARCH_DEPTH, expr=milvus_expr))
                    models_in_task_order.append("bge")
                    
                if "unite" in models_to_use and results_by_model.get("unite"):
                    is_fusion = search_data_model.use_unite_fusion
                    unite_col = unite_collection_fusion if is_fusion else unite_collection
                    unite_name = UNITE_FUSION_COLLECTION_NAME if is_fusion else UNITE_COLLECTION_NAME
                    milvus_tasks.append(search_milvus_async(unite_col, unite_name, results_by_model["unite"], SEARCH_DEPTH, expr=milvus_expr))
                    models_in_task_order.append("unite")

                if "ops_mm" in models_to_use and results_by_model.get("ops_mm"):
                    milvus_tasks.append(search_milvus_async(ops_mm_collection, OPS_MM_COLLECTION_NAME, results_by_model["ops_mm"], SEARCH_DEPTH, expr=milvus_expr))
                    models_in_task_order.append("ops_mm")

                # Gọi gather và gán kết quả một cách an toàn
                milvus_results_list = await asyncio.gather(*milvus_tasks)
                timings["vector_search_s"] = time.time() - start_milvus
                
                start_post_proc = time.time()
                
                # Tạo dictionary kết quả dựa trên thứ tự đã lưu
                results_for_rrf = {model_name: result for model_name, result in zip(models_in_task_order, milvus_results_list)}
                
                milvus_weights = {m: w for m, w in MODEL_WEIGHTS.items() if m in models_to_use}
                final_fused_results = reciprocal_rank_fusion(results_for_rrf, milvus_weights)
                timings["post_processing_s"] = time.time() - start_post_proc
                
    elif is_filter_search:
        start_post_proc = time.time()
        for res in es_results_for_standalone_search:
            res['rrf_score'] = res.pop('score', 0.0)
        final_fused_results = sorted(es_results_for_standalone_search, key=lambda x: x.get('rrf_score', 0), reverse=True)
        timings["post_processing_s"] = time.time() - start_post_proc
    
    start_final_proc = time.time()
    if final_fused_results:
        clustered_results = process_and_cluster_results_optimized(final_fused_results)
        
        if search_data_model.filters and clustered_results:
            all_filepaths = {s['filepath'] for c in clustered_results for s in c.get('shots', []) if 'filepath' in s}
            valid_filepaths = await asyncio.to_thread(get_valid_filepaths_for_strict_search, all_filepaths, search_data_model.filters)
            
            final_results_all = []
            for cluster in clustered_results:
                valid_shots_in_cluster = [s for s in cluster.get('shots', []) if s.get('filepath') in valid_filepaths]
                if valid_shots_in_cluster:
                    new_cluster = cluster.copy()
                    new_cluster['shots'] = valid_shots_in_cluster
                    if new_cluster.get('best_shot') and new_cluster['best_shot'].get('filepath') not in valid_filepaths:
                        new_cluster['best_shot'] = max(valid_shots_in_cluster, key=lambda x: x.get('rrf_score', 0))
                    final_results_all.append(new_cluster)
        else:
            final_results_all = clustered_results
    else:
        final_results_all = []
    
    timings["final_processing_s"] = time.time() - start_final_proc

    total_results = len(final_results_all)
    start_index = (search_data_model.page - 1) * search_data_model.page_size
    end_index = start_index + search_data_model.page_size
    paginated_results = final_results_all[start_index:end_index]

    response = package_response_with_urls(paginated_results, str(request.base_url))
    response_content = json.loads(response.body)
    response_content.update({"processed_query": processed_query_for_ui, "total_results": total_results})
    timings["total_request_s"] = time.time() - start_total_time
    response_content["timing_info"] = timings
    return ORJSONResponse(content=response_content)

@app.post("/temporal_search", response_class=ORJSONResponse)
async def temporal_search(request_data: TemporalSearchRequest, request: Request):
    start_total_time = time.time()
    timings = {}
    models, stages, filters, ambiguous = request_data.models, request_data.stages, request_data.filters, request_data.ambiguous
    if not stages or not models:
        raise HTTPException(status_code=400, detail="Stages and models are required.")
    processed_queries_for_ui = []
    async def get_stage_results(client: httpx.AsyncClient, stage: StageData):
        has_vector_query = bool(stage.query or stage.query_image_name or stage.generated_image_name)
        has_ocr_asr_filter = bool(stage.ocr_query or stage.asr_query)
        milvus_expr = None
        all_es_results = []
        if has_ocr_asr_filter:
            es_tasks = []
            if stage.ocr_query: es_tasks.append(search_ocr_on_elasticsearch_async(stage.ocr_query, limit=SEARCH_DEPTH * 5))
            if stage.asr_query: es_tasks.append(search_asr_on_elasticsearch_async(stage.asr_query, limit=SEARCH_DEPTH * 5))
            es_results_lists = await asyncio.gather(*es_tasks)
            es_res_map = {res['filepath']: res for res_list in es_results_lists for res in res_list}
            all_es_results = list(es_res_map.values())
            if not all_es_results: return []
            
            candidate_frame_names = [os.path.basename(res['filepath']) for res in all_es_results]
            if candidate_frame_names:
                formatted_names = [f'"{name}"' for name in candidate_frame_names]
                milvus_expr = f'frame_name in [{",".join(formatted_names)}]'
            else:
                return []
                
        if has_vector_query:
            results_by_model = {}
            is_fusion_stage = bool(stage.generated_image_name and stage.query)
            if is_fusion_stage:
                models_for_stage = ["unite"]
                base_query = await translate_query(stage.query)
                queries_to_embed = [base_query]
                processed_queries_for_ui.append(f"Gen-Image Fusion: {stage.query}")
                temp_filepath = TEMP_UPLOAD_DIR / stage.generated_image_name
                if not temp_filepath.is_file(): return []
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": stage.generated_image_name, "content_type": "image/png"}
                results_by_model = await get_embeddings_for_query(client, queries_to_embed, image_content, models_for_stage, query_image_info, is_fusion=True)
            elif stage.query_image_name:
                temp_filepath = TEMP_UPLOAD_DIR / stage.query_image_name
                if not temp_filepath.is_file(): return []
                image_content = temp_filepath.read_bytes()
                query_image_info = {"filename": stage.query_image_name, "content_type": "image/jpeg"}
                models_for_image = ["bge"]
                results_by_model = await get_embeddings_for_query(client, [], image_content, models_for_image, query_image_info, is_fusion=False)
                processed_queries_for_ui.append(f"Image: {stage.query_image_name}")
            else:
                processed_query_response = await process_query(ProcessQueryRequest(query=stage.query, enhance=stage.enhance, expand=stage.expand))
                processed_query = processed_query_response["processed_query"]
                processed_queries_for_ui.append(processed_query)
                results_by_model = await get_embeddings_for_query(client, [processed_query], None, models, is_fusion=False)
            
            if has_ocr_asr_filter and not milvus_expr: return []
            if not any(results_by_model.values()): return []

            milvus_tasks = []
            if "beit3" in models and results_by_model.get("beit3"): milvus_tasks.append(search_milvus_async(beit3_collection, BEIT3_COLLECTION_NAME, results_by_model["beit3"], SEARCH_DEPTH_PER_STAGE, expr=milvus_expr))
            if "bge" in models and results_by_model.get("bge"): milvus_tasks.append(search_milvus_async(bge_collection, BGE_COLLECTION_NAME, results_by_model["bge"], SEARCH_DEPTH_PER_STAGE, expr=milvus_expr))
            if "unite" in models and results_by_model.get("unite"):
                unite_col = unite_collection_fusion if stage.use_unite_fusion else unite_collection
                unite_name = UNITE_FUSION_COLLECTION_NAME if stage.use_unite_fusion else UNITE_COLLECTION_NAME
                milvus_tasks.append(search_milvus_async(unite_col, unite_name, results_by_model["unite"], SEARCH_DEPTH_PER_STAGE, expr=milvus_expr))
                
            if "ops_mm" in models and results_by_model.get("ops_mm"):
                milvus_tasks.append(search_milvus_async(ops_mm_collection, OPS_MM_COLLECTION_NAME, results_by_model["ops_mm"], SEARCH_DEPTH_PER_STAGE, expr=milvus_expr))
            
            milvus_results = await asyncio.gather(*milvus_tasks)
            results_dict = {}
            model_idx = 0
            def assign_result(model_name):
                nonlocal model_idx
                if model_name in models:
                    results_dict[model_name] = milvus_results[model_idx] if len(milvus_results) > model_idx else []
                    model_idx += 1
            
            assign_result("beit3")
            assign_result("bge")
            assign_result("unite")
            assign_result("ops_mm") # <-- THÊM DÒNG NÀY
            
            return reciprocal_rank_fusion(results_dict, MODEL_WEIGHTS)
        elif has_ocr_asr_filter:
            processed_queries_for_ui.append(f"OCR/ASR: {stage.ocr_query or ''} / {stage.asr_query or ''}")
            for res in all_es_results:
                res['rrf_score'] = res.pop('score', 0.0)
            return sorted(all_es_results, key=lambda x: x.get('rrf_score', 0), reverse=True)
        else:
            return []

    start_stages = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        stage_tasks = [get_stage_results(client, stage) for stage in request_data.stages]
        all_stage_candidates = await asyncio.gather(*stage_tasks, return_exceptions=True)
    timings["stage_candidate_gathering_s"] = time.time() - start_stages
    
    valid_stage_results = [res for res in all_stage_candidates if isinstance(res, list)]
    if len(valid_stage_results) < len(stages):
        response = package_response_with_urls([], str(request.base_url))
        content = json.loads(response.body)
        content.update({"processed_queries": processed_queries_for_ui, "total_results": 0, "timing_info": {**timings, "total_request_s": time.time() - start_total_time}})
        return ORJSONResponse(content=content)
        
    clustered_results_by_stage = [process_and_cluster_results_optimized(res) for res in valid_stage_results]
    
    start_assembly = time.time()
    for stage_clusters in clustered_results_by_stage:
        for cluster in stage_clusters:
            if cluster.get('shots'):
                shot_ids_int = [s['shot_id_int'] for s in cluster['shots'] if 'shot_id_int' in s]
                if shot_ids_int:
                    cluster['min_shot_id'] = min(shot_ids_int)
                    cluster['max_shot_id'] = max(shot_ids_int)
                    cluster['video_id'] = cluster['best_shot']['video_id']
    clusters_by_video = defaultdict(lambda: defaultdict(list))
    for i, stage_clusters in enumerate(clustered_results_by_stage):
        for cluster in stage_clusters:
            if 'video_id' in cluster:
                clusters_by_video[cluster['video_id']][i].append(cluster)
    
    all_valid_sequences = []
    if not ambiguous:
        for video_id, video_stages in clusters_by_video.items():
            if len(video_stages) < len(stages):
                continue
            def find_sequences_recursive(stage_idx: int, current_sequence: list):
                if stage_idx == len(stages):
                    all_valid_sequences.append(list(current_sequence))
                    return
                for next_cluster in video_stages.get(stage_idx, []):
                    if not current_sequence or next_cluster.get('min_shot_id', -1) > current_sequence[-1].get('max_shot_id', -1):
                        current_sequence.append(next_cluster)
                        find_sequences_recursive(stage_idx + 1, current_sequence)
                        current_sequence.pop()
            find_sequences_recursive(0, [])
    else:
        for video_id, video_stages in clusters_by_video.items():
            if len(video_stages) < len(stages):
                continue
            best_clusters_for_video = []
            for stage_idx in range(len(stages)):
                stage_clusters = video_stages.get(stage_idx, [])
                if not stage_clusters:
                    best_clusters_for_video = []
                    break
                best_cluster = max(stage_clusters, key=lambda c: c.get('cluster_score', 0))
                best_clusters_for_video.append(best_cluster)
            if best_clusters_for_video:
                all_valid_sequences.append(best_clusters_for_video)
    timings["sequence_assembly_s"] = time.time() - start_assembly
    
    if not all_valid_sequences:
        response = package_response_with_urls([], str(request.base_url))
        content = json.loads(response.body)
        content.update({"processed_queries": processed_queries_for_ui, "total_results": 0, "timing_info": {**timings, "total_request_s": time.time() - start_total_time}})
        return ORJSONResponse(content=content)

    start_final_proc = time.time()
    
    TEMPORAL_PENALTY_WEIGHT = 0.05
    processed_sequences = []
    for cluster_seq in all_valid_sequences:
        if not cluster_seq: continue
        avg_score = sum(c.get('cluster_score', 0) for c in cluster_seq) / len(cluster_seq)
        
        total_temporal_gap = 0
        if len(cluster_seq) > 1 and not ambiguous:
            for i in range(len(cluster_seq) - 1):
                current_cluster_end = cluster_seq[i].get('max_shot_id', 0)
                next_cluster_start = cluster_seq[i+1].get('min_shot_id', 0)
                if next_cluster_start > current_cluster_end:
                    total_temporal_gap += (next_cluster_start - current_cluster_end)
        
        combined_score = avg_score / (1 + (total_temporal_gap * TEMPORAL_PENALTY_WEIGHT))
        
        shots_to_display = [c['best_shot'] for c in cluster_seq] if not ambiguous else [shot for c in cluster_seq for shot in c.get('shots', [])]
            
        processed_sequences.append({
            "combined_score": combined_score,
            "average_rrf_score": avg_score,
            "temporal_gap": total_temporal_gap,
            "clusters": cluster_seq,
            "shots": shots_to_display,
            "video_id": cluster_seq[0].get('video_id', 'N/A')
        })

    sequences_to_filter = sorted(processed_sequences, key=lambda x: x['combined_score'], reverse=True)

        # --- THIS IS THE KEY CHANGE for using the new corrected logic ---
    if filters and (filters.counting or filters.positioning):
        print(f"\n--- [DEBUG] Starting Temporal Sequence Validation for {len(sequences_to_filter)} sequences ---")
        # Use a simple list comprehension to call our new, correct validation function on each sequence.
        # This is now logically sound.
        final_sequences_all = [
            seq for seq in sequences_to_filter 
            if is_temporal_sequence_valid(seq, filters)
        ]
    else:
        final_sequences_all = sequences_to_filter
    # --- END OF CHANGE ---
    
    total_sequences = len(final_sequences_all)
    start_index = (request_data.page - 1) * request_data.page_size
    end_index = start_index + request_data.page_size
    paginated_sequences = final_sequences_all[start_index:end_index]
    
    timings["final_processing_s"] = time.time() - start_final_proc
    
    response = package_response_with_urls(paginated_sequences, str(request.base_url))
    content = json.loads(response.body)
    content.update({
        "processed_queries": processed_queries_for_ui,
        "is_temporal_search": not ambiguous,
        "is_ambiguous_search": ambiguous,
        "total_results": total_sequences
    })
    timings["total_request_s"] = time.time() - start_total_time
    content["timing_info"] = timings
    return ORJSONResponse(content=content)

@app.post("/get_frame_at_timestamp")
async def get_frame_at_timestamp(video_id: str = Form(...), timestamp: float = Form(...)):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")
    
    video_path_str = os.path.join(VIDEO_BASE_DIR, video_id)
    if not os.path.isfile(video_path_str) and not video_id.lower().endswith('.mp4'):
        video_path_str += '.mp4'

    if not os.path.isfile(video_path_str):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    try:
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")
        
        # Convert timestamp (seconds) to milliseconds for OpenCV
        time_ms = timestamp * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=404, detail=f"Could not read frame at timestamp {timestamp}s.")
            
        # Convert the frame to a JPEG image in memory
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode frame.")
        
        # Encode the image bytes to a base64 string
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        return {"image_data": f"data:image/jpeg;base64,{img_base64}"}

    except Exception as e:
        # It's good practice to release the capture object in case of an error
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print(f"Error getting frame at timestamp: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video frame: {e}")

@app.post("/check_temporal_frames")
async def check_temporal_frames(request_data: CheckFramesRequest) -> List[str]:
    """
    Phiên bản siêu tối ưu: Sử dụng cache đã tính toán trước từ file JSON.
    1. Lấy tên file từ đường dẫn đầy đủ.
    2. Dùng tên file làm key để tra cứu trong cache.
    3. Ghép lại đường dẫn đầy đủ cho các file xung quanh và trả về.
    """
    base_filepath = request_data.base_filepath
    
    # Nếu cache không được tải vì lý do nào đó, trả về chính file đó để không bị lỗi
    print(base_filepath)
    if not FRAME_CONTEXT_CACHE:
        return [base_filepath]
    
    # 1. Lấy tên file (key) từ đường dẫn đầy đủ mà frontend gửi lên
    # Ví dụ: từ "/path/to/L26_V462_0001_000000.webp" -> "L26_V462_0001_000000.webp"
    key_filename = os.path.basename(base_filepath)
    
    # 2. Tra cứu danh sách các tên file xung quanh từ cache
    # Nếu không tìm thấy, trả về một danh sách chỉ chứa file gốc
    neighbor_filenames = FRAME_CONTEXT_CACHE.get(key_filename)
    
    if not neighbor_filenames:
        return [base_filepath]

    # 3. Ghép lại đường dẫn đầy đủ và trả về cho frontend
    # Frontend cần đường dẫn đầy đủ để có thể hiển thị ảnh
    # Ví dụ: từ "L26_V462_0001_000016.webp" -> "/.../Keyframes/webp_keyframes/L26_V462_0001_000016.webp"
    full_neighbor_paths = [os.path.join(IMAGE_BASE_PATH, fname) for fname in neighbor_filenames]
    
    return full_neighbor_paths


@app.get("/videos/{video_id}")
async def get_video(video_id: str, range: str = Header(None)):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")
    
    video_path = os.path.join(VIDEO_BASE_DIR, video_id)
    if not os.path.isfile(video_path) and not video_id.endswith('.mp4'):
        video_path += '.mp4'

    if not os.path.isfile(video_path):
         raise HTTPException(status_code=404, detail=f"Video not found at path: {video_path}")

    video_size = os.path.getsize(video_path)
    
    start, end = 0, video_size - 1
    status_code = 200
    headers = {
        'Content-Length': str(video_size),
        'Accept-Ranges': 'bytes',
        'Content-Type': 'video/mp4'
    }

    if range is not None:
        range_match = re.match(r'bytes=(\d+)-(\d*)', range)
        if range_match:
            start_str, end_str = range_match.groups()
            start = int(start_str)
            if end_str:
                end = int(end_str)
            else:
                end = start + 1024 * 1024 * 2 # Gửi 2MB mỗi lần
            end = min(end, video_size - 1)
            
            if start >= video_size or end < start:
                 raise HTTPException(status_code=416, detail="Requested range not satisfiable")

            status_code = 206 # Partial Content
            headers['Content-Length'] = str(end - start + 1)
            headers['Content-Range'] = f'bytes {start}-{end}/{video_size}'
    
    async def video_iterator(start_pos, end_pos):
        with open(video_path, "rb") as video_file:
            video_file.seek(start_pos)
            bytes_to_read = end_pos - start_pos + 1
            chunk_size = 8192
            while bytes_to_read > 0:
                chunk = video_file.read(min(chunk_size, bytes_to_read))
                if not chunk:
                    break
                bytes_to_read -= len(chunk)
                yield chunk

    return StreamingResponse(video_iterator(start, end), status_code=status_code, headers=headers, media_type="video/mp4")

@app.get("/images/{encoded_path}")
async def get_image(encoded_path: str):
    try:
        original_path = base64.urlsafe_b64decode(encoded_path).decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 path.")
    
    safe_base = os.path.realpath(ALLOWED_BASE_DIR)
    safe_path = os.path.realpath(original_path)
    
    if not safe_path.startswith(safe_base) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    return FileResponse(safe_path)

@app.get("/video_info/{video_id}", response_model=VideoInfoResponse)
async def get_video_info(video_id: str):
    if "/" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")
    
    video_path = os.path.join(VIDEO_BASE_DIR, video_id)
    if not os.path.isfile(video_path):
        if not video_id.lower().endswith('.mp4'):
            video_path = os.path.join(VIDEO_BASE_DIR, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
             raise HTTPException(status_code=404, detail=f"Video info not found for: {video_id}")

    fps = await asyncio.to_thread(get_video_fps, video_path)
    return VideoInfoResponse(fps=fps)