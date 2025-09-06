# utils_query.py

# =========================
# Imports
# =========================
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from typing import List

import asyncio
import time # Thêm để test tốc độ

# --- Tối ưu thư viện dịch và cache ---
from fast_langdetect import LangDetector
from googletrans import Translator
from async_lru import alru_cache # Import thư viện cache cho hàm async

import sys
path_to_api = '/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/DataPreprocessing/Api'
sys.path.append(path_to_api)
from Api_key import api_key

# =========================
# Config
# =========================
MODEL_ENHANCE = "gemini-2.5-flash"
MODEL_EXPAND = "gemini-2.5-flash"
NUM_EXPAND_WORKERS = 10
MAX_ATTEMPT = len(api_key)

# =========================
# Initialization (Khởi tạo một lần)
# =========================
# Khởi tạo detector một lần duy nhất để tái sử dụng, tránh load model nhiều lần
# Chọn "low_mem" để cân bằng giữa tốc độ và bộ nhớ
LANG_DETECTOR = LangDetector() 

# Khởi tạo translator một lần để tái sử dụng connection pool (nếu có)
TRANSLATOR = Translator()

# =========================
# API Key Rotation
# =========================
def get_client_for_thread(thread_id: int):
    """
    Lấy client theo thread_id để phân phối API key đều.
    """
    key_idx = thread_id % len(api_key)
    return genai.Client(api_key=api_key[key_idx])

# =========================
# Query Enhancement (Hàm đồng bộ)
# =========================
def enhance_query(original_query: str) -> str:
    """
    Improve a search query for better retrieval accuracy.
    This function takes a query in any language, translates it to English if necessary,
    and then enhances it for optimal search performance.
    """
    # ... (giữ nguyên code của bạn)
    for attempt in range(MAX_ATTEMPT):
        try:
            client = genai.Client(api_key=api_key[attempt % len(api_key)])
            prompt = (
                f"You are an expert in search query optimization for accurate and relevant retrieval.\n"
                f"Here is the original search query:\n"
                f"\"{original_query}\"\n\n"
                "Your task involves a two-step process:\n"
                "1. **Language Check & Translation:** First, analyze the provided query. If it is not in English, your primary step is to translate it into clear and accurate English. If it is already in English, proceed directly to the next step.\n"
                "2. **Enhancement:** Take the resulting English query and rewrite it to maximize the chances of retrieving highly relevant results. Make the wording clear, precise, and rich in meaningful keywords. Preserve the original intent and all essential details, but remove any ambiguity or unnecessary words.\n\n"
                "Return ONLY the final, enhanced English query. Do not include the original query, translations, or any explanations in your response."
            )

            resp = client.models.generate_content(
                model=MODEL_ENHANCE,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0
                ),
            )
            enhanced_text = resp.text.strip().strip('"')
            return enhanced_text
        except Exception:
            print("Fail, trying next key")
            continue
    return original_query
    
# =========================
# Query Translate Logic (Hàm bất đồng bộ - ĐÃ TỐI ƯU)
# =========================
# @alru_cache(maxsize=1024) tự động cache kết quả của hàm.
# Nếu gọi lại `translate_query` với cùng `query`, kết quả sẽ được trả về ngay lập tức từ cache.
@alru_cache(maxsize=1024)
async def translate_query(query: str, dest: str = 'en') -> str:
    """
    Dịch truy vấn sang 'dest' nếu không phải tiếng Anh.
    Nếu đã là tiếng Anh thì trả nguyên văn.
    Hàm này được tối ưu bằng caching.
    """
    if not query or not query.strip():
        return ""

    try:
        # 1. Tối ưu: Sử dụng instance LANG_DETECTOR đã được khởi tạo sẵn
        if LANG_DETECTOR.detect(query)['lang'] == 'en':
            print("cc")
            # print(f"'{query}' is already English. Skipping translation.")
            return query
        
        # 2. Tối ưu: Chỉ dịch khi cần thiết. Lệnh gọi API vẫn là điểm chậm nhất
        # nhưng cache sẽ giúp cho các lần gọi lặp lại.
        # print(f"Translating '{query}' to English...")
        result = await TRANSLATOR.translate(query, dest=dest)
        return result.text
    
    except Exception as e:
        print(f"--- WARNING: Translation/Detection failed for query '{query}'. Error: {e}. Returning original query. ---")
        return query

# =========================
# Query Expansion (Hàm đồng bộ)
# =========================
def _expand_once(short_query: str, thread_id: int) -> List[str]:
    # ... (giữ nguyên code của bạn)
    client = get_client_for_thread(thread_id)
    prompt_text = f"""Expand the short user query into several distinct, detailed video scene descriptions. 
Each description should represent a plausible, specific scenario. 
Start each scenario on a new line with a hyphen (-).

Query: "{short_query}"
Scenarios:"""

    resp = client.models.generate_content(
        model=MODEL_EXPAND,  
        contents=prompt_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return [
        line.strip().lstrip('-').strip()
        for line in resp.text.strip().split('\n')
        if line.strip()
    ]


def expand_query_parallel(short_query: str, num_requests: int = NUM_EXPAND_WORKERS) -> str:
    # ... (giữ nguyên code của bạn)
    results = []
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(_expand_once, short_query, i)
            for i in range(num_requests)
        ]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass
    seen = set()
    final_results = []
    for s in results:
        if s not in seen:
            seen.add(s)
            final_results.append(s)
    return "\n".join(final_results)

# =========================
# Main test (Cập nhật để thấy rõ hiệu quả cache)
# =========================
async def main():
    result = await translate_query("a men is holding something")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())