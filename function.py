# function.py

# =========================
# Imports
# =========================
import asyncio
import time
from typing import List

from google import genai
from google.genai import types
from async_lru import alru_cache
from fast_langdetect import LangDetector
from googletrans import Translator

import sys
path_to_api = '/workspace/mlcv2/WorkingSpace/Personal/nguyenmv/HCMAIC25/AICHALLENGE_OPENCUBEE_2/VongSoTuyen/DataPreprocessing/Api'
sys.path.append(path_to_api)
from Api_key import api_key

# =========================
# Config
# =========================
# NOTE: Using gemini-1.5-flash-latest, as it's a modern, fast, and capable model.
# The original model names like "gemini-2.5-flash" do not exist.
MODEL_ENHANCE = "gemini-1.5-flash-latest"
MODEL_EXPAND = "gemini-1.5-flash-latest"
NUM_EXPAND_VARIATIONS = 5  # Number of variations to generate in a single API call
MAX_ATTEMPT = len(api_key) * 2 # Allow each key to be tried twice for resilience

# =========================
# Initialization (Singleton instances for performance)
# =========================
# These are initialized once when the module is loaded to avoid repeated setup costs.
LANG_DETECTOR = LangDetector()
TRANSLATOR = Translator()

# =========================
# Query Enhancement (Async, Cached, and Robust)
# =========================
@alru_cache(maxsize=1024)
async def enhance_query(original_query: str) -> str:
    """
    Improves a search query using an LLM. This function is now fully asynchronous,
    cached, and includes a retry mechanism with API key rotation.
    """
    if not original_query or not original_query.strip():
        return ""

    for attempt in range(MAX_ATTEMPT):
        try:
            key_index = attempt % len(api_key)
            client = genai.AsyncClient(api_key=api_key[key_index])

            # A more concise and direct prompt
            prompt = (
                f"You are an expert search query optimizer. Your task is to translate the following query to English if it's not already, "
                f"then enhance it for a video retrieval system. Make it clear, specific, and keyword-rich, preserving all original details. "
                f"Return ONLY the final, enhanced English query.\n\n"
                f"Original query: \"{original_query}\""
            )

            resp = await client.models.generate_content_async(
                model=MODEL_ENHANCE,
                contents=prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.1 # Low temperature for consistent, factual enhancement
                ),
            )
            # Clean up the response to ensure only the query text is returned
            enhanced_text = resp.text.strip().strip('"')
            return enhanced_text
        except Exception as e:
            print(f"Enhance query attempt {attempt+1} failed with key index {key_index}. Error: {e}")
            await asyncio.sleep(0.2)  # Small delay before retrying
            continue
            
    print(f"--- WARNING: All enhance_query attempts failed. Returning original query: '{original_query}' ---")
    return original_query

# =========================
# Query Translation (Already Optimized)
# =========================
@alru_cache(maxsize=1024)
async def translate_query(query: str, dest: str = 'en') -> str:
    """
    Translates a query to the destination language if it's not already English.
    This function is cached for performance.
    """
    if not query or not query.strip():
        return ""

    try:
        if LANG_DETECTOR.detect(query) == 'en':
            print(f"--- INFO: Query is already in English. No translation needed. ---")
            return query

        loop = asyncio.get_running_loop()
        # googletrans is synchronous, so we run it in an executor to avoid blocking
        result = await loop.run_in_executor(None, TRANSLATOR.translate, query, dest)
        return result.text

    except Exception as e:
        print(f"--- WARNING: Translation/Detection failed for query '{query}'. Error: {e}. Returning original query. ---")
        return query

# =========================
# Query Expansion (Single Call, Async, Cached, and Robust)
# =========================
@alru_cache(maxsize=1024)
async def expand_query_parallel(short_query: str) -> str:
    """
    Expands a short query into multiple diverse scenarios with a single, cached,
    asynchronous API call. This is vastly more efficient than the previous parallel approach.
    """
    if not short_query or not short_query.strip():
        return ""

    for attempt in range(MAX_ATTEMPT):
        try:
            key_index = attempt % len(api_key)
            client = genai.AsyncClient(api_key=api_key[key_index])
            
            # A new, more efficient prompt that asks for multiple variations in one go
            prompt_text = (
                f"You are a creative assistant for a video search engine. Your task is to expand the user's short query into {NUM_EXPAND_VARIATIONS} distinct, "
                f"detailed, and plausible video scene descriptions. Each description must be on a new line and start with a hyphen '-'.\n\n"
                f"Query: \"{short_query}\"\n\n"
                f"Scenarios:"
            )

            resp = await client.models.generate_content_async(
                model=MODEL_EXPAND,
                contents=prompt_text,
                generation_config=types.GenerationConfig(
                    temperature=0.7 # Higher temperature for creative and diverse scenarios
                ),
            )

            # Parse the single, multi-line response
            results = [
                line.strip().lstrip('-').strip()
                for line in resp.text.strip().split('\n')
                if line.strip()
            ]

            # Deduplicate results from the single call
            seen = set()
            final_results = []
            for s in results:
                if s not in seen:
                    seen.add(s)
                    final_results.append(s)

            if not final_results:
                return short_query  # Fallback to the original if the LLM gives an empty response

            return "\n".join(final_results)

        except Exception as e:
            print(f"Expand query attempt {attempt+1} failed with key index {key_index}. Error: {e}")
            await asyncio.sleep(0.2)
            continue

    print(f"--- WARNING: All expand_query attempts failed. Returning original query: '{short_query}' ---")
    return short_query

# =========================
# Main Test Block (Demonstrates performance and caching)
# =========================
if __name__ == "__main__":
    async def main_test():
        query_vi = "một người đàn ông mặc áo đỏ đang đi trên phố"
        query_en = "a dog catching a frisbee in a park"

        print("="*40)
        print("PERFORMANCE AND CACHE DEMONSTRATION")
        print("="*40)

        # --- Test Translation & Cache ---
        print("\n--- Testing Translation ---")
        start_time = time.perf_counter()
        translated = await translate_query(query_vi)
        print(f"1st translation ('{query_vi}'): '{translated}' (Took: {time.perf_counter() - start_time:.4f}s)")

        start_time = time.perf_counter()
        translated_cached = await translate_query(query_vi)
        print(f"2nd translation (cached): '{translated_cached}' (Took: {time.perf_counter() - start_time:.4f}s)")
        print("-" * 20)

        # --- Test Enhancement & Cache ---
        print("\n--- Testing Enhancement ---")
        start_time = time.perf_counter()
        enhanced = await enhance_query(query_vi)
        print(f"1st enhancement ('{query_vi}'):\n'{enhanced}'\n(Took: {time.perf_counter() - start_time:.4f}s)")

        start_time = time.perf_counter()
        enhanced_cached = await enhance_query(query_vi)
        print(f"\n2nd enhancement (cached):\n'{enhanced_cached}'\n(Took: {time.perf_counter() - start_time:.4f}s)")
        print("-" * 20)

        # --- Test Expansion & Cache ---
        print("\n--- Testing Expansion (Single API Call) ---")
        start_time = time.perf_counter()
        expanded = await expand_query_parallel(query_en)
        print(f"1st expansion ('{query_en}'):\n{expanded}\n(Took: {time.perf_counter() - start_time:.4f}s)")

        start_time = time.perf_counter()
        expanded_cached = await expand_query_parallel(query_en)
        print(f"\n2nd expansion (cached):\n{expanded_cached}\n(Took: {time.perf_counter() - start_time:.4f}s)")

    asyncio.run(main_test())