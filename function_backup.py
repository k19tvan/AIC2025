import time
from openai import OpenAI

# Point the client to your local server
client = OpenAI(
    base_url="http://nguyenmv_aicity2025-nguyen_dfine_s:6262/v1",
    api_key="not-needed"  # API key is not required for local servers
)

def enhance_query(original_query: str) -> tuple[str, float]:
    """
    Enhances a short, incomplete, or Vietnamese image search query into a
    clean, natural English caption using a locally hosted Qwen model.

    Args:
        original_query: The user's original image search query.

    Returns:
        A tuple containing the enhanced English caption and the latency in seconds.
    """
    try:
        system_prompt = """<PROMPT CUA ANH EM>"""

        user_prompt = f"Original Query: {original_query}"

        start_time = time.perf_counter()

        completion = client.chat.completions.create(
            model="./Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=50,    # Reduced to prevent over-elaboration
        )

        end_time = time.perf_counter()
        latency = end_time - start_time

        caption = completion.choices[0].message.content.strip()
        # Remove any quotes that might be added
        caption = caption.strip('"\'')
        return caption, latency

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return error_message, 0.0


# --- Continuous Input Loop ---
print("--- Enter your image search queries (press Ctrl+C to exit) ---")

try:
    while True:
        query = input("Enter query: ").strip()
        if not query:
            print("Please enter a non-empty query.")
            continue

        enhanced_caption, query_latency = enhance_query(query)
        print(f"Enhanced:   '{enhanced_caption}'")
        print(f"Latency:    {query_latency:.4f} seconds")
        print("-" * 40)

except KeyboardInterrupt:
    print("\nExiting... Goodbye!")