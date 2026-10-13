import gc
from llama_cpp import Llama
from nyx_sieve import anonymize

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen3-4B-GGUF",
#     filename="Qwen3-4B-Q4_K_M.gguf",
#     n_ctx=4096,
# )

llm = Llama(
    model_path="models/Qwen_Qwen3-4B-Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False,
)

try:
    with open("sample_ehr.txt", "r", encoding="utf-8") as file:
            raw_note = file.read()

    sanitized_note = anonymize(raw_note)


    output = llm(
        "Analyze this sanitized payload : "+sanitized_note,
        max_tokens=1500,
        temperature=0.0,
    )

    print(output["choices"][0]["text"])

finally:
    # Best-effort clearing of the inference KV cache.
    # This is not a guarantee of complete memory zeroization.
    if llm is not None:
        try:
            llm._ctx.kv_cache_clear()
        except (AttributeError, TypeError) as exc:
            print(f"KV-cache clearing unavailable: {exc}")

    # Drop Python references to sensitive content.
    raw_note = None
    sanitized_note = None
    output = None

    # Release the model and its associated native resources
    # when the object is no longer referenced.
    del llm
    gc.collect()