import gc
import subprocess
import psutil
from llama_cpp import Llama
from nyx_sieve import anonymize

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen3-4B-GGUF",
#     filename="Qwen3-4B-Q4_K_M.gguf",
#     n_ctx=4096,
# )

def get_gpu_memory():
    memory = psutil.virtual_memory()

    return {
        "used": memory.used / (1024 ** 2),
        "free": memory.available / (1024 ** 2),
    }


# def get_gpu_memory():
#     """Return total GPU memory used and free, in MiB."""
#     try:
#         result = subprocess.run(
#             [
#                 "nvidia-smi",
#                 "--query-gpu=memory.used,memory.free",
#                 "--format=csv,noheader,nounits",
#             ],
#             capture_output=True,
#             text=True,
#             check=True,
#             timeout=10,
#         )

#         used, free = map(int, result.stdout.strip().splitlines()[0].split(","))
#         return {"used": used, "free": free}

    # except Exception as exc:
    #     print(f"Unable to measure GPU memory: {exc}")
    #     return None


def report(label):
    memory = get_gpu_memory()

    if memory is None:
        return None

    print(f"\n{label}")
    print(f"VRAM used : {memory['used']} MiB")
    print(f"VRAM free : {memory['free']} MiB")

    return memory

before = report("BEFORE BAUTA VALVE")
llm = Llama(
    model_path="models/Qwen_Qwen3-4B-Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False,
)
after_load = report("AFTER MODEL LOAD")

# try:
with open("sample_ehr.txt", "r", encoding="utf-8") as file:
        raw_note = file.read()

sanitized_note = anonymize(raw_note)


output = llm(
    "Analyze this sanitized payload : "+sanitized_note,
    max_tokens=1500,
    temperature=0.0,
)

print(output["choices"][0]["text"])
after_inference = report("AFTER INFERENCE")

# finally:
#     # Best-effort clearing of the inference KV cache.
#     # This is not a guarantee of complete memory zeroization.
#     if llm is not None:
#         try:
#             llm._ctx.kv_cache_clear()
#         except (AttributeError, TypeError) as exc:
#             print(f"KV-cache clearing unavailable: {exc}")

#     # Drop Python references to sensitive content.
#     raw_note = None
#     sanitized_note = None
#     output = None

#     # Release the model and its associated native resources
#     # when the object is no longer referenced.
#     del llm
#     gc.collect()


# 1. Clear the KV cache while the context still exists.
if llm is not None:
    try:
        llm._ctx.kv_cache_clear()
        print("KV cache clear requested.")
    except Exception as exc:
        print(f"KV cache clear unavailable: {exc}")

    # 2. Release the model/context using its public API if present.
    try:
        close_method = getattr(llm, "close", None)
        if callable(close_method):
            close_method()
    except Exception as exc:
        print(f"Model close warning: {exc}")

# 3. Drop Python references to inference data and the model.
response = None
sanitized_note = None
raw_note = None
llm = None

gc.collect()

# 4. Give the driver a moment before taking the measurement.
# time.sleep(1)
after_cleanup = report("AFTER BAUTA VALVE CLEANUP")

print(
    "Cleanup completed. "
)

# 6. Calculate observed memory changes
if all(x is not None for x in (
    before, after_load, after_inference, after_cleanup
)):
    print("\n========== VRAM SUMMARY ==========")

    print(
        "Model-load VRAM increase:",
        f"{after_load['used'] - before['used']} MiB"
    )

    print(
        "Inference VRAM increase:",
        f"{after_inference['used'] - after_load['used']} MiB"
    )

    print(
        "VRAM usage reduction after cleanup:",
        f"{after_inference['used'] - after_cleanup['used']} MiB"
    )

    print(
        "Net VRAM reduction vs baseline:",
        f"{before['used'] - after_cleanup['used']} MiB"
    )

    print(
        "Free VRAM increase after cleanup:",
        f"{after_cleanup['free'] - after_inference['free']} MiB"
    )
