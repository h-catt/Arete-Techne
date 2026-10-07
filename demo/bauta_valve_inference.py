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

with open("sample_ehr.txt", "r", encoding="utf-8") as file:
        raw_note = file.read()

sanitized_note = anonymize(raw_note)


output = llm(
    "Analyze this sanitized payload : "+sanitized_note,
    max_tokens=1500,
    temperature=0.0,
)

print(output["choices"][0]["text"])

# Cleanup / VRAM release
del llm
gc.collect()