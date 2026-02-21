# Fix pydub/audioop incompatibility with Python 3.13
import sys
import types
audioop_mock = types.ModuleType("audioop")
sys.modules["audioop"] = audioop_mock

# Fix HfFolder removed in huggingface_hub>=0.25
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
    _hf_hub.HfFolder = _HfFolder

import gradio as gr

# Patch gradio_client bug: 'APIInfoParseError: Cannot parse schema True'
# The crash is in _json_schema_to_python_type when schema is a bool (e.g. additionalProperties=True)
try:
    import gradio_client.utils as _gc_utils
    _orig_json_schema = _gc_utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "any"
        return _orig_json_schema(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_WEIGHTS = "Caline0/finance-llm-lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ── Load model (CPU-compatible, no quantization) ──────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
model.eval()
print("Model ready!")

# ── Chat function ─────────────────────────────────────────────────────
def finance_chat_fn(user_message, chat_history):
    if not user_message.strip():
        return "", chat_history

    context = ""
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n\n"

    prompt = f"{context}### Instruction:\n{user_message.strip()}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_output:
        bot_response = full_output.split("### Response:")[-1].strip()
    else:
        bot_response = full_output.strip()

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_response[:400]})
    return "", chat_history

# ── Gradio UI ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What is portfolio diversification?",
    "How do bonds work?",
    "Explain compound interest.",
    "What is asset allocation?",
    "Define volatility in investing.",
    "What is the P/E ratio?",
    "What is dollar-cost averaging?",
]

with gr.Blocks(theme=gr.themes.Soft(), title="Finance AI Assistant") as demo:
    gr.Markdown("""
    # 💰 Finance AI Assistant
    **Fine-tuned LLM for financial questions.** Ask about investing, portfolio management, financial instruments, risk, and more.
    > Model: TinyLlama-1.1B + LoRA | Domain: Finance
    """)

    chatbot = gr.Chatbot(label="Conversation", height=440, bubble_full_width=False, type="messages")

    with gr.Row():
        message_box = gr.Textbox(
            placeholder="E.g., What is a P/E ratio?",
            label="Your Question", lines=2, scale=5
        )
        send_button = gr.Button("Send ➤", variant="primary", scale=1)

    gr.ClearButton([message_box, chatbot], value="Clear Chat")

    gr.Examples(examples=EXAMPLES, inputs=message_box, label="Example Questions:")

    gr.Markdown("---\n*Responses are AI-generated for educational purposes. Not financial advice.*")

    send_button.click(finance_chat_fn, [message_box, chatbot], [message_box, chatbot])
    message_box.submit(finance_chat_fn, [message_box, chatbot], [message_box, chatbot])

demo.launch(show_api=False)
