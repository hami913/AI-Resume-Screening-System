import subprocess

MODEL = "models/gguf/llama-3.1-8b-instruct-q4_k_m.gguf"
LORA = "models/gguf/llama-3.1-8b-career-assistant-lora-f16.gguf"
LLAMA = "/Users/abrar/llama.cpp/build/bin/llama-cli"

SYSTEM_PROMPT = """You are an AI Career Assistant.
Give practical, specific and structured career advice.
Help users with career planning, skill gaps, resumes, projects,
job preparation, interviews, AI/ML careers, and learning roadmaps.
Always answer the complete question.
Use headings, bullet points, and actionable steps.
"""

print("\n========================================")
print("       AI CAREER ASSISTANT")
print("========================================")
print("Type /exit to quit.\n")

while True:
    user_prompt = input("You: ")

    if user_prompt.strip().lower() == "/exit":
        print("\nGoodbye! 👋")
        break

    if not user_prompt.strip():
        continue

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    command = [
        LLAMA,
        "-m", MODEL,
        "--lora", LORA,
        "-ngl", "99",
        "-c", "2048",
        "-n", "512",
        "--temp", "0.7",
        "--top-p", "0.9",
        "-p", prompt,
    ]

    print("\nAI Career Assistant:\n")

    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nGeneration stopped.\n")

    print("\n" + "-" * 50 + "\n")
