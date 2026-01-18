import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(dotenv_path="configs/main.env")

# --- Runtime Configuration ---
RUNTIME = os.getenv("RUNTIME", "mlx")

# --- Agent Model Type ---
AGENT_MODEL_TYPE = os.getenv("AGENT_MODEL_TYPE", "vlm")

# --- LLM Model Configuration ---
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "mlx-community/Qwen3-8B-Instruct-4bit")

# --- VLM Model Configuration ---
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "mlx-community/llava-v1.6-mistral-7b-4bit")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen3-vl:4b")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b")


# --- Environment Configuration ---
GYM_ENVIRONMENT = os.getenv("GYM_ENVIRONMENT", "MiniGrid-Empty-5x5-v0")
RENDER_MODE = os.getenv("RENDER_MODE", "rgb_array")

# --- Agent Run Configuration ---
MAX_STEPS = int(os.getenv("MAX_STEPS", 100))
INSTRUCTION = os.getenv("INSTRUCTION", "Explore the room.")
MEMORY_LENGTH = int(os.getenv("MEMORY_LENGTH", 5)) # Number of past steps to remember

# --- Recording Configuration ---
RECORDING_DIR = os.getenv("RECORDING_DIR", "outputs/recordings")
