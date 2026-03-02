import os
from dotenv import load_dotenv

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the main.env file relative to the current file
dotenv_path = os.path.join(current_dir, "../../configs/main.env")
load_dotenv(dotenv_path=dotenv_path)

# --- Runtime Configuration ---
RUNTIME = os.getenv("RUNTIME", "ollama")

# --- Agent Architecture ---
AGENT_ARCH = os.getenv("AGENT_ARCH", "world_model")

# --- VLM Model Configuration (for Perception) ---
PERCEPTION_MODEL_ID = os.getenv("PERCEPTION_MODEL_ID", "mlx-community/llava-v1.6-mistral-7b-4bit")

# --- LLM Model Configuration (for Controller) ---
CONTROLLER_MODEL_ID = os.getenv("CONTROLLER_MODEL_ID", "mlx-community/Qwen3-1.7B-4bit")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_PERCEPTION_MODEL = os.getenv("OLLAMA_PERCEPTION_MODEL", "llava:latest")
OLLAMA_CONTROLLER_MODEL = os.getenv("OLLAMA_CONTROLLER_MODEL", "qwen3:1.7b")
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 300)) # Default to 5 minutes
OLLAMA_CONTEXT_SIZE = int(os.getenv("OLLAMA_CONTEXT_SIZE", 8192)) # Default to 8k tokens



# --- Environment Configuration ---
GYM_ENVIRONMENT = os.getenv("GYM_ENVIRONMENT", "MiniGrid-Empty-5x5-v0")
RENDER_MODE = os.getenv("RENDER_MODE", "rgb_array")

# --- Agent Run Configuration ---
MAX_STEPS = int(os.getenv("MAX_STEPS", 10))
INSTRUCTION = os.getenv("INSTRUCTION", "Explore the room.")
MEMORY_LENGTH = int(os.getenv("MEMORY_LENGTH", 5)) # Number of past steps to remember

# --- Recording Configuration ---
RECORDING_DIR = os.getenv("RECORDING_DIR", "outputs/recordings")
