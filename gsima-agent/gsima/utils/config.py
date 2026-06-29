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

# --- Model Configuration (runtime-specific) ---
PERCEPTION_MODEL_ID = os.getenv("PERCEPTION_MODEL_ID", "mlx-community/llava-v1.6-mistral-7b-4bit")
CONTROLLER_MODEL_ID = os.getenv("CONTROLLER_MODEL_ID", "mlx-community/Qwen3-1.7B-4bit")

# --- Ollama Runtime Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_PERCEPTION_MODEL = os.getenv("OLLAMA_PERCEPTION_MODEL", "llava:latest")
OLLAMA_CONTROLLER_MODEL = os.getenv("OLLAMA_CONTROLLER_MODEL", "qwen3:1.7b")
OLLAMA_CONTROLLER_FALLBACK_MODEL = os.getenv("OLLAMA_CONTROLLER_FALLBACK_MODEL", "qwen3:0.6b")
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 300))
OLLAMA_CONTEXT_SIZE = int(os.getenv("OLLAMA_CONTEXT_SIZE", 8192))
OLLAMA_PERCEPTION_CONTEXT_SIZE = int(os.getenv("OLLAMA_PERCEPTION_CONTEXT_SIZE", OLLAMA_CONTEXT_SIZE))
OLLAMA_CONTROLLER_CONTEXT_SIZE = int(os.getenv("OLLAMA_CONTROLLER_CONTEXT_SIZE", OLLAMA_CONTEXT_SIZE))

# --- Environment Selection ---
GYM_ENVIRONMENT = os.getenv("GYM_ENVIRONMENT", "MiniGrid-Empty-5x5-v0")
ENV_TYPE = os.getenv("ENV_TYPE", "")
RENDER_MODE = os.getenv("RENDER_MODE", "rgb_array")

# --- Agent Run Configuration ---
MAX_STEPS = int(os.getenv("MAX_STEPS", 10))
INSTRUCTION = os.getenv("INSTRUCTION", "Explore the room.")
MEMORY_LENGTH = int(os.getenv("MEMORY_LENGTH", 5))

# --- Agent Planning Configuration ---
ROLLOUT_DEPTH = int(os.getenv("ROLLOUT_DEPTH", 3))
TURN_PENALTY = float(os.getenv("TURN_PENALTY", 0.1))
COLLISION_PENALTY = float(os.getenv("COLLISION_PENALTY", 1.0))
GOAL_BONUS = float(os.getenv("GOAL_BONUS", 10.0))
STOP_DISTANCE_THRESHOLD = int(os.getenv("STOP_DISTANCE_THRESHOLD", 1))

# --- Recording Configuration ---
RECORDING_DIR = os.getenv("RECORDING_DIR", "outputs/recordings")
