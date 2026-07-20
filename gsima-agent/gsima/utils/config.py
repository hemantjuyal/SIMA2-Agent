import os
from dotenv import load_dotenv

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the main.env file relative to the current file
dotenv_path = os.path.join(current_dir, "../../configs/main.env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# Infer environment type and load environment-specific overrides
gym_environment = os.getenv("GYM_ENVIRONMENT", "VizdoomBasic-v1")
env_type = os.getenv("ENV_TYPE", "")
if not env_type:
    normalized = gym_environment.strip().lower()
    if normalized == "minigrid-empty-8x8-v0":
        env_type = "minigrid_empty"
    elif normalized == "babyai-unlockpickup-v0" or normalized == "minigrid-unlockpickup-v0":
        env_type = "babyai_unlock_pickup"
    elif normalized in ("vizdoombasic-v1", "vizdoombasic-multibinary-v1"):
        env_type = "vizdoom_basic"
    elif normalized in ("vizdoompredictposition-v1", "vizdoompredictposition-multibinary-v1"):
        env_type = "vizdoom_predict"
    else:
        env_type = gym_environment.split("-")[0].lower()

if env_type:
    env_dotenv_path = os.path.join(current_dir, f"../../configs/{env_type}.env")
    if os.path.exists(env_dotenv_path):
        load_dotenv(dotenv_path=env_dotenv_path, override=True)

# --- Runtime Configuration ---
RUNTIME = os.getenv("RUNTIME", "gemini")

# --- Gemini Runtime Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- Evaluation Selection ---
EVAL_EPISODES = int(os.getenv("EVAL_EPISODES", 1))

# --- Environment Selection ---
GYM_ENVIRONMENT = os.getenv("GYM_ENVIRONMENT", "VizdoomBasic-v1")
ENV_TYPE = os.getenv("ENV_TYPE", "")
RENDER_MODE = os.getenv("RENDER_MODE", "record")
FRAME_SKIP = os.getenv("FRAME_SKIP")
if FRAME_SKIP is not None:
    try:
        FRAME_SKIP = int(FRAME_SKIP)
    except ValueError:
        FRAME_SKIP = None

# --- Agent Run Configuration ---
MAX_STEPS = int(os.getenv("MAX_STEPS", 10))
INSTRUCTION = os.getenv("INSTRUCTION", "Explore the room.")
MEMORY_LENGTH = int(os.getenv("MEMORY_LENGTH", 5))

# --- UI Configuration ---
UI_GAME_NAME = os.getenv("UI_GAME_NAME", "SIMA 2 Demo")

# --- Recording Configuration ---
RECORDING_DIR = os.getenv("RECORDING_DIR", "outputs/recordings")
RECORDING_FPS = os.getenv("RECORDING_FPS")
if RECORDING_FPS is not None:
    try:
        RECORDING_FPS = int(RECORDING_FPS)
    except ValueError:
        RECORDING_FPS = None
