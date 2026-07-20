import inspect
import os
import importlib
import logging
import datetime
import re
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, HumanRendering

from gsima.utils import config
from gsima.environments.base import BaseAdapter
from gsima.memory import create_memory


def _infer_env_type(env_name: str) -> str:
    """Infer the environment backend from a Gym environment name.

    Handles names that use a scenario suffix after the backend prefix, such as
    ``VizdoomBasic-v1`` and ``VizdoomBasic-MultiBinary-v1``.
    """
    normalized = env_name.strip().lower()

    if normalized == "minigrid-empty-8x8-v0":
        return "minigrid_empty"
    if normalized == "babyai-unlockpickup-v0" or normalized == "minigrid-unlockpickup-v0":
        return "babyai_unlock_pickup"
    if normalized in ("vizdoombasic-v1", "vizdoombasic-multibinary-v1"):
        return "vizdoom_basic"
    if normalized in ("vizdoompredictposition-v1", "vizdoompredictposition-multibinary-v1"):
        return "vizdoom_predict"

    # Fallback to the first hyphen-separated token for naming conventions.
    prefix = env_name.split("-")[0].lower()
    return prefix


def _to_adapter_class_name(env_type: str) -> str:
    """Convert a backend slug into a stable adapter class name.

    This keeps the factory resilient for future environments that use
    underscores, hyphens, or mixed-case backend identifiers.
    """
    parts = re.split(r"[^a-zA-Z0-9]+", env_type)
    return "".join(part.capitalize() for part in parts if part) + "Adapter"


class LingerRecordVideo(RecordVideo):
    """A custom RecordVideo wrapper that lingers on the final frame before closing."""
    def __init__(self, env, linger_frames=30, **kwargs):
        super().__init__(env, **kwargs)
        self.linger_frames = linger_frames

    def close_video_recorder(self):
        # Intercept the close signal to pump extra frames before it actually closes
        if getattr(self, "recording", False) and getattr(self, "video_recorder", None):
            logging.info(f"Lingering video for {self.linger_frames} extra frames before closing...")
            for _ in range(self.linger_frames):
                self.video_recorder.capture_frame()
        super().close_video_recorder()


def create_env_and_adapter() -> tuple:
    """
    Dynamically creates a Gym environment and all its associated, fully configured components.

    This factory is responsible for:
    1. Detecting the environment type from the config.
    2. Dynamically importing the correct adapter and prompt functions.
    3. Creating and configuring the appropriate memory system for the environment.
    4. Applying all necessary wrappers to the environment.

    Returns:
        A tuple containing all components needed to run the agent:
        - The Gym environment instance.
        - The environment-specific adapter.
        - The fully configured memory system.
        - The `get_prompt` function.
        - The `get_visual_prompt` function.
        - The `get_outcome_from_reward` function.
    """
    env_name = config.GYM_ENVIRONMENT
    logging.info(f"Attempting to create environment: {env_name}")

    # --- Dynamic Environment Type Detection ---
    env_type = config.ENV_TYPE.strip().lower() if config.ENV_TYPE else ""
    if not env_type:
        try:
            env_type = _infer_env_type(env_name)
        except (AttributeError, IndexError):
            raise ValueError(f"Invalid environment name format: {env_name}")

    if not env_type:
        raise ValueError("Environment type could not be determined from ENV_TYPE or GYM_ENVIRONMENT.")

    adapter_class_name = _to_adapter_class_name(env_type)
    logging.info(f"Detected environment type: '{env_type}'")

    # --- Dynamic Module and Component Loading ---
    try:
        env_package = importlib.import_module(f"gsima.environments.{env_type}")
        if hasattr(env_package, "ensure_registration"):
            env_package.ensure_registration()

        adapter_module = importlib.import_module(f"gsima.environments.{env_type}.adapter")
        adapter_class = getattr(adapter_module, adapter_class_name, None)
        if adapter_class is None:
            adapter_candidates = [
                getattr(adapter_module, name)
                for name, obj in inspect.getmembers(adapter_module, inspect.isclass)
                if issubclass(obj, BaseAdapter) and obj is not BaseAdapter
            ]
            if adapter_candidates:
                adapter_class = adapter_candidates[0]
                logging.warning(
                    f"Could not find adapter class '{adapter_class_name}'. "
                    f"Falling back to concrete adapter '{adapter_class.__name__}'."
                )
            else:
                raise AttributeError(
                    f"Adapter class '{adapter_class_name}' not found in module {adapter_module.__name__}."
                )

        prompt_module = importlib.import_module(f"gsima.environments.{env_type}.prompt")
        get_multimodal_prompt_func = getattr(prompt_module, "get_multimodal_prompt")
        create_memory_summary_func = getattr(prompt_module, "create_memory_summary")

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not find or load components for env type '{env_type}': {e}")

    # --- Memory System Creation & Configuration ---
    memory_system = create_memory()
    memory_system.set_summarizer(create_memory_summary_func)
    logging.info(f"Created and configured memory system: {type(memory_system).__name__}")

    # --- Gym Environment Creation ---
    # The base environment must be 'rgb_array' for VLM perception and wrappers to work.
    gym_make_render_mode = "rgb_array" 
    logging.info(f"Creating Gym environment '{env_name}' with base render_mode='{gym_make_render_mode}'")
    
    gym_make_kwargs = {"render_mode": gym_make_render_mode}
    if env_type == "vizdoom" and getattr(config, "FRAME_SKIP", None) is not None:
        gym_make_kwargs["frame_skip"] = config.FRAME_SKIP
        logging.info(f"Using frame_skip={config.FRAME_SKIP} for environment.")

    try:
        env = gym.make(env_name, **gym_make_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create Gym environment: {e}")
    
    adapter = adapter_class(env)

    # --- Environment-Specific Wrappers ---
    if hasattr(env_package, "apply_env_wrappers"):
        env = env_package.apply_env_wrappers(env)

    # --- User-Facing Rendering Wrappers ---
    if config.RENDER_MODE == "human":
        env = HumanRendering(env)
    elif config.RENDER_MODE == "record":
        # Create a unique sub-folder for this run using a timestamp
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_video_dir = os.path.join(config.RECORDING_DIR, f"{env_name}_{run_timestamp}")
        os.makedirs(run_video_dir, exist_ok=True)
        
        # Override the metadata render_fps to slow down the recording if configured
        if getattr(config, "RECORDING_FPS", None) is not None:
            env.metadata["render_fps"] = config.RECORDING_FPS
            logging.info(f"Overriding env.metadata['render_fps'] to {config.RECORDING_FPS} for slower recording playback.")
        
        # Use our custom LingerRecordVideo to pause on the final frame
        env = LingerRecordVideo(env, video_folder=run_video_dir, name_prefix="episode", linger_frames=30)
        logging.info(f"Video recordings will be saved to '{run_video_dir}'.")

    logging.info(f"Environment '{env_name}' and components created successfully.")
    return (
        env,
        adapter,
        memory_system,
        get_multimodal_prompt_func,
    )

