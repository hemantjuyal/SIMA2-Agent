import os
import importlib
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, HumanRendering
from gsima.utils import config
from gsima.environments.base import BaseAdapter
import minigrid # Explicitly import minigrid to register its environments
# from minigrid.wrappers import PositionBonus # Removed PositionBonus wrapper
from gsima.environments.minigrid.custom_reward_wrapper import CustomMiniGridRewardWrapper # Import our custom reward wrapper

def create_env_and_adapter() -> tuple[gym.Env, BaseAdapter, callable, callable, callable, callable]:
    """
    Dynamically creates a Gym environment and its associated components.

    This factory function reads the environment name from the configuration,
    determines the environment type (e.g., 'minigrid'), and then dynamically
    imports the corresponding adapter and prompt-related functions.

    Returns:
        A tuple containing:
        - The Gym environment instance.
        - The environment-specific adapter.
        - The `get_prompt` function.
        - The `get_visual_prompt` function.
        - The `get_outcome_from_reward` function.
        - The `create_memory_summary` function.
    """
    env_name = config.GYM_ENVIRONMENT
    logging.info(f"Attempting to create environment: {env_name}")

    # --- Dynamic Environment Type Detection ---
    try:
        # e.g., "MiniGrid-Empty-5x5-v0" -> "minigrid"
        env_type = env_name.split('-')[0].lower()
        # Handle specific capitalization for MiniGrid
        if env_type == "minigrid":
            adapter_class_name = "MiniGridAdapter"
        else:
            # Capitalize for other class names, e.g., "myenv" -> "MyenvAdapter"
            adapter_class_name = f"{env_type.capitalize()}Adapter"
        logging.info(f"Detected environment type: '{env_type}'")
    except IndexError:
        logging.error(f"Could not determine environment type from name: '{env_name}'. Expected format like 'Type-Name-v0'.")
        raise ValueError(f"Invalid environment name format: {env_name}")

    # --- Dynamic Module and Component Loading ---
    try:
        adapter_module = importlib.import_module(f"gsima.environments.{env_type}.adapter")
        adapter_class = getattr(adapter_module, adapter_class_name)

        prompt_module = importlib.import_module(f"gsima.environments.{env_type}.prompt")
        get_prompt_func = getattr(prompt_module, "get_prompt")
        get_visual_prompt_func = getattr(prompt_module, "get_visual_prompt")
        get_outcome_from_reward_func = getattr(prompt_module, "get_outcome_from_reward")
        create_memory_summary_func = getattr(prompt_module, "create_memory_summary")

    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or load components for env type '{env_type}': {e}")
        raise ImportError(f"Could not find or load components for env type '{env_type}': {e}")

    # If a VLM is used, the environment MUST be in 'rgb_array' mode for rendering.
    gym_make_render_mode = "rgb_array" if config.AGENT_MODEL_TYPE == "vlm" else config.RENDER_MODE
    logging.info(f"Creating Gym environment '{env_name}' with base render_mode='{gym_make_render_mode}'")
    
    try:
        env = gym.make(env_name, render_mode=gym_make_render_mode)
    except Exception as e:
        logging.error(f"Failed to create Gym environment '{env_name}': {e}")
        raise RuntimeError(f"Failed to create Gym environment: {e}")
    
    adapter = adapter_class(env)

    # --- Environment-Specific Wrappers ---
    if env_type == "minigrid":
        logging.info("Applying CustomMiniGridRewardWrapper for shaping rewards.")
        env = CustomMiniGridRewardWrapper(env)

    # --- User-Facing Rendering Wrappers ---
    if config.RENDER_MODE == "human":
        logging.info("Live viewing requested. Applying HumanRendering wrapper.")
        env = HumanRendering(env)
    elif config.RENDER_MODE == "record":
        logging.info("Recording requested. Applying RecordVideo wrapper.")
        os.makedirs(config.RECORDING_DIR, exist_ok=True)
        env = RecordVideo(env, video_folder=config.RECORDING_DIR, name_prefix=env_name)
        logging.info(f"Video recordings will be saved to '{config.RECORDING_DIR}'.")

    logging.info(f"Environment '{env_name}' and components created successfully.")
    return env, adapter, get_prompt_func, get_visual_prompt_func, get_outcome_from_reward_func, create_memory_summary_func
