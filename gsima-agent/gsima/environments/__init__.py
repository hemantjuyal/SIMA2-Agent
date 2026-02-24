import os
import importlib
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, HumanRendering

from gsima.utils import config
from gsima.environments.base import BaseAdapter
from gsima.memory import create_memory
import minigrid  # Explicitly import minigrid to register its environments
from gsima.environments.minigrid.custom_reward_wrapper import CustomMiniGridRewardWrapper

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
    try:
        env_type = env_name.split('-')[0].lower()
        if env_type == "minigrid":
            adapter_class_name = "MiniGridAdapter"
        else:
            adapter_class_name = f"{env_type.capitalize()}Adapter"
        logging.info(f"Detected environment type: '{env_type}'")
    except IndexError:
        raise ValueError(f"Invalid environment name format: {env_name}")

    # --- Dynamic Module and Component Loading ---
    try:
        adapter_module = importlib.import_module(f"gsima.environments.{env_type}.adapter")
        adapter_class = getattr(adapter_module, adapter_class_name)

        prompt_module = importlib.import_module(f"gsima.environments.{env_type}.prompt")
        get_visual_prompt_func = getattr(prompt_module, "get_visual_prompt")
        get_simulator_prompt_func = getattr(prompt_module, "get_simulator_prompt")
        get_controller_prompt_func = getattr(prompt_module, "get_controller_prompt")
        get_outcome_from_reward_func = getattr(prompt_module, "get_outcome_from_reward")
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
    
    try:
        env = gym.make(env_name, render_mode=gym_make_render_mode)
    except Exception as e:
        raise RuntimeError(f"Failed to create Gym environment: {e}")
    
    adapter = adapter_class(env)

    # --- Environment-Specific Wrappers ---
    if env_type == "minigrid":
        logging.info("Applying CustomMiniGridRewardWrapper for shaping rewards.")
        env = CustomMiniGridRewardWrapper(env)

    # --- User-Facing Rendering Wrappers ---
    if config.RENDER_MODE == "human":
        env = HumanRendering(env)
    elif config.RENDER_MODE == "record":
        os.makedirs(config.RECORDING_DIR, exist_ok=True)
        env = RecordVideo(env, video_folder=config.RECORDING_DIR, name_prefix=env_name)
        logging.info(f"Video recordings will be saved to '{config.RECORDING_DIR}'.")

    logging.info(f"Environment '{env_name}' and components created successfully.")
    return (
        env,
        adapter,
        memory_system,
        get_visual_prompt_func,
        get_simulator_prompt_func,
        get_controller_prompt_func,
        get_outcome_from_reward_func,
    )

