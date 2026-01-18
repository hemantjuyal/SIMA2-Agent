import logging
import gymnasium as gym

from gsima.agent import run_agent_loop
from gsima import environments
from gsima.utils.logging import setup_logging
from gsima.utils import config
from gsima.runtime import get_llm_runtime, get_vlm_runtime

def log_configuration():
    """Logs the agent's starting configuration."""
    logging.info("Starting agent run with the following configuration:")
    logging.info(f"RUNTIME: {config.RUNTIME}")
    logging.info(f"AGENT_MODEL_TYPE: {config.AGENT_MODEL_TYPE}")
    if config.AGENT_MODEL_TYPE == "llm":
        logging.info(f"LLM_MODEL_ID: {config.LLM_MODEL_ID}")
    elif config.AGENT_MODEL_TYPE == "vlm":
        if config.RUNTIME == "ollama":
            logging.info(f"OLLAMA_VLM_MODEL: {config.OLLAMA_VLM_MODEL}")
            logging.info(f"OLLAMA_LLM_MODEL: {config.OLLAMA_LLM_MODEL}")
        else: # Default to mlx
            logging.info(f"VLM_MODEL_ID: {config.VLM_MODEL_ID}")
    logging.info(f"GYM_ENVIRONMENT: {config.GYM_ENVIRONMENT}")
    logging.info(f"RENDER_MODE: {config.RENDER_MODE}")
    logging.info(f"MAX_STEPS: {config.MAX_STEPS}")
    logging.info(f"INSTRUCTION: {config.INSTRUCTION}")
    logging.info(f"MEMORY_LENGTH: {config.MEMORY_LENGTH}")

def main():
    """Sets up and runs the gsima-agent."""
    setup_logging()
    log_configuration()
    
    # Initialize runtimes based on config
    vlm_runtime = get_vlm_runtime() if config.AGENT_MODEL_TYPE == "vlm" else None
    llm_runtime = get_llm_runtime()
    
    # Create the environment and its associated components
    env, adapter, get_prompt, get_visual_prompt, get_outcome_from_reward, create_memory_summary = environments.create_env_and_adapter()
    
    try:
        # Pass all necessary components to the agent loop
        run_agent_loop(
            env=env,
            adapter=adapter,
            vlm_runtime=vlm_runtime,
            llm_runtime=llm_runtime,
            get_prompt=get_prompt,
            get_visual_prompt=get_visual_prompt,
            get_outcome_from_reward=get_outcome_from_reward,
            create_memory_summary=create_memory_summary,
        )
    except Exception as e:
        logging.critical(f"An unhandled exception occurred in the agent loop: {e}", exc_info=True)
    finally:
        env.close()
        logging.info("Environment closed. Run complete.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
