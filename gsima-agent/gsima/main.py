import logging

from gsima import environments
from gsima.utils.logging import setup_logging
from gsima.utils import config
from gsima.runtime import get_llm_runtime, get_vlm_runtime
from gsima.agents import create_agent
from gsima.agents.context import AgentContext

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
    
    env = None # Ensure env is defined for the finally block
    try:
        # 1. Create all modular components using their factories
        vlm_runtime = get_vlm_runtime() if config.AGENT_MODEL_TYPE == "vlm" else None
        llm_runtime = get_llm_runtime()
        
        (
            env,
            adapter,
            memory_system,
            get_prompt,
            get_visual_prompt,
            get_outcome_from_reward,
        ) = environments.create_env_and_adapter()

        # 2. Assemble the context object
        context = AgentContext(
            env=env,
            adapter=adapter,
            vlm_runtime=vlm_runtime,
            llm_runtime=llm_runtime,
            memory_system=memory_system,
            get_prompt=get_prompt,
            get_visual_prompt=get_visual_prompt,
            get_outcome_from_reward=get_outcome_from_reward,
        )

        # 3. Create the agent and run it with the context
        agent = create_agent()
        logging.info(f"Running agent: {agent.name}...")
        agent.run(context)

    except Exception as e:
        logging.critical(f"A critical error occurred during agent setup or execution: {e}", exc_info=True)
    finally:
        if env:
            env.close()
        logging.info("Run complete.")

if __name__ == "__main__":
    main()
