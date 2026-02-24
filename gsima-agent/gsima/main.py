import logging

from gsima import environments
from gsima.utils.logging import setup_logging
from gsima.utils import config
from gsima import runtime as runtime_factory
from gsima.agents import create_agent
from gsima.agents.context import AgentContext

def log_configuration():
    """Logs the agent's starting configuration."""
    logging.info("Starting agent run with the following configuration:")
    logging.info(f"AGENT_ARCH: {config.AGENT_ARCH}")
    logging.info(f"RUNTIME: {config.RUNTIME}")

    if config.RUNTIME == "ollama":
        logging.info(f"OLLAMA_PERCEPTION_MODEL: {config.OLLAMA_PERCEPTION_MODEL}")
        logging.info(f"OLLAMA_SIMULATOR_MODEL: {config.OLLAMA_SIMULATOR_MODEL}")
        logging.info(f"OLLAMA_CONTROLLER_MODEL: {config.OLLAMA_CONTROLLER_MODEL}")
    else: # Default to mlx
        logging.info(f"PERCEPTION_MODEL_ID: {config.PERCEPTION_MODEL_ID}")
        logging.info(f"SIMULATOR_MODEL_ID: {config.SIMULATOR_MODEL_ID}")
        logging.info(f"CONTROLLER_MODEL_ID: {config.CONTROLLER_MODEL_ID}")
    
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
        # Create runtimes
        perception_runtime = None
        if config.AGENT_ARCH == "world_model": # Only world_model agent uses VLM for perception currently
            if config.RUNTIME == "ollama":
                perception_runtime = runtime_factory.create_runtime(config.RUNTIME, "vlm", config.OLLAMA_PERCEPTION_MODEL)
            else:
                perception_runtime = runtime_factory.create_runtime(config.RUNTIME, "vlm", config.PERCEPTION_MODEL_ID)
        
        simulator_runtime = None
        controller_runtime = None

        if config.RUNTIME == "ollama":
            simulator_runtime = runtime_factory.create_runtime(config.RUNTIME, "llm", config.OLLAMA_SIMULATOR_MODEL)
            controller_runtime = runtime_factory.create_runtime(config.RUNTIME, "llm", config.OLLAMA_CONTROLLER_MODEL)
        else: # mlx
            simulator_runtime = runtime_factory.create_runtime(config.RUNTIME, "llm", config.SIMULATOR_MODEL_ID)
            controller_runtime = runtime_factory.create_runtime(config.RUNTIME, "llm", config.CONTROLLER_MODEL_ID)
        
        (
            env,
            adapter,
            memory_system,
            get_visual_prompt,
            get_simulator_prompt,
            get_controller_prompt,
            get_outcome_from_reward,
        ) = environments.create_env_and_adapter()

        # 2. Assemble the context object
        context = AgentContext(
            env=env,
            adapter=adapter,
            perception_runtime=perception_runtime,
            simulator_runtime=simulator_runtime,
            controller_runtime=controller_runtime,
            memory_system=memory_system,
            get_visual_prompt=get_visual_prompt,
            get_simulator_prompt=get_simulator_prompt,
            get_controller_prompt=get_controller_prompt,
            get_outcome_from_reward=get_outcome_from_reward,
        )

        # 3. Create the agent and run it with the context
        agent = create_agent()
        logging.info(f"Running agent: {agent.name}...")
        
        # Initialize human rendering window if applicable
        if config.RENDER_MODE == "human":
            env.render()
        
        agent.run(context)

    except Exception as e:
        logging.critical(f"A critical error occurred during agent setup or execution: {e}", exc_info=True)
    finally:
        if env:
            env.close()
        logging.info("Run complete.")

if __name__ == "__main__":
    main()
