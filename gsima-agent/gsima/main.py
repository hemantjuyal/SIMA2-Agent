import logging
import os
import json
from datetime import datetime

from gsima import environments
from gsima.utils.logging import setup_logging
from gsima.utils import config
from gsima import runtime as runtime_factory
from gsima.agents import create_agent
from gsima.agents.context import AgentContext

def log_configuration():
    """Logs the agent's starting configuration."""
    logging.info("Starting agent run with the following configuration:")
    logging.info(f"RUNTIME: {config.RUNTIME}")
    if config.RUNTIME == "gemini":
        logging.info(f"GEMINI_MODEL: {config.GEMINI_MODEL}")
    
    logging.info(f"GYM_ENVIRONMENT: {config.GYM_ENVIRONMENT}")
    logging.info(f"ENV_TYPE: {config.ENV_TYPE}")
    logging.info(f"RENDER_MODE: {config.RENDER_MODE}")
    logging.info(f"EVAL_EPISODES: {config.EVAL_EPISODES}")
    logging.info(f"MAX_STEPS: {config.MAX_STEPS}")
    logging.info(f"INSTRUCTION: {config.INSTRUCTION}")
    logging.info(f"MEMORY_LENGTH: {config.MEMORY_LENGTH}")

def main():
    """Sets up and runs the gsima-agent evaluation harness."""
    setup_logging()
    log_configuration()
    
    env = None
    try:
        # 1. Create all modular components using their factories
        multimodal_runtime = None
        if config.RUNTIME == "gemini":
            multimodal_runtime = runtime_factory.create_runtime(config.RUNTIME, "multimodal", config.GEMINI_MODEL)
        else:
            raise ValueError("Only Gemini multimodal runtime is supported.")
        
        (
            env,
            adapter,
            memory_system,
            get_multimodal_prompt,
        ) = environments.create_env_and_adapter()

        logging.info(f"Adapter metadata: {adapter.get_env_metadata()}")
        logging.info(f"Action metadata: {adapter.get_action_metadata()}")

        context = AgentContext(
            env=env,
            adapter=adapter,
            multimodal_runtime=multimodal_runtime,
            memory_system=memory_system,
            get_multimodal_prompt=get_multimodal_prompt,
        )

        agent = create_agent()
        logging.info(f"Running agent: {agent.name} for {config.EVAL_EPISODES} episodes...")
        
        # Initialize human rendering window if applicable
        if config.RENDER_MODE == "human":
            env.render()
        
        eval_results = []
        output_dir = os.path.join("outputs", "evals", config.GYM_ENVIRONMENT)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        
        for episode in range(config.EVAL_EPISODES):
            logging.info(f"=== Starting Episode {episode + 1}/{config.EVAL_EPISODES} ===")
            metrics = agent.run(context)
            metrics["episode"] = episode + 1
            eval_results.append(metrics)
            logging.info(f"Episode {episode + 1} finished with metrics: {metrics}")
            
            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(eval_results, f, indent=4)
                
        # Calculate summary
        success_count = sum(1 for r in eval_results if r["success"])
        success_rate = (success_count / config.EVAL_EPISODES) * 100
        logging.info(f"=== Evaluation Complete ===")
        logging.info(f"Total Episodes: {config.EVAL_EPISODES}")
        logging.info(f"Success Rate: {success_rate:.1f}%")
        logging.info(f"Results saved to: {output_file}")

    except Exception as e:
        logging.critical(f"A critical error occurred during agent setup or execution: {e}", exc_info=True)
    finally:
        if env:
            env.close()
        logging.info("Run complete.")

if __name__ == "__main__":
    main()
