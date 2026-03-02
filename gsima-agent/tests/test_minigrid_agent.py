import unittest
from unittest.mock import MagicMock

from gsima import environments
from gsima.utils import config
from gsima.agents import create_agent
from gsima.agents.context import AgentContext
from gsima.environments.minigrid.schema import SUPPORTED_ACTIONS
from tests.base_test_agent import BaseAgentTests

class TestMiniGridAgent(BaseAgentTests):
    """
    Concrete test suite for running the agent in a MiniGrid environment.
    """

    def setUp(self):
        """
        Sets up the testing environment for MiniGrid.
        - Mocks the Perception and Controller runtimes.
        - Initializes a real MiniGrid environment and memory system via the environment factory.
        """
        config.GYM_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
        config.AGENT_ARCH = "world_model"
        config.MAX_STEPS = 5
        config.RENDER_MODE = "rgb_array"

        # --- Mock Runtimes ---
        self.perception_runtime = MagicMock()
        self.controller_runtime = MagicMock()

        # Mock the Perception VLM to return a Markdown string
        self.perception_runtime.get_model_response.return_value = (
            "### Scene Analysis\n"
            "- **Agent Orientation**: [SOUTH]\n"
            "- **Goal Relative Position**: [front-right]\n"
            "- **Obstacle In Front**: [false]"
        )

        # Mock the Controller LLM to return a thought and action.
        # The agent will now receive perfect "imagined futures" from the deterministic simulator.
        self.controller_runtime.get_model_response.side_effect = [
            "- **thought**: [The path is clear and the goal is generally in front of me, so I will move forward.]\n- **action**: [MOVE_FORWARD]",
            "- **thought**: [Continuing forward seems correct.]\n- **action**: [MOVE_FORWARD]",
            "- **thought**: [I believe I am near the goal now, I should stop to complete the mission.]\n- **action**: [STOP]",
        ]

        # --- Get Environment and its Configured Components ---
        (
            self.env,
            self.adapter,
            self.memory_system,
            self.get_visual_prompt,
            self.get_controller_prompt,
            self.get_outcome_from_reward,
        ) = environments.create_env_and_adapter()
    
    def test_agent_run_completes_without_crash(self):
        """
        An integration test to ensure the agent.run() method can execute
        with all the modular components working together.
        """
        # 1. Assemble the context for the agent
        context = AgentContext(
            env=self.env,
            adapter=self.adapter,
            perception_runtime=self.perception_runtime,
            controller_runtime=self.controller_runtime,
            memory_system=self.memory_system,
            get_visual_prompt=self.get_visual_prompt,
            get_controller_prompt=self.get_controller_prompt,
            get_outcome_from_reward=self.get_outcome_from_reward,
        )

        # 2. Create the agent and run it
        agent = create_agent()
        
        try:
            agent.run(context)
        except Exception as e:
            self.fail(f"agent.run() raised an unexpected exception: {e}")
            
        # 3. Assert that the mocked models were called as expected
        self.assertGreater(self.perception_runtime.get_model_response.call_count, 0)
        self.assertGreater(self.controller_runtime.get_model_response.call_count, 0)
