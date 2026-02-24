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
        - Mocks the Perception, Simulator, and Controller runtimes.
        - Initializes a real MiniGrid environment and memory system via the environment factory.
        """
        config.GYM_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
        config.AGENT_ARCH = "world_model"
        config.MAX_STEPS = 5
        config.RENDER_MODE = "rgb_array"

        # --- Mock Runtimes ---
        self.perception_runtime = MagicMock()
        self.simulator_runtime = MagicMock()
        self.controller_runtime = MagicMock()

        # Mock the Perception VLM to return a Markdown string
        self.perception_runtime.get_model_response.return_value = (
            "### Scene Analysis\n"
            "- **Agent Orientation**: [SOUTH]\n"
            "- **Goal Relative Position**: [front-right]\n"
            "- **Obstacle In Front**: [false]"
        )

        # Mock the Simulator LLM to return a predicted next state for each action
        # The WorldModelAgent tries all supported actions
        base_simulator_mock_effects = []
        for action in SUPPORTED_ACTIONS:
            # Simple mock: assume moving forward clears obstacle, turns change orientation
            if action.name == "MOVE_FORWARD":
                base_simulator_mock_effects.append(
                    "- **Agent Orientation**: [SOUTH]\n"
                    "- **Goal Relative Position**: [front-right]\n"
                    "- **Obstacle In Front**: [false]" # Still false after moving
                )
            elif action.name == "TURN_LEFT":
                base_simulator_mock_effects.append(
                    "- **Agent Orientation**: [EAST]\n" # Assuming agent was facing SOUTH, turning left makes it face EAST
                    "- **Goal Relative Position**: [right]\n"
                    "- **Obstacle In Front**: [false]"
                )
            elif action.name == "TURN_RIGHT":
                base_simulator_mock_effects.append(
                    "- **Agent Orientation**: [WEST]\n" # Assuming agent was facing SOUTH, turning right makes it face WEST
                    "- **Goal Relative Position**: [left]\n"
                    "- **Obstacle In Front**: [false]"
                )
            elif action.name == "STOP":
                base_simulator_mock_effects.append(
                    "- **Agent Orientation**: [SOUTH]\n"
                    "- **Goal Relative Position**: [front-right]\n"
                    "- **Obstacle In Front**: [false]"
                )
            else:
                base_simulator_mock_effects.append("- **Unknown State**: [true]")
        
        # Repeat the base effects for each step
        self.simulator_runtime.get_model_response.side_effect = base_simulator_mock_effects * config.MAX_STEPS

        # Mock the Controller LLM to return a thought and action based on perceived state and imagined futures
        # This will be called repeatedly, so we need a list of effects
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
            self.get_simulator_prompt,
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
            simulator_runtime=self.simulator_runtime,
            controller_runtime=self.controller_runtime,
            memory_system=self.memory_system,
            get_visual_prompt=self.get_visual_prompt,
            get_simulator_prompt=self.get_simulator_prompt,
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
        # Simulator should be called for each possible action for each step
        self.assertGreater(self.simulator_runtime.get_model_response.call_count, 0)
        self.assertGreater(self.controller_runtime.get_model_response.call_count, 0)
