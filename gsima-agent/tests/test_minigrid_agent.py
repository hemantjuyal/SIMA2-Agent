import unittest
from unittest.mock import MagicMock

from gsima.agent import run_agent_loop
from gsima import environments
from gsima.utils import config
from tests.base_test_agent import BaseAgentTests

class TestMiniGridAgent(BaseAgentTests):
    """
    Concrete test suite for running the agent in a MiniGrid environment.
    """

    def setUp(self):
        """
        Sets up the testing environment for MiniGrid.
        - Mocks the LLM and VLM runtimes to return predictable values.
        - Initializes a real MiniGrid environment.
        """
        # --- 1. Configure the environment for this test suite ---
        config.GYM_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
        config.AGENT_MODEL_TYPE = "vlm"
        config.MAX_STEPS = 5 # Ensure the test runs quickly
        config.RENDER_MODE = "rgb_array" # Necessary for VLM agent

        # --- 2. Create mock runtimes to simulate model responses ---
        # Mock VLM will return a fixed description
        self.vlm_runtime = MagicMock()
        self.vlm_runtime.get_model_response.return_value = "You see an empty room with a green square."

        # Mock LLM will always decide to move forward, then stop.
        self.llm_runtime = MagicMock()
        self.llm_runtime.get_model_response.side_effect = [
            '{"action": "MOVE_FORWARD"}',
            '{"action": "MOVE_FORWARD"}',
            '{"action": "STOP"}'
        ]

        # --- 3. Get the real environment components using the factory ---
        (
            self.env,
            self.adapter,
            self.get_prompt,
            self.get_visual_prompt,
            self.get_outcome_from_reward,
            self.create_memory_summary,
        ) = environments.create_env_and_adapter()

    def test_run_agent_loop_completes_without_crash(self):
        """
        An integration test to ensure the `run_agent_loop` can execute
        with the MiniGrid environment and mocked runtimes without crashing.
        This validates that all the refactored components work together correctly.
        """
        try:
            run_agent_loop(
                env=self.env,
                adapter=self.adapter,
                vlm_runtime=self.vlm_runtime,
                llm_runtime=self.llm_runtime,
                get_prompt=self.get_prompt,
                get_visual_prompt=self.get_visual_prompt,
                get_outcome_from_reward=self.get_outcome_from_reward,
                create_memory_summary=self.create_memory_summary,
            )
        except Exception as e:
            self.fail(f"run_agent_loop raised an unexpected exception: {e}")
            
        # --- 4. Assert that the models were called as expected ---
        self.assertGreater(self.vlm_runtime.get_model_response.call_count, 0, "VLM runtime was not called.")
        self.assertGreater(self.llm_runtime.get_model_response.call_count, 0, "LLM runtime was not called.")
        # The loop should have stopped early because of the mocked "STOP" action
        self.assertEqual(self.llm_runtime.get_model_response.call_count, 3)


if __name__ == '__main__':
    unittest.main()
