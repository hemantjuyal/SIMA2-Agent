import unittest
from unittest.mock import MagicMock
import gymnasium as gym
from abc import ABC, abstractmethod

from gsima.utils import config

class BaseAgentTests(unittest.TestCase, ABC):
    """
    Abstract base class for agent tests.

    This class should not be run directly. Subclasses must implement the
    `setUp` method to configure the environment and mock the necessary components.
    The `run_agent_loop` function is the primary target for integration testing.
    """

    @abstractmethod
    def setUp(self):
        """
        This method MUST be overridden by subclasses.

        It is responsible for:
        1. Setting the `config.GYM_ENVIRONMENT` and other relevant configs.
        2. Creating mock instances for `vlm_runtime` and `llm_runtime`.
        3. Calling `environments.create_env_and_adapter()` to get the
           environment-specific components.
        4. Assigning all created components to `self`.
        """
        # --- To be implemented by subclass ---
        self.env: gym.Env = None
        self.adapter = None
        self.vlm_runtime: MagicMock = None
        self.llm_runtime: MagicMock = None
        self.get_prompt: callable = None
        self.get_visual_prompt: callable = None
        self.get_outcome_from_reward: callable = None
        self.create_memory_summary: callable = None

        # --- Example Mock Setup ---
        # self.llm_runtime = MagicMock()
        # self.llm_runtime.get_model_response.return_value = '{"action": "STOP"}'
        
        # raise NotImplementedError("The setUp method must be implemented by a test subclass.") # This line is no longer needed

    def test_components_are_initialized(self):
        """
        A generic test to ensure that the setUp method of a subclass
        has correctly initialized all the necessary components.
        """
        self.assertIsNotNone(self.env, "Environment (env) was not initialized in setUp.")
        self.assertIsNotNone(self.adapter, "Adapter was not initialized in setUp.")
        self.assertIsNotNone(self.llm_runtime, "LLM Runtime was not initialized in setUp.")
        self.assertIsNotNone(self.get_prompt, "get_prompt function was not initialized.")

        if config.AGENT_MODEL_TYPE == "vlm":
            self.assertIsNotNone(self.vlm_runtime, "VLM Runtime was not initialized for a VLM agent.")
            self.assertIsNotNone(self.get_visual_prompt, "get_visual_prompt function was not initialized.")

    def tearDown(self):
        """
        Cleans up resources after each test.
        """
        if self.env:
            self.env.close()
            
