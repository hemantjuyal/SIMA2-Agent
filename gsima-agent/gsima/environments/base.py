"""Base classes for environment and adapter definitions."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAdapter(ABC):
    """Abstract base class for all environment adapters.

    Subclasses should provide environment-specific behavior while keeping
    a small set of generic capabilities that the agent can query.
    """

    def __init__(self, env):
        self.env = env

    def get_env_metadata(self) -> Dict[str, Any]:
        """Return a generic description of the environment capabilities."""
        return {}

    def get_action_metadata(self) -> Dict[str, Any]:
        """Return metadata about supported action names and semantics."""
        return {}

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a lightweight, environment-specific state snapshot."""
        return {}

    def get_current_env_state(self) -> Dict[str, Any]:
        """Return the raw environment state if available."""
        return {}

    def get_distance_to_goal(self, agent_pos=None) -> Optional[int]:
        """Return a progress signal when the environment has a goal concept."""
        return None

    def supports_stop(self) -> bool:
        """Whether the environment exposes a meaningful STOP action."""
        return False

    def get_progress(self) -> Dict[str, Any]:
        """Return a normalized progress view the agent can use for planning."""
        return {}

    def normalize_perception(
        self,
        structured_perception: Dict[str, Any],
        raw_response: str = "",
    ) -> Dict[str, Any]:
        """Normalize VLM perception into adapter-owned semantic keys.

        The base implementation is intentionally a pass-through. Backends can
        map model variations such as singular/plural labels or domain synonyms
        without teaching the generic world model about a specific environment.
        """
        return structured_perception

    def supports_simulation(self) -> bool:
        """Whether the adapter can simulate future trajectories deterministically."""
        return False

    def supports_goal_distance(self) -> bool:
        """Whether the environment exposes a meaningful distance-to-goal signal."""
        return False

    def suggest_action(self, structured_perception: Dict[str, Any]) -> Optional[str]:
        """Return an environment-specific action override when one is warranted.

        The default implementation leaves action selection to the generic planner.
        Environment adapters may override this hook to inject domain-specific
        heuristics without coupling the world model to a particular gym backend.
        """
        return None

    def get_fallback_action(self, structured_perception: Dict[str, Any]) -> Optional[str]:
        """Return a conservative adapter-owned fallback action.

        This is used only when the controller/planner cannot provide an action.
        The default follows the adapter's own supported-action order rather than
        encoding environment preferences in the world model.
        """
        actions = self.get_canonical_actions()
        return actions[0].name if actions else None

    def should_explain_action(
        self,
        action_name: str,
        structured_perception: Dict[str, Any],
    ) -> bool:
        """Whether the agent should ask the controller to explain an action.

        Some environments need fast reflex actions where blocking on an
        explanation-only model call hurts control quality. The default keeps the
        explanatory loop enabled for backends that do not opt out.
        """
        return True

    @abstractmethod
    def translate_action(self, action_name: str) -> any:
        """Translates a canonical action name into an env-specific action."""
        pass

    @abstractmethod
    def get_canonical_actions(self) -> list:
        """Returns a list of canonical actions supported by the environment."""
        pass
