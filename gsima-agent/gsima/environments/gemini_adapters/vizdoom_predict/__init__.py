"""VizDoom environment package.

This package contains the environment-specific adapter, prompt templates,
and schema definitions needed to plug a VizDoom Gymnasium environment into
SIMA2-Agent without changing the agent runtime.
"""

import gymnasium as gym
import numpy as np

try:
    import vizdoom  # noqa: F401
    from vizdoom import gymnasium_wrapper  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-specific dependency
    vizdoom = None
    gymnasium_wrapper = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class DelayTerminationWrapper(gym.Wrapper):
    """Wrapper to delay episode termination after the target is killed.

    When the underlying environment returns terminated=True (meaning the monster is dead),
    this wrapper intercepts it, returns terminated=False, and for the next `delay_steps` steps,
    it overrides the agent's actions to NOOP [0, 0, 0] (discrete action 0) to allow the death
    animation to fully play out in the video recording.
    """
    def __init__(self, env, delay_steps: int = 25):
        super().__init__(env)
        self.delay_steps = delay_steps
        self.steps_after_termination = 0
        self.termination_triggered = False
        self.saved_terminated = False
        self.saved_truncated = False
        self.last_valid_obs = None

    def reset(self, **kwargs):
        self.steps_after_termination = 0
        self.termination_triggered = False
        self.saved_terminated = False
        self.saved_truncated = False
        self.last_valid_obs = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.termination_triggered:
            # We are in the delay phase. Force NOOP action to let the animation play.
            if isinstance(self.action_space, gym.spaces.Discrete):
                noop_action = 0
            elif isinstance(self.action_space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
                noop_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
            else:
                # Fallback to general zero list
                noop_action = [0, 0, 0]

            obs, reward, term, trunc, info = self.env.step(noop_action)
            self.steps_after_termination += 1
            
            # During the delay phase, force terminated/truncated to False
            # until we have completed the requested delay_steps.
            if self.steps_after_termination >= self.delay_steps:
                return self.last_valid_obs, reward, self.saved_terminated, self.saved_truncated, info
            else:
                info["is_delay_phase"] = True
                return self.last_valid_obs, reward, False, False, info
        
        # Normal phase
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # First time reaching termination. Start the delay phase!
            self.termination_triggered = True
            self.saved_terminated = terminated
            self.saved_truncated = truncated
            self.last_valid_obs = obs
            # Return False for termination to keep the episode alive for recording
            info["is_delay_phase"] = True
            return obs, reward, False, False, info
            
        return obs, reward, terminated, truncated, info


def apply_env_wrappers(env):
    """Apply environment wrappers for VizDoom.

    Wraps the environment with DelayTerminationWrapper to ensure the monster's
    death animation is fully captured in video recordings.
    """
    return DelayTerminationWrapper(env, delay_steps=25)


def ensure_registration():
    """Ensure VizDoom Gymnasium registrations are available.

    The official wrapper registers environments only when the package is imported.
    This helper makes that dependency explicit for the factory.
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "VizDoom Gymnasium support could not be imported. "
            f"Original error: {_IMPORT_ERROR}"
        )
    return True
