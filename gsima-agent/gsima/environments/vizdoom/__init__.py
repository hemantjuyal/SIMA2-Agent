"""VizDoom environment package.

This package contains the environment-specific adapter, prompt templates,
and schema definitions needed to plug a VizDoom Gymnasium environment into
SIMA2-Agent without changing the agent runtime.
"""

try:
    import vizdoom  # noqa: F401
    from vizdoom import gymnasium_wrapper  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-specific dependency
    vizdoom = None
    gymnasium_wrapper = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def apply_env_wrappers(env):
    """Return the environment unchanged for now.

    VizDoom Gymnasium environments already expose the observation and reward
    interfaces needed by the generic agent loop.
    """
    return env


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
