"""MiniGrid environment package.

This package exposes the MiniGrid-specific registration hook needed so
Gymnasium can discover the environment IDs before the factory creates them.
"""

try:
    import minigrid  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-specific dependency
    minigrid = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def ensure_registration():
    """Ensure MiniGrid Gymnasium environments are registered.

    Importing the top-level ``minigrid`` package registers the MiniGrid envs.
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "MiniGrid support could not be imported. "
            f"Original error: {_IMPORT_ERROR}"
        )
    return True
