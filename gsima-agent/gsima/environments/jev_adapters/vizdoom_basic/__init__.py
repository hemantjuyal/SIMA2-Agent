"""Package marker for ViZDoom Basic Jev Adapter."""

def ensure_registration():
    """Ensure the ViZDoom environments are registered in Gymnasium."""
    import vizdoom  # noqa: F401
    from vizdoom import gymnasium_wrapper  # noqa: F401
    
    # vizdoom environments are auto-registered when gymnasium_wrapper is imported
    pass
