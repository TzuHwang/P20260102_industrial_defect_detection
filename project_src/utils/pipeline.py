def get_attr(obj, name, default=None):
    """
    Safely get an attribute from an object with a default fallback.

    Args:
        obj: The object to get the attribute from (e.g., argparse.Namespace, SimpleNamespace).
        name: The attribute name.
        default: The default value if the attribute doesn't exist.

    Returns:
        The attribute value if it exists, otherwise the default value.
    """
    return getattr(obj, name, default)
