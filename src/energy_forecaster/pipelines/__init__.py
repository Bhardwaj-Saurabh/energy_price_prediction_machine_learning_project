"""Kedro-based ML pipelines.

Each subpackage owns one logical pipeline (feature engineering, training,
inference, monitoring). Kedro itself is imported only inside this tree —
the application layer never sees it.
"""
