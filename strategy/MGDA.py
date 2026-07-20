import torch
from strategy.MinNorm import MinNormSolver

def update_MGDA(grads):
    """
    Update dynamic weights using MGDA.

    Args:
        grads (dict):
            Mapping from model name to gradient.

    Returns:
        dict:
            Mapping from model name to MGDA weight.
    """
    model_names = list(grads.keys())

    sol, _ = MinNormSolver.find_min_norm_element(
        list(grads.values())
    )

    return {
        model: float(weight)
        for model, weight in zip(model_names, sol)
    }