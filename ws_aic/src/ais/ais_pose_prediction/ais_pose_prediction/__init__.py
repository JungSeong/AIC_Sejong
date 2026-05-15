__all__ = ["PosePredictor"]


def __getattr__(name: str):
    if name == "PosePredictor":
        from .predictor import PosePredictor

        return PosePredictor
    raise AttributeError(name)
