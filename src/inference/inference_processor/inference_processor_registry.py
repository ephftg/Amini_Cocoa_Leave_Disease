from src.utils.base_registry import BaseRegistry


class InferenceProcessorRegistry(BaseRegistry):
    """
    Class that contains a registry of all the InferenceProcessors
    available for the different models as different models
    require different pre-processing of image as input and
    post-processing of output (bounding boxes).

    Each new InferenceProcessor needs to be registered in this registry
    to use it.

    Inherits from BaseRegistry.
    """
