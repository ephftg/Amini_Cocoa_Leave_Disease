from src.utils.base_registry import BaseRegistry


class ProcessorRegistry(BaseRegistry):
    """
    Class that contains a registry of all the Processors
    available for the different models as different models
    require different processing.

    Each new Processor needs to be registered in this registry
    with the same name as the model's name for the model to use it.

    Inherits from BaseRegistry.
    """
