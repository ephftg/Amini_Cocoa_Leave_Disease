import importlib
import logging
from typing import Dict, List, Type

from src.utils.logging import setup_logging
from src.processor.base_processor import BaseProcessor

setup_logging()
logger = logging.getLogger("pipeline")


class ProcessorRegistry:
    """
    Class that contains a registry of all the Processors
    available for the different models as different models
    require different processing.

    Each new Processor needs to be registered in this registry
    with the same name as the model's name for the model to use it.
    """

    # registry with the model's name being the keys
    _REGISTRY: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        A decorator factory that registers a processor under a name.

        Usage: Add @ProcessorRegistry.register(<name to register under)
                above a processor class.

        Args:
            name: Name for which the processor will be registered under.

        Returns:
            Callable: A decorator function that registers the input processor
                    and then returns the processor.
        """

        def decorator(proc_cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
            """
            Register the Processor into the Processor regsitry and returns it.
            """
            # over write previously registered processor
            if name in cls._REGISTRY:
                logger.info(
                    f"Model {name} is already registered, overwriting with {(proc_cls.__name__,)}"
                )
            cls._REGISTRY[name] = proc_cls
            logger.info(f"Model {proc_cls.__name__} is registerd under {name}")
            return proc_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseProcessor]:
        """
        Get the processor available in the registry using the registered name.

        Args:
            name: Name that the processor is registered under in the registry.

        Returns:
            Processor that correspond to the name in the registry.

        Raises:
            KeyError: if name is not found in registry.
        """
        # list available processor if not found
        try:
            return cls._REGISTRY[name]
        except KeyError:
            available = ", ".join(sorted(cls._REGISTRY)) or "<none>"
            raise KeyError(
                f"No processor registered for '{name}'. Available: {available}"
            ) from None

    @classmethod
    def available(cls) -> List[str]:
        """
        Return a list of the names of the registered processor in registry.
        """
        return cls._REGISTRY

    @classmethod
    def load_plugins(cls, module_paths: List[str]) -> None:
        """
        Import a list of modules.

        The modules should contain the classes with decorator
        "ProcessorRegistry.register()" to so as to register the processors
        to the processor registry for use.

        This allows processors to be registered without explicit imports
        in the main code, where adding a new processor means just
        adding to the list of modules.

        Args:
            module_paths: List containing module paths of processor
                        to be registered into processor registry relative
                        to the main code.
        """
        for path in module_paths:
            importlib.import_module(path)
            logger.info(f"Loaded model: {path}")
