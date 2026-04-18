import importlib
import logging
from typing import Dict, List, Type

from src.utils.logging import setup_logging
from src.processor.base_processor import BaseProcessor

setup_logging()
logger = logging.getLogger("pipeline")


class ProcessorRegistry:
    _REGISTRY: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Register the model specific processor to train/test set.

        Usage::

            @ProcessorRegistry.register("FasterRCNN")
            class FasterRCNNProcessor(BaseProcessor):
                ...
        """

        def decorator(proc_cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
            if name in cls._REGISTRY:
                logger.info(
                    f"Model {name} is already registered, overwriting with {(proc_cls.__name__,)}"
                )
            cls._REGISTRY[name] = proc_cls
            logger.info(f"Model {proc_cls.__name__} is registerd under {name}")
            return proc_cls  # always return the class unchanged

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseProcessor]:
        try:
            return cls._REGISTRY[name]
        except KeyError:
            available = ", ".join(sorted(cls._REGISTRY)) or "<none>"
            raise KeyError(
                f"No processor registered for '{name}'. Available: {available}"
            ) from None

    @classmethod
    def available(cls) -> List[str]:
        return cls._REGISTRY

    @classmethod
    def load_plugins(cls, module_paths: List[str]) -> None:
        """Import modules so their @ProcessorRegistry.register decorators fire."""
        for path in module_paths:
            importlib.import_module(path)
            logger.info(f"Loaded model: {path}")
