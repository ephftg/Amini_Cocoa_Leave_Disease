import importlib
import logging
from typing import Any, Callable

from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("pipeline")


class BaseRegistry:
    """
    Generic registry that allows classes to register itself
    to the registry which is a dictionary of all available types.

    The registry is refreshed for each subclass as they represent different types of classes.
    The keys of the registry is the name to reference the type in the registry.
    """

    _REGISTRY: dict[str, Any] = {}

    def __init_subclass__(cls):
        """
        Execute once when the new subclass is defined.
        This ensures a subclass starts with empty registry.
        """
        super().__init_subclass__()
        cls._REGISTRY = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        """
        A decorator factory that registers a class under a name.

        Usage: Add @<registry name>.register(<name to register under)
                above a class.

        Args:
            name: Name for which the type will be registered under.

        Returns:
            Callable: A decorator function that registers and returns the type.
        """

        def decorator(entry_cls: type[Any]) -> type[Any]:
            if name in cls._REGISTRY:
                logger.info(
                    f"{name!r} already registered in {cls.__name__}, "
                    f"overwriting with {entry_cls.__name__!r}"
                )
            cls._REGISTRY[name] = entry_cls
            logger.info(
                f"{entry_cls.__name__!r} registered under {name!r} in {cls.__name__}"
            )
            return entry_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Any]:
        """
        Get the class available in the registry using the registered name.

        Args:
            name: Name that the class is registered under in the registry.

        Returns:
            Class that correspond to the name in the registry.

        Raises:
            KeyError: if name is not found in registry.
        """
        # list available processor if not found
        try:
            return cls._REGISTRY[name]
        except KeyError:
            available = ", ".join(sorted(cls._REGISTRY)) or "<none>"
            raise KeyError(
                f"No entry registered for {name!r} in {cls.__name__}. "
                f"Available: {available}"
            ) from None

    @classmethod
    def available(cls) -> list[str]:
        """
        Return a list of the names of the registered classes in registry.
        """
        return sorted(cls._REGISTRY)

    @classmethod
    def load_plugins(cls, module_paths: list[str]) -> None:
        """
        Import a list of modules.

        The modules should contain the classes with decorator
        "<registry>.register()" that is use to register the class to registry.

        This allows classes to be registered without explicit imports
        in the main code, where adding a new class means just
        adding to the list of modules.

        Args:
            module_paths: List containing module paths of class
                        to be registered into registry relative
                        to the main code.
        """
        for path in module_paths:
            importlib.import_module(path)
            logger.info(f"Loaded plugin module: {path}")
