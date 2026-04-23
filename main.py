import hydra
from omegaconf import DictConfig
import logging

from src.pipeline.train import Train
from src.utils.logging import setup_logging
from src.processor.processor_registry import ProcessorRegistry

setup_logging()
logger = logging.getLogger("pipeline")


@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # initialize train class and register all models
    Train.load_plugins(cfg["model_plugins"])
    trainer = Train(cfg)

    # initialize processor class and register all processors
    ProcessorRegistry.load_plugins(cfg["processor_plugins"])

    # use mode to call relevant function
    mode = cfg["mode"]

    if mode == "hyperparameter_tune":
        best_params = trainer.hyperparameter_tune()
        logger.info(f"best parameter from training: {best_params}")
    elif mode == "train_full_set":
        test_eval = trainer.train_full_set()
        logger.info(f"Test set evaluation: {test_eval}")
    else:
        raise ValueError(
            f"Invalid mode {mode}. Must be either hyperparameter_tune or train_full_set"
        )


if __name__ == "__main__":
    main()
