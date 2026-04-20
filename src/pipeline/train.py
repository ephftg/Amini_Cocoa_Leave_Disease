import importlib
import logging
from typing import Dict, List, Type, Tuple, Any, Optional
import pandas as pd
from omegaconf import DictConfig
import torch
from torch import Tensor
from pathlib import Path
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import gc

from src.utils.logging import setup_logging
from src.model.base_model import BaseModel
from src.utils.data_processing import train_test_split
from src.pipeline.train_helper import (
    DetectionDataset,
    EarlyStopping,
    EpochMetrics,
    Params,
)

setup_logging()
logger = logging.getLogger("pipeline")


def collate_fn(
    batch: List[Tuple[Tensor, Dict]],
) -> Tuple[Tuple[Tensor, ...], Tuple[Dict, ...]]:
    """
    Customized collate_fn for DataLoader for object detection, since image comes in different sizes
    and have different number of objects per image which is incompatible with default collation.

    Args:
        batch: List of samples from the Dataset where each sample is a tuple of image tensor and
             target dictionary containing bounding boxes and labels for all objects in the image.

    Returns:
        A tuple of two tuples:
            - Tuple of image tensors.
            - Tuple of target dicts.
    """
    return tuple(zip(*batch))


class Train:
    """
    Train class that allow for training the model either for hyperparameter tuning
    or  training with the full train+valid set.

    It also have a model registry that register all models available.
    """

    _MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        A decorator factory that registers a model under a name.

        Usage: Add @Train.register(<name to register under)
                above a model class.

        Args:
            name: Name for which the model will be registered under.

        Returns:
            Callable: A decorator function that registers the input model
                    and then returns the model.
        """

        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            """
            Register the Model into the Model regsitry and returns it.
            """
            if name in cls._MODEL_REGISTRY:
                logger.info(
                    f"Model {name} is already registered, overwriting with {(model_cls.__name__,)}"
                )
            cls._MODEL_REGISTRY[name] = model_cls
            logger.info(f"Model {model_cls.__name__} is registerd under {name}")
            return model_cls

        return decorator

    @classmethod
    def available_models(cls) -> List[str]:
        """Return a list of registered model names."""
        return cls._MODEL_REGISTRY

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        """
        Get the model available in the registry using the registered name.

        Args:
            name: Name that the model is registered under in the registry.

        Returns:
            Model that correspond to the name in the registry.

        Raises:
            KeyError: if name is not found in registry.
        """
        try:
            return cls._MODEL_REGISTRY[name]
        except KeyError:
            available = ", ".join(cls.available_models()) or "<none>"
            raise KeyError(
                f"Model '{name}' is not registered. Available models: {available}"
            ) from None

    @classmethod
    def load_model_plugins(cls, module_paths: List[str]) -> None:
        """
        Import a list of model modules.

        The model modules should contain the class with decorator
        "Train.register()" to so as to register the model to the model registry for use.

        This allows models to be registered without explicit imports
        in the main code, where adding a new model means just
        adding to the list of modules.

        Args:
            module_paths: List containing model's module paths relative to the main code.
        """
        for path in module_paths:
            importlib.import_module(path)
            logger.info(f"Loaded model: {path}")

    def __init__(self, cfg: DictConfig) -> None:
        """
        Create Train instance using the input configuration that provides the
        parameters values for training.

        Args:
            Configuration file contain the information required for training, including
            data paths, model and processor modules paths, mlflow information,
            training model name, mode and parameters.

        """
        self.cfg = cfg

        # use cuda if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device for training: {self.device}")

        # MLflow setup
        mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        self._label_map = None

        self.model_name = cfg["model_name"]

    def get_datasets(
        self,
    ) -> Tuple[DetectionDataset, DetectionDataset, DetectionDataset]:
        """
        Get the train, validation, and test datasets using the fraction indicated in config file.

        Returns:
            A tuple of train_dataset, val_dataset, test_dataset.
        """

        # read bounding box csv
        df = pd.read_csv(self.cfg["data"]["bbox_csv"])

        # get train+val and test dataframe first
        train_val_df, test_df = train_test_split(
            df, test_size=self.cfg["test_size"], seed=self.cfg["seed"]
        )

        # split train+val into train and val
        # val_size if relative to train+val
        val_size = self.cfg["val_size"]
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, seed=self.cfg["seed"]
        )

        image_dir = Path(self.cfg["data"]["image_dir"])

        train_ds = DetectionDataset(train_df, image_dir, self.model_name)
        val_ds = DetectionDataset(val_df, image_dir, self.model_name)
        test_ds = DetectionDataset(test_df, image_dir, self.model_name)

        # Sync label map
        self._label_map = train_ds.label_map

        logger.info(
            f"Datasets obtained. Train: {len(train_ds)}, "
            f"Val: {len(val_ds)}, Test: {len(test_ds)}"
        )

        return train_ds, val_ds, test_ds

    def get_kfold_splits(
        self, dataset: DetectionDataset
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Generate multilabel stratified K-Fold splits which ensures class distribution
        in train, valid set is maintained across all splits.

        Splitting is performed at the image level since each image could have multiple objects,
        to prevent data leakage between train and validation sets.

        Args:
            dataset: The dataset to split which is of type DetectionDataset, which has
                    "df" which is DataFrame of all objects present in each image and their corresponding class.
                    It has sorted image names for reproducibility.

        Returns:
            A list of containing k number of (train_indices, val_indices) tuples, where each indices are indices in dataset.image_names.
        """

        df = dataset.df
        image_names = dataset.image_names  # sorted list

        all_labels = sorted(df["class"].unique())

        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        # create binary matrix which indicate which objects are present in each image
        mlb_matrix = np.zeros((len(image_names), len(all_labels)), dtype=int)
        for i, name in enumerate(image_names):
            for lbl in df[df["Image_ID"] == name]["class"]:
                mlb_matrix[i, label_to_idx[lbl]] = 1

        # each image can contain multiple objects, so use MultilabelStratifiedKFold
        # instead of StratifiedKFold
        mskf = MultilabelStratifiedKFold(
            n_splits=self.cfg["kfold"], shuffle=True, random_state=self.cfg["seed"]
        )

        # split at image level using image name
        # a list of (train_indices, val_indices) tuples for each k fold split,
        splits = [
            (list(ti), list(vi))
            for ti, vi in mskf.split(np.arange(len(image_names)), mlb_matrix)
        ]

        logger.info(f"{self.cfg['kfold']} fold split done on Dataset")
        return splits

    def _run_one_epoch(
        self,
        model: BaseModel,
        loader: DataLoader,
        optimizer: Optional[Any],
        is_train: bool,
    ) -> Tuple[float, float]:
        """
        Run a single training or evaluation epoch over a DataLoader.

        In train mode, performs forward pass, loss backpropagation, and optimizer step.
        In evaluation mode, computes both loss and mAP metrics using separate passes.
        Metrics not evaluated in train mode to reduce double pass.

        When cuda is present, use automatic mixed precision for forward pass to reduce memory usage and FP32 for
        precise backpropagation.

        Clear all used variables and release unused cached memory to prevent OOM issue due to limited RAM.

        Args:
            model: Object detection model.
            loader: DataLoader that loads the Dataset that provides the correct format for the model in batches.
            optimizer: Optimizer for parameter updates. Required in train mode.
            is_train: Boolean value for train mode.

        Returns:
            A tuple of:
                - Average loss across all batches.
                - mAP@0.50:0.95, which is 0 in train mode.
                - mAP@0.50, which is 0 in train mode.
        """
        # if is_train = True, equivalent to model.train()
        # if is_train = False, equivalent to model.eval()
        model.train(is_train)

        total_loss = 0.0  # loss over all batches

        # set the metric that uses bbox iou for object detection
        metric = MeanAveragePrecision(iou_type="bbox")

        device_type = self.device.type
        use_amp = device_type == "cuda"

        logger.info(f"use amp: {use_amp}")

        phase = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=phase, unit="batch", leave=False)

        # logging interval
        log_interval = 100
        total_batches = len(loader)

        start_time = time.perf_counter()

        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.autocast(
                device_type=device_type, enabled=use_amp, dtype=torch.bfloat16
            ):
                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    loss_dict = model(
                        images, targets
                    )  # return all losses in dictionary
                    loss = sum(loss_dict.values())  # sum all types of losses
                    loss_dict.clear()

            if is_train:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                del loss
            else:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device_type, enabled=use_amp, dtype=torch.bfloat16
                    ):
                        # set the model to train mode so that output is loss dictionary to calculate loss
                        model.train()
                        loss_dict = model(images, targets)
                        total_loss += sum(loss_dict.values()).item()
                        loss_dict.clear()

                        # set the model to eval mode to get predictions, targets is omited
                        model.eval()
                        preds = model(images)

                    # move to cpu to free gpu
                    preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
                    targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                    # store predictions and targets by batch to evaluate when full dataset is available
                    metric.update(preds_cpu, targets_cpu)

                    del preds, preds_cpu, targets_cpu

            # log dataloader batch progress
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"[{phase}] Batch {batch_idx + 1}/{total_batches} completed"
                )

            # remove from RAM to free memory to prevent OOM
            del images, targets
            torch.cuda.empty_cache()

        mean_loss = total_loss / max(len(loader), 1)
        elapsed = time.perf_counter() - start_time

        # key value "map" value is mAP@0.50:0.95
        # key value "map_50" is mAP@0.50
        if not is_train:
            metric_results = metric.compute()
            map_score = metric_results["map"].item()
            map_50_score = metric_results["map_50"].item()

            # free the accumulated predictions and targets immediately to prevent OOM
            metric.reset()
        else:
            map_score = 0.0
            map_50_score = 0.0

        logger.info(
            f"[{phase}] Epoch done in {elapsed:.1f}s — "
            f"loss: {mean_loss:.4f}, mAP: {map_score:.4f}, mAP_50: {map_50_score:.4f}"
        )
        return mean_loss, map_score, map_50_score

    def get_hp_search_space(self, trial: optuna.Trial) -> dict:
        """
        Provide a set of hyperparameters from the search space provided in config file for provided Optuna trial.
        The sampling rules accords for catergorical and log scaled variables.

        Args:
            trial: The current Optuna trial object that contains the context of the hyperparameter tuning process
                    up to now and needs the next set of hyperparameter values.

        Returns:
            Dictionary with keys for the hyperparameter names and values for the sampled hyperparameter values.
        """

        space_cfg = self.cfg["hyperparameters"]
        params = {}
        for name, spec in space_cfg.items():
            # for categorical
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif (
                isinstance(spec["low"], int)
                and isinstance(spec["high"], int)
                and "log" not in spec
            ):
                # integer value for range
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], step=spec.get("step", 1)
                )
            else:
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
        return params

    def hyperparameter_tune(
        self,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning on train+valid set using Optuna search that aims to maximise the
        validation set mAP:0.5 since task does not required exact boxes.

        Train and evaluate loop defined in Optuna objective function.

        Training configurations and hyperparameter search space are defined in the config file.
        It includes the number of trials and start up trials.

        All trials' parameters and metrics are logged under a nested parent Mlflow run,
        and the best parameters and score among all trials are logged into the parent Mlflow run.

        Returns:
            Dictionary of the parameters that has the best score from all the trials or None
            if none of the trials completed.
        """

        if self.model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Registered: {self.available_models()}"
            )

        model_cls = self._MODEL_REGISTRY[self.model_name]
        train_ds, valid_ds, _ = self.get_datasets()
        max_epochs = self.cfg["max_epochs"]
        num_classes = self.cfg["num_classes"]
        run_name_prefix = "hparam_search"

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for Optuna search which consist of building the model, DataLoader,
            training and evaluating model across epochs.

            Log each trial's parameters and per epoch loss and metrics in Mlflow for tracking and storage.
            Do not log model's artifact to save storage and since it is not final model.

            For each epoch training, check for early stopping using validation loss to save compute and prevent overfitting.

            Omit kfold split train/evaluate to save on training time.
            Did not include trial prunner since epoch score can be noisy and already using early stopping.

            Remove any used objects and cache to prevent OOM error.

            Returns:
                The best mAP:0.5 score across the epochs as this metric evaluate performance of trial.

            Raises:
                Any OOM error.
            """
            torch.cuda.empty_cache()
            gc.collect()

            params = self.get_hp_search_space(trial)
            lr = params["learning_rate"]
            wd = params["weight_decay"]
            bs = params["batch_size"]

            # include trial number for tracking
            run_name = f"{run_name_prefix}_{self.model_name}_trial{trial.number}"

            obj_detect_model = None
            optimizer = None
            tr_loader = None
            val_loader = None

            # nested logging in common parent run
            with mlflow.start_run(run_name=run_name, nested=True):
                # log params
                fp = Params(learning_rate=lr, weight_decay=wd, batch_size=bs)
                mlflow.log_params(fp.as_dict())

                try:
                    tr_loader = DataLoader(
                        train_ds,
                        batch_size=int(bs),
                        shuffle=True,
                        num_workers=2,  # not too high to prevent OOM
                        pin_memory=True,  # faster cpu to gpu transfer
                        collate_fn=collate_fn,  # customized collate_fn for object detection
                        persistent_workers=False,  # respawn workers for each epoch to free memory
                    )
                    val_loader = DataLoader(
                        valid_ds,
                        batch_size=int(bs),
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        persistent_workers=False,
                    )

                    obj_detect_model = model_cls(num_classes=num_classes).to(
                        self.device
                    )
                    optimizer = AdamW(
                        obj_detect_model.trainable_parameters(),
                        lr=lr,
                        weight_decay=wd,
                    )
                    early_stop = EarlyStopping(
                        patience=self.cfg["early_stopping_patience"],
                        min_delta=self.cfg["early_stopping_min_delta"],
                    )

                    # track the best mAP@0.5 across epochs
                    best_map50 = 0.0

                    for epoch in tqdm(
                        range(1, max_epochs + 1), desc="Training", unit="epoch"
                    ):
                        tr_loss, _, _ = self._run_one_epoch(
                            obj_detect_model, tr_loader, optimizer, is_train=True
                        )
                        val_loss, val_map, val_map_50 = self._run_one_epoch(
                            obj_detect_model,
                            val_loader,
                            optimizer=None,
                            is_train=False,
                        )

                        # update the best
                        best_map50 = max(best_map50, val_map_50)

                        # log the epoch metrics
                        em = EpochMetrics(
                            epoch=epoch,
                            train_loss=tr_loss,
                            val_loss=val_loss,
                            val_map=val_map,
                            val_map_50=val_map_50,
                        )
                        mlflow.log_metrics(em.as_dict(), step=epoch)

                        # check for early stop criteria
                        if early_stop(val_loss):
                            logger.info(
                                f"Early stopping at epoch {epoch} for trial {trial.number}"
                            )
                            break

                except (
                    RuntimeError,
                    torch.cuda.OutOfMemoryError,
                    torch.AcceleratorError,
                ) as e:
                    err_str = str(e).lower()
                    is_oom = (
                        isinstance(
                            e, (torch.cuda.OutOfMemoryError, torch.AcceleratorError)
                        )
                        or "out of memory" in err_str
                        or "not enough memory" in err_str
                        or "defaultcpuallocator" in err_str
                    )
                    if is_oom:
                        logger.info(
                            f"OOM for trial {trial.number} at batch_size={bs}. Pruning."
                        )
                        mlflow.set_tag("pruned_reason", f"OOM at batch_size={bs}")
                        trial.set_user_attr("oom_batch_size", bs)
                        raise optuna.exceptions.TrialPruned() from None
                    raise
                finally:
                    # clean up after each trial to prevent OOM error
                    if tr_loader is not None:
                        tr_loader._iterator = None
                    if val_loader is not None:
                        val_loader._iterator = None
                    if obj_detect_model is not None:
                        obj_detect_model.cpu()
                        del obj_detect_model
                    if optimizer is not None:
                        del optimizer
                    del tr_loader, val_loader
                    gc.collect()
                    torch.cuda.empty_cache()

            return float(best_map50)

        logger.info("Starting Optuna HP search ...")
        with mlflow.start_run(run_name=f"{run_name_prefix}_{self.model_name}"):
            sampler = optuna.samplers.TPESampler(
                seed=self.cfg["seed"],
                n_startup_trials=self.cfg[
                    "num_startup_trials"
                ],  # number of random sample before TPE start
                multivariate=True,
            )
            # no pruner
            pruner = optuna.pruners.NopPruner()

            # maximise mAP:0.5
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )
            study.optimize(
                objective, n_trials=self.cfg["num_trials"], show_progress_bar=False
            )
            # check if any trial is completed
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                logger.info("HP search completed. No trials completed successfully.")
                return None

            # log hyperparameter with best metrics score
            best_params = study.best_params
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_map50", study.best_value)

        logger.info(
            f"HP search completed. Best params: {best_params}, "
            f"val_map50: {study.best_value}"
        )

        return best_params

    def train_full_set(self) -> Dict[str, Any]:
        """
        Train on full training set (train + val) and evaluate on test set.
        Log per epoch train loss in Mlflow and register the trained model at the end of training epochs to Mlflow registry.
        Set the test evaluation loss and metrics as tags in Mlflow.

        No early stopping in training since there is no validation set, so max epoch should be set from earlier
        hyperparameter tuning.

        Returns:
            Dictionary containing test set loss, mAP@0.5:0.95 and mAP@0.5.
        """

        if self.model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Registered: {self.available_models()}"
            )

        model_cls = self._MODEL_REGISTRY[self.model_name]
        train_ds, val_ds, test_ds = self.get_datasets()

        # Merge train and validation sets for training
        full_train_ds = train_ds + val_ds

        # config
        max_epochs = self.cfg["max_epochs"]
        num_classes = self.cfg["num_classes"]
        lr = self.cfg["best_params"]["learning_rate"]
        wd = self.cfg["best_params"]["weight_decay"]
        bs = self.cfg["best_params"]["batch_size"]

        # model and dataloader
        tr_loader = DataLoader(
            full_train_ds,
            batch_size=int(bs),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=int(bs),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=False,
        )

        obj_detect_model = model_cls(num_classes=num_classes).to(self.device)

        optimizer = AdamW(
            obj_detect_model.trainable_parameters(), lr=lr, weight_decay=wd
        )

        run_name_prefix = "train_full_set"
        run_name = f"{run_name_prefix}_{self.model_name}"

        # clear cache before training
        torch.cuda.empty_cache()

        logger.info(f"Doing train full set with model {self.model_name}")

        with mlflow.start_run(run_name=run_name) as run:
            artifact_uri = run.info.artifact_uri
            logger.info(f"Artifact URI: {artifact_uri}")

            fp = Params(
                learning_rate=lr,
                weight_decay=wd,
                batch_size=bs,
            )
            mlflow.log_params(fp.as_dict())

            try:
                for epoch in tqdm(
                    range(1, max_epochs + 1), desc="Training", unit="epoch"
                ):
                    # just training only, no validation set
                    tr_loss, _, _ = self._run_one_epoch(
                        obj_detect_model, tr_loader, optimizer, is_train=True
                    )

                    em = EpochMetrics(
                        epoch=epoch,
                        train_loss=tr_loss,
                        val_loss=np.nan,
                        val_map=np.nan,
                        val_map_50=np.nan,
                    )
                    mlflow.log_metrics(em.as_dict(), step=epoch)

            except (
                RuntimeError,
                torch.cuda.OutOfMemoryError,
                torch.AcceleratorError,
            ) as e:
                err_str = str(e).lower()
                is_oom = (
                    isinstance(e, (torch.cuda.OutOfMemoryError, torch.AcceleratorError))
                    or "out of memory" in err_str
                    or "not enough memory" in err_str
                    or "defaultcpuallocator" in err_str
                )
                if is_oom:
                    logger.info("Out of memory while training on full train set.")
                    mlflow.set_tag("failure reason", "OOM")
                raise

            # log model to mlflow
            # provide all code needed to run the model for inference
            mlflow.pytorch.log_model(
                obj_detect_model,
                artifact_path="model",  # path to store the weights in artifact folder
                registered_model_name=self.model_name,  # name to register the model under
                code_paths=["src/"],
            )

            logger.info(
                "Training completed. Model weights registered as MLflow artifact"
            )

            # do evaluation on test set using trained moddel and log the results
            logger.info("Evaluating trained model on test set")

            test_loss, test_map, test_map_50 = self._run_one_epoch(
                obj_detect_model, test_loader, optimizer=None, is_train=False
            )

            test_logs = {
                "test_loss": str(test_loss),
                "test_map": str(test_map),
                "test_map_50": str(test_map_50),
            }

            mlflow.set_tags(test_logs)
            logger.info(
                f"Test set logs added: test_loss: {test_loss}, test_map: {test_map}, test_map_50: {test_map_50}"
            )

            return test_logs
