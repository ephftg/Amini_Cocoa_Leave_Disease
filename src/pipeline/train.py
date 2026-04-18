import importlib
import logging
from typing import Dict, List, Type, Tuple, Any, Optional
import pandas as pd
from omegaconf import DictConfig
import torch
from pathlib import Path
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import gc
import resource

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


def collate_fn(batch):
    return tuple(zip(*batch))


class Train:
    """
    Train class that have a model registry of models available
    """

    # store all model using key value pair
    _MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

    # allow each model class to self register to the model registry
    @classmethod
    def register(cls, name: str):
        """
        Class decorator that registers a model under *name*.
        """

        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._MODEL_REGISTRY:
                logger.info(
                    f"Model {name} is already registered, overwriting with {(model_cls.__name__,)}"
                )
            cls._MODEL_REGISTRY[name] = model_cls
            logger.info(f"Model {model_cls.__name__} is registerd under {name}")
            return model_cls  # always return the class unchanged

        return decorator

    @classmethod
    def available_models(cls) -> List[str]:
        """Return a list of registered model names."""
        return cls._MODEL_REGISTRY

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        """
        Return the model class for *name*, with a clear error if missing.

        Raises
        ------
        KeyError
            If *name* is not in the registry. The error message lists all
            registered names so the caller knows what's available.
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
        Import modules listed in *module_paths* so their @Train.register
        decorators fire.  Call this once at startup if you keep third-party
        or project-specific models in separate files.

        Train.load_model_plugins(cfg.get("model_plugins", []))
        """
        for path in module_paths:
            importlib.import_module(path)
            logger.info(f"Loaded model: {path}")

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device for training: {self.device}")

        # MLflow setup
        mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        # Data to be assigned later
        self._train_df = None
        self._test_df = None
        self._label_map = None

        # model under training
        self.model_name = cfg["model_name"]

    def get_datasets(self) -> Tuple[DetectionDataset, DetectionDataset]:
        """
        Get the train and test datasets.

        Parameters
        ----------
        model_name : str
            Key in the model registry (e.g. ``"FasterRCNN"``).

        Returns
        -------
        train_dataset, test_dataset : Tuple[DetectionDataset, DetectionDataset]
        """

        # read bounding box csv
        df = pd.read_csv(self.cfg["data"]["bbox_csv"])

        #  get train and test dataframe
        train_df, test_df = train_test_split(
            df, test_size=self.cfg["test_size"], seed=self.cfg["seed"]
        )

        self._train_df = train_df
        self._test_df = test_df

        # apply the processing to the data to get datasets
        train_ds = DetectionDataset(
            train_df, Path(self.cfg["data"]["image_dir"]), self.model_name
        )
        test_ds = DetectionDataset(
            test_df, Path(self.cfg["data"]["image_dir"]), self.model_name
        )

        # Sync label map
        self._label_map = train_ds.label_map

        logger.info("Train and test datasets obtained.")

        return train_ds, test_ds

    def get_kfold_splits(
        self, dataset: DetectionDataset
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Split the Dataset to k folds such that the class distribution is roughly maintained across all splits.

        Returns
        -------
        splits : List[Tuple[List[int], List[int]]]
            List of ``(train_indices, val_indices)`` pairs.
        """

        df = dataset.df
        image_names = dataset.image_names  # sorted list

        all_labels = sorted(df["class"].unique())

        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        mlb_matrix = np.zeros((len(image_names), len(all_labels)), dtype=int)
        for i, name in enumerate(image_names):
            for lbl in df[df["Image_ID"] == name]["class"]:
                mlb_matrix[i, label_to_idx[lbl]] = 1

        mskf = MultilabelStratifiedKFold(
            n_splits=self.cfg["kfold"], shuffle=True, random_state=self.cfg["seed"]
        )

        # split at image level using image name
        splits = [
            (list(ti), list(vi))
            for ti, vi in mskf.split(np.arange(len(image_names)), mlb_matrix)
        ]

        #  a list of (train_indices, val_indices) tuples, where indices are for image_names
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
        Run one training or evaluation epoch.

        Returns
        -------
        mean_loss : float
        map_score : float  (mAP@0.5:0.95)
        """
        # if is_train = True, equivalent to model.train()
        # if is_train = False, equivalent to model.eval()
        model.train(is_train)

        total_loss = 0.0  # loss over all batches

        # bbox for object detection
        metric = MeanAveragePrecision(iou_type="bbox")

        device_type = self.device.type
        use_amp = device_type == "cuda"

        logger.info(f"use amp: {use_amp}")

        # scaler = torch.amp.GradScaler(enabled=use_amp)

        phase = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=phase, unit="batch", leave=False)

        log_interval = 100
        total_batches = len(loader)

        start_time = time.perf_counter()

        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # use FP16 for forward pass
            with torch.autocast(
                device_type=device_type, enabled=use_amp, dtype=torch.bfloat16
            ):
                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                    loss_dict.clear()

            # in FP32 for precision
            if is_train:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                del loss
            else:
                with torch.no_grad():
                    # use FP16 for forward pass
                    with torch.autocast(
                        device_type=device_type, enabled=use_amp, dtype=torch.bfloat16
                    ):
                        # get loss with targets provided
                        model.train()  # set to train to get loss since in eval mode no loss
                        loss_dict = model(images, targets)
                        total_loss += sum(loss_dict.values()).item()
                        loss_dict.clear()

                        # get predictions with no targets to calculate metric
                        model.eval()  # switch back to eval mode
                        preds = model(images)

                    # move to cpu
                    preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
                    targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                    metric.update(preds_cpu, targets_cpu)
                    del preds, preds_cpu, targets_cpu

            # log dataloader batch progress
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"[{phase}] Batch {batch_idx + 1}/{total_batches} completed"
                )

            # remove from VRAM to free memory to prevent OOM in docker container
            del images, targets
            torch.cuda.empty_cache()

        mean_loss = total_loss / max(len(loader), 1)
        elapsed = time.perf_counter() - start_time

        # set to 0 when in train mode, since training metric is unstable and need double calculation
        # "map" value is mAP @ IoU 0.50:0.95
        # "map_50" is mAP @ IoU 0.50
        if not is_train:
            metric_results = metric.compute()
            map_score = metric_results["map"].item()
            map_50_score = metric_results["map_50"].item()
            metric.reset()  # reest to free internal state
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
        Build the HP search space from config ranges.

        The trial object is Optuna's context carrier, record sampled values and result.

        Each objective call gets a different trial, which will use get_hp_search_space to sampled values from this space
        guided by TPE learning from previous trials.

        """
        space_cfg = self.cfg["hyperparameters"]
        params = {}
        for name, spec in space_cfg.items():
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif (
                isinstance(spec["low"], int)
                and isinstance(spec["high"], int)
                and "log" not in spec
            ):
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
        Bayesian hyper-parameter search (TPE via Optuna) over K-fold CV using objective function.
        """
        if self.model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Registered: {self.available_models()}"
            )

        model_cls = self._MODEL_REGISTRY[self.model_name]
        train_ds, _ = self.get_datasets()  # only train set used for hyperparameter tune
        splits = self.get_kfold_splits(train_ds)
        max_epochs = self.cfg["max_epochs"]
        num_classes = self.cfg["num_classes"]  # number of object classes

        run_name_prefix = "hparam_search"

        logger.info(f"Doing hyperparameter tuning with model {self.model_name}")

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for optuna.
            Returns the validation loss as the metric to optimize for search.
            Includes clearing of cache to reduce RAM usage to prevent OOM errors.

            """
            # clear cache before running a trial
            torch.cuda.empty_cache()
            gc.collect()

            # get the hyperparameter for each trial
            params = self.get_hp_search_space(trial)
            lr = params["learning_rate"]
            wd = params["weight_decay"]
            bs = params["batch_size"]

            # track loss per fold
            fold_val_losses = []

            logger.info(f"Start kfold evaluation for trial {trial.number}")
            for fold_idx, (tr_idx, val_idx) in enumerate(splits):
                run_name = f"{run_name_prefix}_{self.model_name}_trial{trial.number}_fold{fold_idx}"

                obj_detect_model = None
                optimizer = None
                tr_loader = None
                val_loader = None

                # nested ensure outer parent run continue, new run added as a child run
                with mlflow.start_run(run_name=run_name, nested=True):
                    # log the parameters of this fol
                    fp = Params(
                        learning_rate=lr,
                        weight_decay=wd,
                        batch_size=bs,
                    )
                    # log fold params
                    mlflow.log_params(fp.as_dict())

                    try:
                        # collate_fn for object detection which expects image, target separated
                        # speed up transfer of data to GPU with number of workers and pin_memory
                        tr_loader = DataLoader(
                            Subset(train_ds, tr_idx),
                            batch_size=int(bs),
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            persistent_workers=False,
                        )

                        # use lower number of workers since val set is smaller
                        val_loader = DataLoader(
                            Subset(train_ds, val_idx),
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

                            em = EpochMetrics(
                                epoch=epoch,
                                train_loss=tr_loss,
                                val_loss=val_loss,
                                val_map=val_map,
                                val_map_50=val_map_50,
                            )
                            mlflow.log_metrics(em.as_dict(), step=epoch)

                            if early_stop(val_loss):
                                logger.info(
                                    f"Early stopping at epoch {epoch} for fold {fold_idx}"
                                )
                                break

                        fold_val_losses.append(early_stop.best_loss)

                        # Report intermediate fold result so Optuna can prune bad trials
                        trial.report(float(np.mean(fold_val_losses)), step=fold_idx)
                        pruned = trial.should_prune()

                        # Tag the fold run with its pruning outcome
                        mlflow.set_tag("pruned_after_fold", pruned)

                        if pruned:
                            raise optuna.exceptions.TrialPruned()

                    # catch any memory issue in training loop
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
                                f"Out of memory for trial: {trial.number}, fold:{fold_idx} "
                                f"with batch_size={bs}. Pruning trial."
                            )
                            mlflow.set_tag("pruned_reason", f"OOM at batch_size={bs}")
                            trial.set_user_attr("oom_batch_size", bs)
                            raise optuna.exceptions.TrialPruned() from None  # ignore original exception context
                        raise  # other time of error
                    # clean up after each fold to free up memory for next fold
                    finally:
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

                        rss_mb = (
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                        )
                        logger.info(
                            f"Cleared cache after fold {fold_idx}. RSS RAM: {rss_mb:.0f} MB"
                        )

                        logger.info("Cleared cache and memory after one fold.")

                logger.info(f"Completed fold {fold_idx}.")

            # mean validation loss across all folds for this trial if no OOM occured
            return float(np.mean(fold_val_losses))

        logger.info("Starting Optuna HP search ...")
        with mlflow.start_run(run_name=f"{run_name_prefix}_{self.model_name}"):
            sampler = optuna.samplers.TPESampler(
                seed=self.cfg["seed"],
                n_startup_trials=self.cfg["num_startup_trials"],
                multivariate=True,
            )
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=self.cfg["kfold"] // 2)

            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )
            study.optimize(
                objective, n_trials=self.cfg["num_trials"], show_progress_bar=False
            )

            # return None if non of the trials completed
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if len(completed_trials) == 0:
                logger.info("HP search completed. No trials completed successfully.")
                return None

            best_params = study.best_params
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_loss", study.best_value)

        logger.info(
            f"HP search completed. Best params: {best_params}, val_loss: {study.best_value}"
        )
        return best_params

    def train_full_set(
        self,
    ) -> Dict[str, Any]:
        """
        Train on full training set and do evaluation on test set.
        Register the trained model in mlflow.
        Output the evaluation metric of the test set.
        """

        if self.model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Registered: {self.available_models()}"
            )

        model_cls = self._MODEL_REGISTRY[self.model_name]
        train_ds, test_ds = self.get_datasets()

        # config
        max_epochs = self.cfg["max_epochs"]
        num_classes = self.cfg["num_classes"]  # number of object classes
        lr = self.cfg["best_params"]["learning_rate"]
        wd = self.cfg["best_params"]["weight_decay"]
        bs = self.cfg["best_params"]["batch_size"]

        # model and dataloader
        tr_loader = DataLoader(
            train_ds,
            batch_size=int(bs),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=int(bs),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        obj_detect_model = model_cls(num_classes=num_classes).to(self.device)

        optimizer = AdamW(
            obj_detect_model.trainable_parameters(), lr=lr, weight_decay=wd
        )

        early_stop = EarlyStopping(
            patience=self.cfg["early_stopping_patience"],
            min_delta=self.cfg["early_stopping_min_delta"],
        )

        run_name_prefix = "train_full_set"
        run_name = f"{run_name_prefix}_{self.model_name}"

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
            # log params
            mlflow.log_params(fp.as_dict())

            try:
                for epoch in tqdm(
                    range(1, max_epochs + 1), desc="Training", unit="epoch"
                ):
                    tr_loss, _, _ = self._run_one_epoch(
                        obj_detect_model, tr_loader, optimizer, is_train=True
                    )

                    # no validation
                    em = EpochMetrics(
                        epoch=epoch,
                        train_loss=tr_loss,
                        val_loss=np.nan,
                        val_map=np.nan,
                        val_map_50=np.nan,
                    )
                    mlflow.log_metrics(em.as_dict(), step=epoch)

                    # use train loss as early stop
                    if early_stop(tr_loss):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            # catch any memory issue in training loop
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
                raise  # other time of error

            # MLflow Model Registry support different version of the model under the same model name

            # add model name in model registry
            # to change alias, use code or UI
            # code paths include all script needed to load the model
            # artifact will include model weights, src, virtual enviroment requirements
            mlflow.pytorch.log_model(
                obj_detect_model,
                artifact_path="model",  #  folder to store weights
                registered_model_name=self.model_name,
                code_paths=["src/"],
            )

            logger.info(
                "Training completed. Model weights registered as MLflow artifact"
            )

            # apply trained model to test set to get loss and metric
            logger.info("Evaluating trained model on test set")

            test_loss, test_map, test_map_50 = self._run_one_epoch(
                obj_detect_model, test_loader, optimizer=None, is_train=False
            )

            # record as tag , can use log_metrics
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
