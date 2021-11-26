#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    #Trainer,
    #TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in ``torch.nn.DistributedDataParallel``. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to :obj:`False` if model parallel or deepspeed is used, or if the default
          ``TrainingArguments.place_model_on_device`` is overridden to return :obj:`False` .
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)

    """

    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. "
                    "`model_init` will overwrite your model when calling the `train` method. This will become a fatal error in the next release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # Setup Sharded DDP training
        self.sharded_ddp = None
        if len(args.sharded_ddp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )

            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif not is_fairscale_available():
                raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
            elif ShardedDDPOption.SIMPLE not in args.sharded_ddp and FullyShardedDDP is None:
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3

        # one place to sort out whether to place the model on device or not
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or (args.deepspeed and args.do_train)
            or (args.fp16_full_eval and not args.do_train)
            or (self.sharded_ddp in [ShardedDDPOption.ZERO_DP_2, ShardedDDPOption.ZERO_DP_3])
        ):
            self.place_model_on_device = False

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        if self.place_model_on_device:
            model = model.to(args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self._signature_columns = None
        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None

        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16 and not args.deepspeed:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = ShardedGradScaler() if self.sharded_ddp is not None else torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable for total_flos used to count as tensors (for distributed + TPU), will be sent in the
        # state at each call to self.log.
        self._total_flos = None
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions", "end_positions"]
            if type(self.model).__name__ in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

    def add_callback(self, callback):
        """
        Add a callback to the current list of :class:`~transformer.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of :class:`~transformer.TrainerCallback` and returns it.

        If the callback is not found, returns :obj:`None` (and no error is raised).

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            :class:`~transformer.TrainerCallback`: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of :class:`~transformer.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        dataset.set_format(type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"])

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, model_input_name=model_input_name
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    model_input_name=model_input_name,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset)
            elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                )
            else:
                return DistributedSampler(
                    self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
                )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.args.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            generator=g
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )

        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = (
            self.args.warmup_steps
            if self.args.warmup_steps > 0
            else math.ceil(num_training_steps * self.args.warmup_ratio)
        )

        return get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            self.optimizer = self.create_optimizer()

        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """ HP search setup code """
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return

        params = self.hp_space(trial) if self.hp_search_backend == HPSearchBackend.OPTUNA else trial
        for key, value in params.items():
            if not hasattr(self.args, key):
                raise AttributeError(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
                )
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info("Trial:", trial.params)

    def _report_to_hp_search(
        self, trial: Union["optuna.Trial", Dict[str, Any]], epoch: int, metrics: Dict[str, float]
    ):
        if self.hp_search_backend is None or trial is None:
            return
        self.objective = self.compute_objective(metrics.copy())
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            trial.report(self.objective, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            from ray import tune

            if self.control.should_save:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)

    def _tune_save_checkpoint(self):
        from ray import tune

        if not self.use_tune_checkpoints:
            return
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def call_model_init(self, trial=None):
        model_init_argcount = len(inspect.signature(self.model_init).parameters)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def _wrap_model(self, model, training=True):
        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)

        elif is_sagemaker_distributed_available():
            model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
        elif self.args.local_rank != -1:
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
            else:
                find_unused_parameters = True
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        return model

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            self._load_state_dict_in_model(state_dict)
            del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and self.args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += float(self.floating_point_ops(inputs))
                
                for name, param in model.named_children():
                    if 'attention' not in name and 'dense' in name and 'weight' in name:
                        tr_loss += self.args.glasso_param * torch.norm(param,dim=1).sum() / np.sqrt(param.shape[0])

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                                error_if_nonfinite=False
                            )

                    # Optimizer step
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # force memory release
            # to restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            else:
                from ray import tune

                run_id = tune.get_trial_id()
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)
        elif self.is_world_process_zero() and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if os.path.isfile(os.path.join(checkpoint, "optimizer.pt")) and os.path.isfile(
            os.path.join(checkpoint, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            if is_torch_tpu_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                optimizer_state = torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location="cpu")
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(os.path.join(checkpoint, "scheduler.pt"), map_location="cpu")
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location=self.args.device)
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))
                reissue_pt_warnings(caught_warnings)

        if self.deepspeed:
            # Not sure how to check if there is a saved deepspeed checkpoint, but since it just return None if it fails to find a deepspeed checkpoint this is sort of a check-n-load function
            self.deepspeed.load_checkpoint(checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True)

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        """
        Launch an hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no metric is
        provided, the sum of all metrics otherwise.

        .. warning::

            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

        Args:
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.trainer_utils.default_hp_space_optuna` or
                :func:`~transformers.trainer_utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:

                - the documentation of `optuna.create_study
                  <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
                - the documentation of `tune.run
                  <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__

        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
        """
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError(
                    "At least one of optuna or ray should be installed. "
                    "To install optuna run `pip install optuna`."
                    "To install ray run `pip install ray[tune]`."
                )
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
        if backend == HPSearchBackend.RAY and not is_ray_tune_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        run_hp_search = run_hp_search_optuna if backend == HPSearchBackend.OPTUNA else run_hp_search_ray
        best_run = run_hp_search(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or dist.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
        ):
            state_dict = self.model.state_dict()
            if self.is_world_process_zero():
                self._save(output_dir, state_dict=state_dict)
        elif self.is_world_process_zero():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        xm.rendezvous("saving_checkpoint")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    save_config=self.is_world_process_zero(),
                    state_dict=self.model.state_dict(),
                    save_function=xm.save,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = self.model.state_dict()
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, save_config=self.is_world_process_zero(), save_function=xm.save)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self._total_flos is not None:
            if self.args.local_rank != -1:
                self.state.total_flos = distributed_broadcast_scalars([self._total_flos]).sum().item()
            else:
                self.state.total_flos = self._total_flos

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_torch_tpu_available():
            tensors = nested_xla_mesh_reduce(tensors, name)
        elif self.args.local_rank != -1:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save):
                self.model.tie_weights()
            else:
                logger.warn(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__ arguments that can be specified on the command
    line.




    Parameters:
        output_dir (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_dir` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not. This argument is not directly used by :class:`~transformers.Trainer`, it's
            intended to be used by your training/evaluation scripts instead. See the `example scripts
            <https://github.com/huggingface/transformers/tree/master/examples>`__ for more details.
        do_eval (:obj:`bool`, `optional`):
            Whether to run evaluation on the validation set or not. Will be set to :obj:`True` if
            :obj:`evaluation_strategy` is different from :obj:`"no"`. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        evaluation_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No evaluation is done during training.
                * :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.
                * :obj:`"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            .. warning::

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every ``gradient_accumulation_steps * xxx_step`` training
                examples.
        eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for :class:`~transformers.AdamW` optimizer.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in
            :class:`~transformers.AdamW` optimizer.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon hyperparameter for the :class:`~transformers.AdamW` optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        lr_scheduler_type (:obj:`str` or :class:`~transformers.SchedulerType`, `optional`, defaults to :obj:`"linear"`):
            The scheduler type to use. See the documentation of :class:`~transformers.SchedulerType` for all possible
            values.
        warmup_ratio (:obj:`float`, `optional`, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to :obj:`learning_rate`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`. Overrides any effect of
            :obj:`warmup_ratio`.
        logging_dir (:obj:`str`, `optional`):
            `TensorBoard <https://www.tensorflow.org/tensorboard>`__ log directory. Will default to
            `runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The logging strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No logging is done during training.
                * :obj:`"epoch"`: Logging is done at the end of each epoch.
                * :obj:`"steps"`: Logging is done every :obj:`logging_steps`.

        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to log and evaluate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs if :obj:`logging_strategy="steps"`.
        save_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No save is done during training.
                * :obj:`"epoch"`: Save is done at the end of each epoch.
                * :obj:`"steps"`: Save is done every :obj:`save_steps`.

        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves if :obj:`save_strategy="steps"`.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            :func:`~transformers.Trainer.model_init` function to instantiate the model if it has some randomly
            initialized parameters.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `Apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        fp16_backend (:obj:`str`, `optional`, defaults to :obj:`"auto"`):
            The backend to use for mixed precision training. Must be one of :obj:`"auto"`, :obj:`"amp"` or
            :obj:`"apex"`. :obj:`"auto"` will use AMP or APEX depending on the PyTorch version detected, while the
            other choices will force the requested backend.
        fp16_full_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use full 16-bit precision evaluation instead of 32-bit. This will be faster and save memory but
            can harm metric values.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            Rank of the process during distributed training.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When training on TPU, whether to print debug metrics or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`):
            Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`. Will default to the
            same value as :obj:`logging_steps` if not set.
        dataloader_num_workers (:obj:`int`, `optional`, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        run_name (:obj:`str`, `optional`):
            A descriptor for the run. Typically used for `wandb <https://www.wandb.com/>`_ logging.
        disable_tqdm (:obj:`bool`, `optional`):
            Whether or not to disable the tqdm progress bars and table of metrics produced by
            :class:`~transformers.notebook.NotebookTrainingTracker` in Jupyter Notebooks. Will default to :obj:`True`
            if the logging level is set to warn or lower (default), :obj:`False` otherwise.
        remove_unused_columns (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If using :obj:`datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.

            (Note that this behavior is not implemented for :class:`~transformers.TFTrainer` yet.)
        label_names (:obj:`List[str]`, `optional`):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to :obj:`["labels"]` except if the model used is one of the
            :obj:`XxxForQuestionAnswering` in which case it will default to :obj:`["start_positions",
            "end_positions"]`.
        load_best_model_at_end (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to load the best model found during training at the end of training.

            .. note::

                When set to :obj:`True`, the parameters :obj:`save_strategy` and :obj:`save_steps` will be ignored and
                the model will be saved after each evaluation.
        metric_for_best_model (:obj:`str`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix :obj:`"eval_"`.
            Will default to :obj:`"loss"` if unspecified and :obj:`load_best_model_at_end=True` (to use the evaluation
            loss).

            If you set this value, :obj:`greater_is_better` will default to :obj:`True`. Don't forget to set it to
            :obj:`False` if your metric is better when lower.
        greater_is_better (:obj:`bool`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` and :obj:`metric_for_best_model` to specify if better
            models should have a greater metric or not. Will default to:

            - :obj:`True` if :obj:`metric_for_best_model` is set to a value that isn't :obj:`"loss"` or
              :obj:`"eval_loss"`.
            - :obj:`False` if :obj:`metric_for_best_model` is not set, or set to :obj:`"loss"` or :obj:`"eval_loss"`.
        ignore_skip_data (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to :obj:`True`, the training will begin faster (as that skipping
            step can take a long time) but will not yield the same results as the interrupted training would have.
        sharded_ddp (:obj:`bool`, :obj:`str` or list of :class:`~transformers.trainer_utils.ShardedDDPOption`, `optional`, defaults to :obj:`False`):
            Use Sharded DDP training from `FairScale <https://github.com/facebookresearch/fairscale>`__ (in distributed
            training only). This is an experimental feature.

            A list of options along the following:

            - :obj:`"simple"`: to use first instance of sharded DDP released by fairscale (:obj:`ShardedDDP`) similar
              to ZeRO-2.
            - :obj:`"zero_dp_2"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-2 mode (with :obj:`reshard_after_forward=False`).
            - :obj:`"zero_dp_3"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-3 mode (with :obj:`reshard_after_forward=True`).
            - :obj:`"offload"`: to add ZeRO-offload (only compatible with :obj:`"zero_dp_2"` and :obj:`"zero_dp_3"`).

            If a string is passed, it will be split on space. If a bool is passed, it will be converted to an empty
            list for :obj:`False` and :obj:`["simple"]` for :obj:`True`.
        deepspeed (:obj:`str`, `optional`):
            Use `Deepspeed <https://github.com/microsoft/deepspeed>`__. This is an experimental feature and its API may
            evolve in the future. The value is the location of its json config file (usually ``ds_config.json``).
        label_smoothing_factor (:obj:`float`, `optional`, defaults to 0.0):
            The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
            labels are changed from 0s and 1s to :obj:`label_smoothing_factor/num_labels` and :obj:`1 -
            label_smoothing_factor + label_smoothing_factor/num_labels` respectively.
        adafactor (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the :class:`~transformers.Adafactor` optimizer instead of
            :class:`~transformers.AdamW`.
        group_by_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group together samples of roughly the same legnth in the training dataset (to minimize
            padding applied and be more efficient). Only useful if applying dynamic padding.
        report_to (:obj:`str` or :obj:`List[str]`, `optional`, defaults to :obj:`"all"`):
            The list of integrations to report the results and logs to. Supported platforms are :obj:`"azure_ml"`,
            :obj:`"comet_ml"`, :obj:`"mlflow"`, :obj:`"tensorboard"` and :obj:`"wandb"`. Use :obj:`"all"` to report to
            all integrations installed, :obj:`"none"` for no integrations.
        ddp_find_unused_parameters (:obj:`bool`, `optional`):
            When using distributed training, the value of the flag :obj:`find_unused_parameters` passed to
            :obj:`DistributedDataParallel`. Will default to :obj:`False` if gradient checkpointing is used, :obj:`True`
            otherwise.
        dataloader_pin_memory (:obj:`bool`, `optional`, defaults to :obj:`True`)):
            Whether you want to pin memory in data loaders or not. Will default to :obj:`True`.
        skip_memory_metrics (:obj:`bool`, `optional`, defaults to :obj:`False`)):
            Whether to skip adding of memory profiler reports to metrics. Defaults to :obj:`False`.

    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=None, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
            "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    fp16_backend: str = field(
        default="auto",
        metadata={"help": "The backend to be used for mixed precision.", "choices": ["auto", "amp", "apex"]},
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    sharded_ddp: str = field(
        default="",
        metadata={
            "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
            "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
            "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
            "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json)"},
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "`DistributedDataParallel`."
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=False, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    _n_gpu: int = field(init=False, repr=False, default=-1)
    cls_dropout: Optional[float] = field(default=None, metadata={"help": "cls drop out."})
    use_deterministic_algorithms: Optional[bool] = field(default=False, metadata={"help": "Whether or not to use deterministic algorithms."})
    glasso_param: Optional[float] = field(default=0.3, metadata={"help": "Group Lasso parameter for Linear layers."})

    def __post_init__(self):
        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True
        if self.eval_steps is None:
            self.eval_steps = self.logging_steps

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if is_torch_available() and self.device.type != "cuda" and (self.fp16 or self.fp16_full_eval):
            raise ValueError(
                "Mixed precision training with AMP or APEX (`--fp16`) and FP16 evaluation can only be used on CUDA devices."
            )
        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from .integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        if isinstance(self.sharded_ddp, bool):
            self.sharded_ddp = "simple" if self.sharded_ddp else ""
        if isinstance(self.sharded_ddp, str):
            self.sharded_ddp = [ShardedDDPOption(s) for s in self.sharded_ddp.split()]
        if self.sharded_ddp == [ShardedDDPOption.OFFLOAD]:
            raise ValueError(
                "`--sharded_ddp offload` can't work on its own. It needs to be added to `--sharded_ddp zero_dp_2` or "
                '`--sharded_ddp zero_dp_3`. For example, `--sharded_ddp "zero_dp_2 offload"`.'
            )
        elif len(self.sharded_ddp) > 1 and ShardedDDPOption.SIMPLE in self.sharded_ddp:
            raise ValueError("`--sharded_ddp simple` is not compatible with any other option.")
        elif ShardedDDPOption.ZERO_DP_2 in self.sharded_ddp and ShardedDDPOption.ZERO_DP_3 in self.sharded_ddp:
            raise ValueError("`--sharded_ddp zero_dp_2` is not compatible with `--sharded_ddp zero_dp_3`.")

    def __repr__(self):
        # We override the default repr to remove deprecated arguments from the repr. This method should be removed once
        # those deprecated arguments are removed form TrainingArguments. (TODO: v5)
        self_as_dict = asdict(self)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]
        attrs_as_str = [f"{k}={v}" for k, v in self_as_dict.items()]
        return f"{self.__class__.__name__}({', '.join(attrs_as_str)})"

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_distributed_available():
            sm_dist.init_process_group()
            self.local_rank = sm_dist.get_local_rank()
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            from .integrations import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="gloo" if sys.platform == "win32" else "nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu

    @property
    @torch_required
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - :obj:`ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - :obj:`ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses :obj:`torch.nn.DataParallel`).
        - :obj:`ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          :obj:`torch.nn.DistributedDataParallel`).
        - :obj:`ParallelMode.TPU`: several TPU cores.
        """
        if is_torch_tpu_available():
            return ParallelMode.TPU
        elif is_sagemaker_distributed_available():
            return ParallelMode.SAGEMAKER_DISTRIBUTED
        elif self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    @torch_required
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if is_torch_tpu_available():
            return xm.xrt_world_size()
        elif is_sagemaker_distributed_available():
            return sm_dist.get_world_size()
        elif self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

    @property
    @torch_required
    def process_index(self):
        """
        The number of processes used in parallel.
        """
        if is_torch_tpu_available():
            return xm.get_ordinal()
        elif is_sagemaker_distributed_available():
            return sm_dist.get_rank()
        elif self.local_rank != -1:
            return torch.distributed.get_rank()
        return 0

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return True

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return not self.deepspeed

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_DISTRIBUTED = "sm_distributed"
    TPU = "tpu"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )

    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=training_args.cls_dropout,
        apply_lora=model_args.apply_lora,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    trainable_params = []
    if model_args.apply_lora:
        if model_args.lora_path is not None:
            lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            logger.info(lora_state_dict.keys())
            model.load_state_dict(lora_state_dict, strict=False)
        trainable_params.append('lora')

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
            assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
        trainable_params.append('adapter')

    if model_args.apply_bitfit:
        trainable_params.append('bias')

    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta'):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
