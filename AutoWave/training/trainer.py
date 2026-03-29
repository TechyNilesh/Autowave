"""HuggingFace Trainer wrapper with smart defaults for audio classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def _compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 for HuggingFace Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


class AutoWaveTrainer:
    """Wraps HuggingFace Trainer with smart defaults for audio classification.

    Handles: mixed precision (fp16/bf16) when GPU available, learning rate
    warmup, early stopping, and per-epoch evaluation.

    Args:
        output_dir: Directory to save checkpoints and final model.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Initial learning rate.
        use_fp16: Enable fp16 mixed precision on CUDA. Auto-detected if None.
    """

    def __init__(
        self,
        output_dir: str | Path = "autowave_output",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        use_fp16: bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._use_fp16 = use_fp16

    def _resolve_fp16(self) -> bool:
        import torch
        if self._use_fp16 is not None:
            return self._use_fp16
        return torch.cuda.is_available()

    def train(
        self,
        model,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        data_collator=None,
    ):
        """Train the model using HuggingFace Trainer.

        Args:
            model: HuggingFace model (e.g. TransformerModel.hf_model).
            train_dataset: PyTorch Dataset for training.
            val_dataset: Optional PyTorch Dataset for validation.
            data_collator: Optional HF data collator.

        Returns:
            HuggingFace TrainOutput namedtuple.
        """
        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
        from transformers import DefaultDataCollator

        collator = data_collator or DefaultDataCollator()
        eval_strategy = "epoch" if val_dataset is not None else "no"
        load_best = val_dataset is not None
        fp16 = self._resolve_fp16()

        warmup_steps = max(1, int(0.1 * len(train_dataset) / self.batch_size * self.epochs))

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_steps=warmup_steps,
            fp16=fp16,
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            load_best_model_at_end=load_best,
            metric_for_best_model="accuracy" if load_best else None,
            logging_steps=10,
            save_total_limit=2,
            report_to="none",
            dataloader_num_workers=0,
        )

        callbacks = []
        if val_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics,
            data_collator=collator,
            callbacks=callbacks if callbacks else None,
        )

        return trainer.train()
