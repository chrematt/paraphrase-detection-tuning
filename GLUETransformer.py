from datetime import datetime
from typing import Optional

import datasets
import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class GLUETransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            task_name: str,
            optimizer_type: str = 'adamw',
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            lr_scheduler: str = 'linear_warmup',
            num_epochs_per_decay: int = 10,
            decay_factor: float = 0.9,
            num_epochs_till_restart: int = 10,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def train_dataloader(self):
        # Assuming GLUE dataset is loaded using the `datasets` library
        glue_dataset = datasets.load_dataset("glue", self.hparams.task_name)
        return DataLoader(glue_dataset, batch_size=self.hparams.train_batch_size, shuffle=True)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Configure optimizer depending on the optimizer type
        if self.hparams.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                          eps=self.hparams.adam_epsilon)
        elif self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                         eps=self.hparams.adam_epsilon)
        elif self.hparams.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Optimizer type '{self.hparams.optimizer_type}' not recognized")

        # Define LR scheduler based on optimizer
        if self.hparams.lr_scheduler == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_scheduler == 'step_decay':
            step_size = self.hparams.num_epochs_per_decay * len(self.train_dataloader())  # Convert epochs to steps
            scheduler = StepLR(optimizer, step_size=step_size, gamma=self.hparams.decay_factor)
        elif self.hparams.lr_scheduler == 'cosine_annealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs_till_restart)
        elif self.hparams.lr_scheduler == 'exponential_decay':
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.decay_factor)
        else:
            raise ValueError(f"Scheduler type '{self.hparams.lr_scheduler}' not recognized")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
