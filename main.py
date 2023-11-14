import os
from argparse import ArgumentParser

import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer

load_dotenv()

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

seed_everything(42)


def run(hparams):
    logger = None

    if WANDB_API_KEY is not None:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=hparams)
        logger = WandbLogger()

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
    )

    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path='distilbert-base-uncased',
        num_labels=2,
        task_name='mrpc',
        learning_rate=hparams['learning_rate'],
        adam_epsilon=hparams['adam_epsilon'],
        warmup_steps=hparams['warmup_steps'],
        weight_decay=hparams['weight_decay'],
        train_batch_size=hparams['train_batch_size'],
        eval_batch_size=hparams['eval_batch_size'],
        optimizer_type=hparams['optimizer_type'],
        lr_scheduler=hparams['lr_scheduler'],
        num_epochs_per_decay=hparams['num_epochs_per_decay'],
        decay_factor=hparams['decay_factor'],
        num_epochs_till_restart=hparams['num_epochs_till_restart'],
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, datamodule=dm)

    if WANDB_PROJECT_NAME is not None and WANDB_ENTITY is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--learning_rate", type=float, default=0.00004514343314938338)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--lr_scheduler", type=str, default="step_decay")
    parser.add_argument("--optimizer_type", type=str, default="adam")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_epochs_per_decay", type=int, default=6)
    parser.add_argument("--decay_factor", type=float, default=0.8240985559720105)
    parser.add_argument("--num_epochs_till_restart", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_splits", type=str, default=None)
    args = parser.parse_args()

    hparams = vars(args)
    print(hparams)

    run(hparams)
