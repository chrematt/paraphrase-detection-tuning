# Be careful, don't run this file on a MacBook. It will make your computer freeze.

import os

import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer

# Load environment variables
load_dotenv()

# Environment variables for Weights and Biases
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Seed for reproducibility
seed_everything(42)

# Hyperparameter tuning configuration using Ray Tune
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "weight_decay": tune.choice([0.0, 0.01, 0.001, 0.0001]),
    "optimizer_type": tune.choice(['adamw', 'adam', 'sgd']),
    "lr_scheduler": tune.choice(['linear_warmup', 'step_decay', 'cosine_annealing', 'exponential_decay']),
    "num_epochs_per_decay": tune.randint(1, 20),
    "decay_factor": tune.uniform(0.5, 1.0),
    "num_epochs_till_restart": tune.randint(1, 20),
}


def run(config):
    # Initialize Weights and Biases if API key is available
    if WANDB_API_KEY is not None:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=config)
        logger = WandbLogger()
    else:
        logger = None

    # Data Module
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
    )
    dm.setup("fit")

    # Model
    model = GLUETransformer(
        model_name_or_path='distilbert-base-uncased',
        num_labels=2,
        task_name='mrpc',
        learning_rate=config['learning_rate'],
        adam_epsilon=config.get('adam_epsilon', 1e-8),
        warmup_steps=config.get('warmup_steps', 0),
        weight_decay=config['weight_decay'],
        train_batch_size=config.get('train_batch_size', 32),
        eval_batch_size=config.get('eval_batch_size', 32),
        optimizer_type=config['optimizer_type'],
        lr_scheduler=config['lr_scheduler'],
        num_epochs_per_decay=config['num_epochs_per_decay'],
        decay_factor=config['decay_factor'],
        num_epochs_till_restart=config['num_epochs_till_restart'],
    )

    # Trainer with Ray Tune integration
    trainer = Trainer(
        logger=logger,
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")]
    )

    # Train the model
    trainer.fit(model, datamodule=dm)

    # Finish the Weights and Biases run
    if WANDB_PROJECT_NAME is not None and WANDB_ENTITY is not None:
        wandb.finish()


if __name__ == "__main__":
    # Start the Ray Tune run
    analysis = tune.run(
        run,
        config=config,
        num_samples=10,
        resources_per_trial={
            "cpu": 12,  # Adjust this based on your system's capabilities
            "gpu": 1    # Set to 0 if not using NVIDIA GPUs
        }
    )
