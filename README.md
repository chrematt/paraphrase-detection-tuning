# Paraphrase-Detection-Tuning

Hyperparameter Tuning for Paraphrase Detection on MRPC with DistilBERT.

The aim of this project was to fine-tune a DistilBERT model on the MRPC dataset to improve the validation accuracy through hyperparameter tuning. Various hyperparameters were explored in a structured manner.

## How to run single training?

*Optional*: Update the '.env' file with your weights & biases credentials.

### Docker
1. Build the docker image: `docker build -t paraphrase-detection-tuning .`
2. Run the docker image: `docker run -it paraphrase-detection-tuning`

### Locally
1. Install the requirements: `pip install -r requirements.txt`
2. Run the script: `python3 main.py`

## How to run multiple trainings?

1. Install the requirements: `pip install -r requirements.txt`
2. Configure the 'raytune.py' script

There you find a section with the configs:
```python
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "weight_decay": tune.choice([0.0, 0.01, 0.001, 0.0001]),
    "optimizer_type": tune.choice(['adamw', 'adam', 'sgd']),
    "lr_scheduler": tune.choice(['linear_warmup', 'step_decay', 'cosine_annealing', 'exponential_decay']),
    "num_epochs_per_decay": tune.randint(1, 20),
    "decay_factor": tune.uniform(0.5, 1.0),
    "num_epochs_till_restart": tune.randint(1, 20)
```
They are used for sampling the hyperparameters. You can add more or remove some. The values are sampled from the given range. For more information, see the [Ray Tune Documentation](https://docs.ray.io/en/master/tune/index.html).

Configure the number of samples and the underling hardware by defining the amount of GPUs and CPUs each run is going to acquire. Be careful with your resource, as the system will freeze when you assign too much. For more information, see the [Ray Tune Documentation](https://docs.ray.io/en/master/tune/index.html).

```python
analysis = tune.run(
        run,
        config=config,
        num_samples=10,
        resources_per_trial={
            "cpu": 1,  # Adjust this based on your system's capabilities
            "gpu": 1    # Set to 0 if not using NVIDIA GPUs
        }
    )
```

Finally, start the script: `python raytune.py`

## Best hyperparameters
The following hyperparameters are the best ones found by the tuning process. They are set as default in the 'main.py' script.

```
learning_rate=0.011044351
weight_decay=0.01
train_batch_size=32
lr_scheduler="step_decay"
optimizer_type="sgd"
adam_epsilon=1e-8
warmup_steps=0
num_epochs_per_decay=16
decay_factor=0.848138679
num_epochs_till_restart=13
eval_batch_size=32
eval_splits=None

accuracy=0.855392158
f1=0.897033155
val_loss=0.375731558
```

## Notes
I initially used torch version 2.1 which led to a error when using it within docker on my MacBook with Apple Silicon. Using torch version 2.0.1 fixed the issue. I did not test it on other systems.