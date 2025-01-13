#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

from tensorflow import keras
import clize
import tensorflow as tf
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from aiproteomics.modelgen.keras_transformer import create_transformer, create_baseline_model
from aiproteomics.tfrecords import get_dataset, BATCH_SIZE

MODEL_DIR = "./data"

TRAIN_SIZE = 0.7
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


@clize.parser.value_converter
class ModelType(Enum):
    TRANSFORMER = "transformer"
    FULLY_CONNECTED = "fully_connected"


def run_experiment(
        model,
        train_data_files,
        validation_data_files,
        test_data_files,
        num_epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        distributed=False
):
    # Make sure model dir exists
    Path(MODEL_DIR).mkdir(exist_ok=True)
    # Set up wandb

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="aiproteomics",

        # track hyperparameters and run metadata
        config={
            "model": model.name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
    )

    train_dataset = get_dataset(train_data_files)
    validation_dataset = get_dataset(validation_data_files)

    print("Start training the model...")
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset,
        callbacks=get_callbacks(MODEL_DIR),

    )
    print("Model training finished")

    _, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history


def train(data_dir, *, distributed: bool = False,
          model_type=ModelType.TRANSFORMER,
          train_size=TRAIN_SIZE,
          file_limit: int = None,
          epochs: int = NUM_EPOCHS,
          batch_size: int = BATCH_SIZE):
    data_dir = Path(data_dir)
    print(f"Training model with data from {data_dir} and a train split of {train_size}.")

    files = list(data_dir.glob("*.tfrecord"))

    if file_limit:
        files = files[:file_limit]

    print(f"Number of files: {len(files)}")

    train_files, remainder = train_test_split(files, train_size=train_size)
    val_files, test_files = train_test_split(remainder, train_size=train_size)

    if distributed:
        print("Training in distributed mode.")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = build_model(model_type)

        run_experiment(model, train_files, val_files, test_files, epochs, batch_size)
    else:
        model = build_model(model_type)

        run_experiment(model, train_data_files=train_files,
                       validation_data_files=val_files,
                       test_data_files=test_files,
                       num_epochs=epochs,
                       batch_size=batch_size)


def build_model(model_type):
    # Build model
    match model_type:
        case ModelType.TRANSFORMER:
            model = create_transformer()
        case ModelType.FULLY_CONNECTED:
            model = create_baseline_model()
        case _:
            raise ValueError(f"Model type {model_type} not supported.")
    return model


def get_callbacks(model_dir_path):
    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}.hdf5".format(
        model_dir_path, epoch_format, loss_format
    )
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True,
                                           save_weights_only=True)

    logger = WandbMetricsLogger(log_freq=5)
    wandb_checkpoint = WandbModelCheckpoint("models")
    return [save, logger, wandb_checkpoint]


if __name__ == "__main__":
    clize.run(train)
