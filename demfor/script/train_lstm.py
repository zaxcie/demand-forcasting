from demfor.data.load import Dataset
from demfor.models.lstm import get_LSTM
from demfor.utils.metrics import keras_SMAPE
from demfor.utils.mlflow import find_or_create_experiment

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, History

import mlflow

import json
import os

if __name__ == '__main__':
    exp = find_or_create_experiment("LSTM", mlflow.tracking.list_experiments())
    if isinstance(exp, list):
        raise TypeError("Multiple experiment with that name where found.")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        active_run = mlflow.active_run()

        dataset = Dataset()
        X_train, Y_train, X_val, Y_val, X_test = dataset.get_split_in_year_time_series()

        model = get_LSTM(X_train)

        # Parameters
        loss = "mae"
        mlflow.log_param("loss", loss)

        optimizer = "adam"
        mlflow.log_param("optimizer", optimizer)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=3,
                                       verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir=active_run.info.artifact_uri, histogram_freq=0,
                                  write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(active_run.info.artifact_uri + "/model.h5",
                                     monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        history = History()

        with open(active_run.info.artifact_uri + "/network_architecture.json", "w") as f:
            json.dump(model.to_json(), f)

        model.compile(loss=loss, optimizer=optimizer, metrics=["mse", keras_SMAPE])
        model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, steps_per_epoch=2,
                  callbacks=[early_stopping, tensorboard, checkpoint],
                  validation_data=(X_val, Y_val))

        print(history.history)

