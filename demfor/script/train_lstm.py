from demfor.data.load import Dataset
from demfor.models.lstm import get_LSTM
from demfor.utils.metrics import keras_SMAPE
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import mlflow
import mlflow.sklearn

import os

if __name__ == '__main__':
    SAVE_DIR = 'models/LSTM/'

    list_exp = mlflow.tracking.list_experiments()

    with mlflow.start_run(experiment_id=1):
        active_run = mlflow.active_run()

        dataset = Dataset()
        X_train, Y_train, X_val, Y_val, X_test = dataset.get_split_in_year_time_series()

        model = get_LSTM(X_train)

        # Parameters
        loss = "mae"
        mlflow.log_param("loss", loss)

        optimizer = "adam"
        mlflow.log_param("optimizer", optimizer)


        mlflow.log_artifact
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=3,
                                       verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir=SAVE_DIR, histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(SAVE_DIR + "model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)

        model.compile(loss='mae', optimizer='adam', metrics=["mse", keras_SMAPE])
        model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, steps_per_epoch=None,
                  callbacks=[early_stopping, tensorboard, checkpoint],
                  validation_data=(X_val, Y_val))
