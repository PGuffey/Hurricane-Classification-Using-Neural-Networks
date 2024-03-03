import keras
from keras import layers
from keras.models import load_model

class ConvLSTM():
    #Initialize all variables
    def __init__(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.buildModel()

    def buildModel(self):
        inp = layers.Input(shape=(None, *self.x_train.shape[2:]))

        # Construct 3 `ConvLSTM2D` layers with decreasing kernel size and batch normalization,
        # followed by a `Conv3D` layer for creating the final outputs.
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            data_format='channels_last', #Needed as dataset.shape has alpha channel last
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            data_format='channels_last',
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            data_format='channels_last',
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        )(x)

        # Build the complete model and compile it.
        model = keras.models.Model(inp, x)
        model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.legacy.Adam(),
        )

        return model

    def fit(self):
        #Callbacks to avoid LR causing plateu in learning as well as early stopping to avoid loss increasing in late epochs
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        # Define training hyperparameters.
        epochs = self.epochs
        batch_size = self.batch_size

        fitted = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=[early_stopping, reduce_lr],
        )

        return fitted

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    #Save model to .keras file
    def save_model(self, filename):
        self.model.save(filename)
        print(f"Model saved as {filename}")

    #Load model from .keras file, do not initialize ConvLSTM
    @classmethod
    def load_model(cls, filename, x_train, y_train, x_val, y_val, epochs, batch_size):
        loaded_model = load_model(filename)
        print(f"Model loaded from {filename}")

        conv_lstm = cls(x_train, y_train, x_val, y_val, epochs, batch_size)
        conv_lstm.model = loaded_model

        return conv_lstm