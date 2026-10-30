# model.py - a simple fully connected Artificial Neural Network
import keras
from keras import layers


def build_model(input_size=3072):
    model = keras.Sequential([
        keras.Input(shape=(input_size,)),          # input layer: 3072 pixel values

        layers.Dense(512, activation="relu"),      # hidden layer 1
        layers.BatchNormalization(),               # keeps values in a stable range -> faster learning
        layers.Dropout(0.3),                       # randomly switches off 30% neurons -> less overfitting

        layers.Dense(256, activation="relu"),      # hidden layer 2
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),      # hidden layer 3
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid"),     # output: probability that the image is REAL
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",                # standard loss for 2-class problems
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
