from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D


class MaciekNet:
    '''
    CNN - MaciekNet.
    INPUT (48px x 48px x 1) =>
        CONV => RELU => POOL =>
        CONV => RELU => POOL =>
        CONV => RELU => POOL =>
        CONV => RELU => POOL =>
        FC => RELU =>
        FC => RELU =>
        SOFTMAX
    '''

    @staticmethod
    def build(dims: tuple, classes: int, optimizer: str, loss: str):
        model = Sequential()

        # 1 CNN
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=dims, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))  # extracts important features from images
        model.add(Dropout(0.25))  # prevent model to get over fitted

        # 2 CNN
        model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3 CNN
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4 CNN
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())  # flatten to have 1 dim data

        # 1 FC
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # 2 FC
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # softmax in order to get probability
        model.add(Dense(classes, activation='softmax'))

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model
