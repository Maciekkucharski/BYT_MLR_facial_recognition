from os.path import join, dirname, abspath
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import nn

MODEL_FILENAME = "model"
NN_TYPE = "MaciekNet"
EPOCHS = 20
CLASSES = 7
BATCH_SIZE = 128
DIMENSIONS = (48, 48)
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
IMAGE_DIMENSIONS = (48, 48, 1)
DATASET_PATH = join(dirname(abspath(__file__)), "images")


train_data_generator = ImageDataGenerator()
validation_data_generator = ImageDataGenerator()
train_dataset = train_data_generator.flow_from_directory(join(DATASET_PATH, "train"),
                                                         target_size=DIMENSIONS,
                                                         color_mode="grayscale",
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='categorical',
                                                         shuffle=True
                                                         )
validation_dataset = validation_data_generator.flow_from_directory(join(DATASET_PATH, "train"),
                                                                   target_size=DIMENSIONS,
                                                                   color_mode="grayscale",
                                                                   batch_size=BATCH_SIZE,
                                                                   class_mode='categorical',
                                                                   shuffle=False
                                                                   )

model = nn.MaciekNet.build(IMAGE_DIMENSIONS, CLASSES, OPTIMIZER, LOSS)
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
ckpt = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
callbacks_list = [earlyStopping, ckpt, reduce_lr_loss]

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=EPOCHS,
                    callbacks=callbacks_list
                    )

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
plt.savefig()
