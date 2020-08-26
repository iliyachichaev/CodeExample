import tensorflow as tf
import numpy as np
from utils import CreateUNet, wbce
from prepare_data import prepare_train_data

seed = 42
np.random.seed = seed

IMG_HEIGHT = 512
IMG_WIDTH = 640
IMG_CHANNELS = 3
num_classes = 6

# Get training data
train_data = prepare_train_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, True)
X_train = train_data['X_train']
Y_class_train = train_data['Y_class_train']
Y_segm_train = train_data['Y_segm_train']

# Create and compile model
losses = {"class_output": "categorical_crossentropy", "segm_output": wbce}
lossWeights = {"class_output": 0.01, "segm_output": 1.0}
model = CreateUNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes)
model.compile(optimizer = 'adam', loss = losses, loss_weights = lossWeights, metrics=['accuracy'])
#model.compile(optimizer = 'adam', loss = wbce, metrics = ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('bacteria_model.h5', verbose = 1, save_best_only = True),
    tf.keras.callbacks.EarlyStopping(patience = 50, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs')]

# Train model
print(X_train.shape)
print(Y_class_train.shape)
print(Y_segm_train.shape)
results = model.fit(x = X_train,
                    y = {"class_output": Y_class_train, "segm_output": Y_segm_train},
                    validation_split = 0.05, batch_size = 16, epochs = 350, callbacks = callbacks)
