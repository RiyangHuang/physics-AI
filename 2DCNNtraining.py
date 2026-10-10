import os
import h5py
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import LearningRateScheduler

# Disable TensorFlow eager execution
# tf.compat.v1.disable_eager_execution()

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9


class build_model:
    @staticmethod
    def myvgg16(im_height, im_width, im_channel, num_classes, activation='sigmoid'):
        chanDim = -1
        model = Sequential()
        model.add(Input(shape=(im_height, im_width, im_channel), dtype='float32'))

        # Block 1
        model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv1'))  # 64
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))  # 128
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))  # 256
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))  # 512
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))  # 2048
        model.add(Dropout(rate=0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(num_classes, activation=activation))
        return model


if __name__ == '__main__':
    model = build_model.myvgg16(28, 28, 1, num_classes=452, activation='sigmoid')
    print(model.summary())


# Initialize parameters
epochs = 100
init_lr = 1e-3
batch_size = 200
num_class = 452
img_h, img_w, img_c = 28, 28, 1

# Data paths
path = '/root/autodl-tmp'
savepath = '/root/autodl-tmp'

# Load training data
file = h5py.File(path + '/Meansub_train_data.h5', 'r')
trainX = file['Meansub_train_data'][:]
file.close()

# Load test data
file = h5py.File(path + '/Meansub_test_data.h5', 'r')
testX = file['Meansub_test_data'][:]
file.close()

# Load training labels
file = h5py.File(path + '/Meansub_train_label.h5', 'r')  # /polar
trainY = file['Meansub_train_label'][:]
file.close()

# Load test labels
file = h5py.File(path + '/Meansub_test_label.h5', 'r')  # /azimuth
testY = file['Meansub_test_label'][:]
file.close()

# mlb = MultiLabelBinarizer()
# trainY = mlb.fit_transform(trainY)
# testY = mlb.fit_transform(testY)

# Ensure the data shapes are correct
print('Train data shape:', trainX.shape)
print('Test data shape:', testX.shape)

# trainX = np.transpose(trainX, (3, 1, 2, 0))
# testX = np.transpose(testX, (3, 1, 2, 0))
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
print('Adjusted Train data shape:', trainX.shape)
print('Adjusted Test data shape:', testX.shape)

print('{} images for training, {} images for validation'.format(len(trainX), len(testX)))

# Build the model
model = build_model.myvgg16(img_h, img_w, img_c, num_classes=num_class, activation='sigmoid')
opt = Adam(learning_rate=init_lr, beta_1=0.9, beta_2=0.999, decay=init_lr / epochs)
# opt = SGD(learning_rate=init_lr, momentum=0.9, decay=init_lr / epochs)
model.compile(
    loss=binary_crossentropy,  # binary_crossentropy / sparse_categorical_crossentropy
    optimizer=opt,
    metrics=['accuracy']
)

# Set the checkpoint save path
checkpoint_savepath = savepath + '/checkpoint/checkmodel.weights.h5'

# Save the best model
checkpoint = ModelCheckpoint(
    filepath=checkpoint_savepath,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
)

# Early stopping
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     verbose=1,
#     restore_best_weights=True
# )

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # 'accuracy'
    factor=0.1,
    patience=10,  # 10
    verbose=1,
    mode='auto'
)

print('======================== Training ===========================')

H = model.fit(
    trainX,
    trainY,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(testX, testY),
    callbacks=[checkpoint, reduce_lr],  # early_stopping
)
model.summary()

# Plot training and validation loss/accuracy
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label='train_loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath, 'train_loss.png'))
plt.close()  # Close the current figure to release memory

plt.figure()
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss', color='green')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath, 'val_loss.png'))
plt.close()

plt.figure()
plt.plot(np.arange(0, epochs), H.history['accuracy'], label='train_acc', color='red')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath, 'train_accuracy.png'))
plt.close()

plt.figure()
plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='val_acc', color='red')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath, 'val_accuracy.png'))
plt.close()

# Extract data from the training history
history_dict = H.history
epochs = range(1, len(history_dict['loss']) + 1)

# Create a DataFrame
history_df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': history_dict['loss'],
    'val_loss': history_dict['val_loss'],
    'train_accuracy': history_dict['accuracy'],
    'val_accuracy': history_dict['val_accuracy']
})

# Save the training history as a CSV file
history_df.to_csv(os.path.join(savepath, 'training_history.csv'), index=False)

print(f"Training history has been saved to {os.path.join(savepath, 'training_history.csv')}")

# Save the model, weights, and training history
model.save(savepath + '/model.h5')
model.save_weights(savepath + '/weights.h5')
with open(savepath + '/history.pickle', 'wb') as file_pi:
    pickle.dump(H.history, file_pi)