import numpy as np 
import keras
from data import load_data
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.models import Model
from Data_augmentation import data_augmentation
from sklearn.model_selection import train_test_split, KFold
from ViT import mlp, PatchEncoder, Patches
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Dropout, Dense, Flatten, Attention, concatenate, BatchNormalization
from model import create_Attention_CNN_model, CNN_Attention_ViT
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
#Load the dataset

path_Abnormal = 'Dataset/Abnormal/*.*'
path_Normal = 'Dataset/Abnormal/*.*'

data, labels = load_data(path_Abnormal, path_Normal)

data = np.array(data)
labels = np.array(labels)

num_classes = 2
input_shape = (336, 336, 3)
X_train, X_val, ytrain, yval = train_test_split(data, labels, test_size=0.2,
                                                random_state=43)
X_train, X_test, ytrain, ytest = train_test_split(X_train, ytrain, test_size=0.2,
                                                random_state=43)

#apply data ausgmentation and create the cnn based attention model
image_size = 336
data_augmentation = data_augmentation(image_size, X_train)
ModelAttentionCNN = create_Attention_CNN_model(input_shape)

def callbacks():
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience= 10, min_lr=0.000003125)
    es_callbacks = EarlyStopping(monitor="val_loss",
                                          mode="min",
                                          verbose=1,
                             patience =20 )
    filepath='model_checkpoint_epoch_{epoch:02d}.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath,
        monitor='val_loss',  
        save_best_only=True,  
        save_weights_only=False,  
        verbose=1  
    )
    return [reduce_lr, es_callbacks, checkpoint_callback]
def train_resVIT_Withoutkf(model):



    optimizer = tf.keras.optimizers.Adam(
        learning_rate = 0.001
    )


    model.compile(
            optimizer = optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )
    model.fit(
    x=X_train,
    y=ytrain,
    batch_size=32,
    epochs=100,
    validation_data = (X_val, yval),
        callbacks = callbacks()
        )


