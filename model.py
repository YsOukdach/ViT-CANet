import cv2
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from data import load_data
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from ViT import mlp, PatchEncoder, Patches
from Data_augmentation import data_augmentation
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split, KFold
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Dropout, Dense, Flatten, Attention, concatenate, BatchNormalization
from ViT import mlp, PatchEncoder, Patches


#Vit huper-parameters

learning_rate = 0.0001
weight_decay = 0.001
batch_size = 16
num_epochs = 100
image_size = 336  
patch_size = 36  
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  
transformer_layers = 8
mlp_head_units = [2048, 1024]
input_shape =(336, 336, 3)

def create_Attention_CNN_model(input_shape):
    inputs = Input(shape = input_shape)
    augmentaion = data_augmentation(inputs)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(augmentaion)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D()(x1)

    # Block 2
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D()(x2)

    # #Middel block
    # ############################################### Middel block ###########################################################
    X_MiddleB_1 = Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding='same')(x2)
    X_MiddleB_1 = BatchNormalization()(X_MiddleB_1)
    X_MiddleB_1 = MaxPooling2D()(X_MiddleB_1)


    X_MiddleB_2 = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding='same')(X_MiddleB_1)
    X_MiddleB_2 = BatchNormalization()(X_MiddleB_2)
    X_MiddleB_2 = MaxPooling2D()(X_MiddleB_2)

    X_MiddleB_3 = Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same')(X_MiddleB_2)
    X_MiddleB_3 = BatchNormalization()(X_MiddleB_3)
    X_MiddleB_3 = MaxPooling2D()(X_MiddleB_3)

    X_MiddleB_4 = Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu', padding='same' )(X_MiddleB_3)
    X_MiddleB_4 = BatchNormalization()(X_MiddleB_4)
    X_MiddleB_4 = MaxPooling2D()(X_MiddleB_4)

    X_MiddleB_5 = Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu', padding='same')(X_MiddleB_4)
    X_MiddleB_5 = BatchNormalization()(X_MiddleB_5)
    X_MiddleB_5 = MaxPooling2D()(X_MiddleB_5)
    FlattenMidelLAyer = Flatten()(X_MiddleB_5)
    ################################################## Middel block ##########################################################

    # Block 3
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    x3 = MaxPooling2D()(x3)

    # Block 4
    x4 = Conv2D(256, (3, 3), activation='relu', padding='same')(x3)
    x4 = Conv2D(256, (3, 3), activation='relu', padding='same')(x4)
    x4 = MaxPooling2D()(x4)

    # Block 5
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same')(x4)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same')(x5)
    x5 = MaxPooling2D()(x5)
    flattenlayer = Flatten()(x5)

    #Extract feature maps
    #"1
    query1 = GlobalAveragePooling2D()(x1)
    key1 = GlobalAveragePooling2D()(x1)
    value1 = GlobalAveragePooling2D()(x1)
    #2
    query2 = GlobalAveragePooling2D()(x2)
    key2 = GlobalAveragePooling2D()(x2)
    value2 = GlobalAveragePooling2D()(x2)

    #3
    query3 = GlobalAveragePooling2D()(x3)
    key3 = GlobalAveragePooling2D()(x3)
    value3 = GlobalAveragePooling2D()(x3)

    #4
    query4 = GlobalAveragePooling2D()(x4)
    key4 = GlobalAveragePooling2D()(x4)
    value4 = GlobalAveragePooling2D()(x4)

    #5
    query5 = GlobalAveragePooling2D()(x5)
    key5 = GlobalAveragePooling2D()(x5)
    value5 = GlobalAveragePooling2D()(x5)
    #
    #concat1

    #

    query_concat1 = concatenate([query1, query5])
    key_concat1 = concatenate([key1, key5])
    value_concat1 = concatenate([value1, value5])

    #concat2
    query_concat2 = concatenate([query2, query5])
    key_concat2 = concatenate([key2, key5])
    value_concat2 = concatenate([value2, value5])

    #concat3
    query_concat3 = concatenate([query3, query5])
    key_concat3 = concatenate([key3, key5])
    value_concat3 = concatenate([value3, value5])

    #concat4
    query_concat4 = concatenate([query4, query5])
    key_concat4 = concatenate([key4, key5])
    value_concat4 = concatenate([value4, value5])
    #Apply Attention layer
    attention_output1 = Attention()([query_concat1, key_concat1, value_concat1])
    attention_output2 = Attention()([query_concat2, key_concat2, value_concat2])
    attention_output3 = Attention()([query_concat3, key_concat3, value_concat3])
    attention_output4 = Attention()([query_concat4, key_concat4, value_concat4])
    attention_output5 = Attention()([query5, key5, value5])

    attention_outputFinal = concatenate([attention_output1, attention_output2, attention_output3, attention_output4, attention_output5])
    flattenLayerBlock_Midel = concatenate([flattenlayer, FlattenMidelLAyer])
    x_Final = concatenate([flattenlayer, FlattenMidelLAyer, attention_outputFinal])

    x_Final = Dense(254, activation='relu')(x_Final)
    x_Final = Dropout(0.25)(x_Final)

    x_Final = Dense(128, activation='relu')(x_Final)
    # x_Final = Dropout(0.25)(x_Final)
    x_Final = Dense(64, activation='relu')(x_Final)
    x_Final = Dense(32, activation='relu')(x_Final)
    x_Final = Dropout(0.25)(x_Final)
    DenseLayer = Dense(16, activation='relu')(x_Final)
    DenseLayer = Dense(2, activation='sigmoid')(DenseLayer)

    CNN_Attention_Model = Model(inputs=inputs, outputs=DenseLayer)
    return CNN_Attention_Model

ModelAttentionCNN = create_Attention_CNN_model(input_shape)

def CNN_Attention_ViT():

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    Cnn_Features = ModelAttentionCNN(augmented)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1
        )
        encoded_patches = layers.Add()([x3, x2])


    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.25)(representation)
    # Add MLP.
    CNN_Feaures = mlp(Cnn_Features, hidden_units=mlp_head_units, dropout_rate=0.5)
    ViT__Features= mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    Total_Feauters = layers.concatenate([CNN_Feaures, ViT__Features], axis = -1)
    logits = layers.Dense(2)(Total_Feauters)
    # # Create the Keras model.
    model = Model(inputs = inputs, outputs = logits)

    model.summary()

    return model