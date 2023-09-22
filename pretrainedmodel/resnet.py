import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, Xception, DenseNet169, NASNetMobile, EfficientNetB0

def create_resnet_model(input_shape):
    # Load pre-trained ResNet50 model
    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in resnet.layers:
        layer.trainable = False

    # Create the custom top layers
    model_resnet = tf.keras.Sequential([
        resnet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_resnet

def create_nasnet_model(input_shape):
    # Load pre-trained NasNet (InceptionResNetV2) model
    InceptionResNetV2 = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in InceptionResNetV2.layers:
        layer.trainable = False

    # Create the custom top layers
    model_nasnet = tf.keras.Sequential([
        InceptionResNetV2,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_nasnet
def create_xception_model(input_shape):
    # Load pre-trained Xception model
    xpt = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in xpt.layers:
        layer.trainable = False

    # Create the custom top layers
    model_xception = tf.keras.Sequential([
        xpt,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_xception

def create_denseNet_model(input_shape):
    # Load pre-trained Xception model
    DenseNet169 = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in DenseNet169.layers:
        layer.trainable = False

    # Create the custom top layers
    model_DenseNet = tf.keras.Sequential([
        DenseNet169,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_DenseNet

def create_NASNetMobile_model(input_shape):
    # Load pre-trained Xception model
    NASNetMobile = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in NASNetMobile.layers:
        layer.trainable = False

    # Create the custom top layers
    model_NASNet = tf.keras.Sequential([
        NASNetMobile,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_NASNet

def create_EfficientNetB0_model(input_shape):
    # Load pre-trained Xception model
    EfficientNetB0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze layers
    for layer in EfficientNetB0.layers:
        layer.trainable = False

    # Create the custom top layers
    model_EfficientNet = tf.keras.Sequential([
        NASNetMobile,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Flatten()
    ])

    return model_EfficientNet
