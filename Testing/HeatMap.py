from Classfication_report import load_saved_model
import keras as ks 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.model_selection import train_test_split, KFold
from matplotlib import cm
from data import load_data
# load the test data

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

Proposed_model = "path/to/the/saved/proposedmodel"

model  = load_saved_model(Proposed_model)

def get_img_array(img):
    array = ks.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


#Dsiaplay the origninal image

img_array = X_test[193]
img_array = np.expand_dims(img_array, axis = 0)
plt.imshow( X_test[193])
plt.axis('off')
plt.show()

grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer('specific_layer').output,  model.output]
)


with tf.GradientTape() as tape:
  last_conv_layer_output, preds = grad_model(img_array)
  print("test",last_conv_layer_output.shape)
  pred_index = tf.argmax(preds[0])
  class_channel = preds[:, pred_index]

grads = tape.gradient(class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis= (0,1))
print("test2", pooled_grads.shape)
last_conv_layer_output = last_conv_layer_output
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# For visualization purpose, we will also normalize the heatmap between 0 & 1

heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
print(np.min(heatmap), np.max(heatmap))
heatmap = np.reshape(heatmap, (9,9))

#display the heatmap 
plt.imshow(heatmap)
plt.axis('off')


# display the grad cam for feature visualze the regions in an input image that were most important  in making a particular prediction
def display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4,preds=[0,0]):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = ks.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = ks.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = ks.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
 

    plt.axis('off')