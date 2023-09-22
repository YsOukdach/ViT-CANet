"""In this file, please load the concatenated models and display ROC curves of all concatenated models in one graph. In the 'model.py' file,
 please call one of the pretrained models stored in the 'pretrained_models' folder and concatenate it with ViT features in the 'model.py' file using the 'CNN_Attention_ViT()' function (line 169 in the 'model.py'). 
 After training, save each model and load it in order to display the ROC curves.

 """


import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np 
from sklearn.model_selection import train_test_split, KFold
from data import load_data

#
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

# load the multiple trained models
def load_saved_model(path):
    model = tf.keras.models.load_model(path)
    model.load_weights(path)

Resent_path = "path/to/the/saved/ResNetViT"
Xeption_path = "path/to/the/saved/XeptionViT"
DenseNet_path = "path/to/the/saved/DenseNetViT"
NaseNet_path = "path/to/the/saved/NaseNetViT"
Effecient_path = "path/to/the/saved/NaseNetViT"
EnceptionResNet_path = "path/to/the/saved/EnceptionResNetViT"
Proposed_model = "path/to/the/saved/proposedmodel"

Resvit  = load_saved_model(Resent_path)
Xvit  = load_saved_model(Xeption_path)
DenseVit  = load_saved_model(DenseNet_path)
NasVit  = load_saved_model(NaseNet_path)
EffVit  = load_saved_model(Effecient_path)
EnceptionResVit  = load_saved_model(EnceptionResNet_path)
Ours = load_saved_model(Proposed_model)

#define a list of loaded models 

models = [Ours, Resvit, Xvit, DenseVit, EffVit, EnceptionResVit, NasVit]
models_name = ['Ours', 'ResVit', 'Xvit', 'DenseVit','EfficientVit' ,'EnceptionResVit', 'NasVit']


roc_data = []

for model, model_name in zip(models,models_name):
    y_prob = np.argmax(model.predict(X_test), axis=1)

    fpr, tpr, thresholds = roc_curve(ytest, y_prob)
    roc_data.append((fpr, tpr, auc(fpr, tpr), model_name))

# plot the roc curves for all models
plt.figure(figsize=(10,8))
for i, model_data in enumerate(roc_data):
    fpr, tpr, auc_score, model_name = model_data
    label = '%s (AUC = %0.2f)' % (model_name, auc_score)
    plt.plot(fpr, tpr, lw=2, label=label)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Different Models')
plt.legend(loc='lower right')
plt.show()
