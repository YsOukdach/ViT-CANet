
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from train import test_set
from sklearn.model_selection import train_test_split, KFold
from data import load_data
import numpy as np 
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

def load_saved_model(path):
    model = tf.keras.models.load_model(path)
    model.load_weights(path)

#load the test set 

path_Abnormal = 'Dataset/Abnormal/*.*'
path_Normal = 'Dataset/Abnormal/*.*'

data, labels = load_data(path_Abnormal, path_Normal)

data = np.array(data)
labels = np.array(labels)

num_classes = 2
input_shape = (336, 336, 3)
X_train, X_val, ytrain, yval = train_test_split(data, labels, test_size=0.2,
                                                random_state=42)
X_train, X_test, ytrain, ytest = train_test_split(X_train, ytrain, test_size=0.2,
                                                random_state=42)
path = 'path/to/the/saved_model'

model = load_saved_model(path)


Y_pred = np.argmax(model.predict(X_test), axis=1)

target_names = ["Abnormal", "Normal"]
print(classification_report(ytest, Y_pred, target_names=target_names))


#load the best saved weights
checkpoint_directory = 'Path/to/to/the/checkpoint/files'

# Initialize empty lists to store precision, recall, F1, and accuracy

precision_history = []
recall_history = []
f1_history = []
accuracy_history = []


for filename in os.listdir(checkpoint_directory):
    if filename.endswith('.h5'):
        checkpoint_file = os.path.join(checkpoint_directory, filename)


        model.load_weights(checkpoint_file)

        # predictions on the test data
        test_predictions =np.argmax(model.predict(X_test), axis=1)


        precision = precision_score(ytest, test_predictions, average='micro')
        recall = recall_score(ytest, test_predictions, average='micro')
        f1 = f1_score(ytest, test_predictions, average='micro')
        accuracy = accuracy_score(ytest, test_predictions)

        # Append metrics to the history lists
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        accuracy_history.append(accuracy)

# Plot the metrics
epochs = range(1, len(precision_history) + 1)

# Plot each metric separately
plt.figure(figsize=(12, 6))

# Plot Precision
plt.subplot(2, 2, 1)
plt.plot(epochs, precision_history, label='Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Plot Recall
plt.subplot(2, 2, 2)
plt.plot(epochs, recall_history, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

# Plot F1 Score
plt.subplot(2, 2, 3)
plt.plot(epochs, f1_history, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

# Plot Accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, accuracy_history, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
