### Libraries ###
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
recall_score, f1_score, classification_report, confusion_matrix,
ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

### Load the Dataset ###
import numpy as np
iris = load_iris()
X = iris.data # shape: (150, 4)
y = iris.target # shape: (150,) values: 0, 1, 2
print('Features:', iris.feature_names)
print('Classes :', iris.target_names)
print('X shape :', X.shape)
print('Class distribution:', np.bincount(y))
print()

### Data Preprocessing ###
# Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2, # 80% train, 20% test
random_state=42, # reproducibility
stratify=y # preserve class proportions in both sets
)
print('Train size:', X_train.shape[0]) # 120
print('Test size :', X_test.shape[0]) # 30

# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train) # fit + transform on train
X_test_sc = scaler.transform(X_test) # transform only on test
# Verify: after scaling, training set should have ~0 mean and ~1 std
print('Mean after scaling:', X_train_sc.mean(axis=0).round(3))
print('Std after scaling:', X_train_sc.std(axis=0).round(3))
print()

### NN Architecture ###
nn = MLPClassifier(
hidden_layer_sizes=(128, 64, 32), # two hidden layers (Exp2: (64,32) or Exp1: (32, ))
activation='relu', # ReLU for hidden layers
solver='adam', # adaptive optimizer
learning_rate_init=0.001, # initial learning rate (suggested: 0.001) (initial: 0.01)
batch_size=16, # mini-batch size
max_iter=500, # maximum epochs
early_stopping=True, # stop if val loss stops improving
validation_fraction=0.1, # 10% of train used for validation
n_iter_no_change=20, # patience: 20 epochs
random_state=42
)
nn.fit(X_train_sc, y_train)
print('Epochs trained:', nn.n_iter_)
print()

### Testing and Model Evaluation ###
# Evaluation Metrics
y_pred = nn.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'Accuracy : {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall : {rec:.4f}')
print(f'F1-Score : {f1:.4f}')
# Detailed per-class breakdown
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Neural Network — Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Training Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(nn.loss_curve_, label='Training Loss', color='steelblue')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()
