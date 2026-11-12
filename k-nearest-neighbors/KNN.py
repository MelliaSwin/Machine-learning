# ── Imports ──────────────────────────────────────────────────────────
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from collections import Counter
# ── Part 1: Load Data ────────────────────────────────────────────────
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
# metadata
print(breast_cancer_wisconsin_diagnostic.metadata)
# variable information
print(breast_cancer_wisconsin_diagnostic.variables)
print()

# EDA — shape, class distribution, heatmap, histograms
# First 5 rows of features and target
print(f'First 5 rows of features:\n{X.head()}\n')
print(f'First 5 rows of target:\n{y.head()}\n')

# Dataframe shapes
print(f'Shape of X:\n{X.shape}\n')  # 30 Columns, 569 rows
print(f'Shape of y:\n{y.shape}\n')  # 1 column, 569 rows

# Class Distribution
print(f'Total Malignant in dataset: {y.value_counts().get('M', 0)}')    # 212
print(f'Total Benign in dataset: {y.value_counts().get('B', 0)}\n')     # 357

# Check for missing values using isnull().sum()
print(f'Missing Values X:\n{X.isnull().sum()}\n')
print(f'Missing Values y:\n{y.isnull().sum()}\n')

# Compute and display summary statistics: mean, std, min, max per feature (describe())
print(f'Summary Statistics:\n{X.describe()}')

# Plot histogram of one feature per histogram, colored by class label (include 2-3 plots)
# Radius
plt.hist(X[y.iloc[:, 0] == 'M']['radius1'], bins=20, color='red', edgecolor="white", alpha=0.5, label='Malignant')
plt.hist(X[y.iloc[:, 0] == 'B']['radius1'], bins=20, color='steelblue', edgecolor="white", alpha=0.5, label='Benign')
plt.title('Radius1 Feature Histogram')
plt.xlabel("Data Range")
plt.ylabel("Frequency")
plt.legend()
plt.show()
# Area
plt.hist(X[y.iloc[:, 0] == 'M']['area1'], bins=20, color='red', edgecolor="white", alpha=0.5, label='Malignant')
plt.hist(X[y.iloc[:, 0] == 'B']['area1'], bins=20, color='steelblue', edgecolor="white", alpha=0.5, label='Benign')
plt.title('Area1 Feature Histogram')
plt.xlabel("Data Range")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Plot a correlation heatmap (seaborn.heatmap) to identify highly correlated features
plt.figure(figsize=(16, 14))
sns.heatmap(X.corr(numeric_only=True))
plt.show()

# ── Part 2: Preprocessing ────────────────────────────────────────────
# train_test_split with stratify=y
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)

# Encode labels
y_train_enc = y_train.iloc[:, 0].map({'M': 1, 'B': 0}).to_numpy()
y_test_enc = y_test.iloc[:, 0].map({'M': 1, 'B': 0}).to_numpy()

# Verify class distribution is similar in train and test sets
print(f'Train set distribution:\n{y_train.iloc[:, 0].value_counts(normalize=True)}\n')
print(f'Test set distribution:\n{y_test.iloc[:, 0].value_counts(normalize=True)}\n')

# StandardScaler — fit on train, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on train
X_test_scaled = scaler.transform(X_test)  # transform test with same scaler
# ── Part 3: Baseline Model ───────────────────────────────────────────
# Train KNN with k=5, euclidean AND manhattan
class KNNClassification:
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    def distance(self, x):
        diff = self.X_train - x
        if self.metric == 'euclidean':
            return np.sqrt((diff ** 2).sum(axis=1))
        elif self.metric == 'manhattan':
            return np.abs(diff).sum(axis=1)
        else:
            raise ValueError(f'Unknown metric: {self.metric}')

    def predict_one(self, x):
        dists = self.distance(x)
        nn_idx = np.argsort(dists)[:self.k]
        nn_vals = self.y_train[nn_idx]
        return Counter(nn_vals).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = np.array(y)
        y_pred = self.predict(X)
        return np.mean(y_true == y_pred)

# Baseline KNN Distance Metric Comparison
knn = KNNClassification(k=5, metric='euclidean')
knn.fit(X_train_scaled, y_train_enc)
euc_acc = knn.accuracy(X_test_scaled, y_test_enc)
print(f'Baseline Accuracy for Euclidean: {euc_acc:.2f}')

knn = KNNClassification(k=5, metric='manhattan')
knn.fit(X_train_scaled, y_train_enc)
man_acc = knn.accuracy(X_test_scaled, y_test_enc)
print(f'Baseline Accuracy for Manhattan: {man_acc:.2f}')

print(f'Difference between Euclidean and Manhattan: {euc_acc - man_acc}')

# ── Part 4: Hyperparameter Tuning ────────────────────────────────────
# Loop over k=1..30, plot CV accuracy vs k
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train_enc, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot k (x-axis) vs mean CV acc (y-axis); identify optimal k
plt.plot(k_range, cv_scores, marker='o')
plt.title('KNN Accuracy')
plt.xlabel('K')
plt.ylabel('Mean CV Accuracy')
plt.show()

# GridSearchCV over n_neighbors, metric, weights

param_grid ={'n_neighbors': k_range,
             'metric': ['euclidean', 'manhattan']}
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                           scoring='accuracy', cv=5)
grid_search.fit(X_train_scaled, y_train_enc)
print(f'\nBest Hyperparameters: {grid_search.best_params_}\n')  # Optimal: euc, k =3
print(f'\nHighest CV: {grid_search.best_score_}\n')
# ── Part 5: Evaluation ───────────────────────────────────────────────
# classification_report, confusion matrix heatmap, ROC curve
best_k = grid_search.best_params_['n_neighbors']
best_metric = grid_search.best_params_['metric']
model = KNNClassification(k=best_k, metric=best_metric)
model.fit(X_train_scaled, y_train_enc)
y_pred = model.predict(X_test_scaled)

# Classification Report
print(f'\n Classification Report:\n{classification_report(y_test_enc, y_pred)}')

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
print(f'Confusion Matrix:\n{cm}')
# Here its: Actual (Y-Axis), Predicted (X-Axis)
# Plot using seaborn.heatmap with annot
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# ROC (from sklearns KNN)
sk_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
sk_knn.fit(X_train_scaled, y_train_enc)
y_prob = sk_knn.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test_enc, y_prob)
print('KNeighborsClassifier AUC from sklearn:')
print('AUC:', auc(fpr, tpr))
# Plot ROC curve and annotatee the AUC score
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'ROC Curve AUC: {auc(fpr, tpr)}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# ── Part 6: Visualization ────────────────────────────────────────────
# PCA → decision boundary, k vs accuracy plot, permutation importanc
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Retrain KNN on 2D PCA features
pca_knn = KNNClassification(k=best_k, metric=best_metric)
pca_knn.fit(X_train_pca, y_train_enc)

X_set, y_set = X_test_pca, y_test_enc
# Plot the decision boundary using a meshgrid
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step=0.1)
                     )
plt.figure(figsize=(10, 10))
plt.contourf(X1, X2,
             pca_knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,
             cmap='PuRd')
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

labels = {0: 'Benign', 1: 'Malignant'}
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                label=labels[j],
                edgecolor='k')
plt.title('KNN Decision Boundary (2D Projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# k vs Accuracy Plot
k_range = range(1, 31)
train_acc_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_enc)
    train_acc_scores.append(knn.score(X_train_scaled, y_train_enc))

# Plot k (x-axis) vs train acc & CV acc (y-axis); identify optimal k
plt.plot(k_range, train_acc_scores, marker='o', label='Train Accuracy')
plt.plot(k_range, cv_scores, marker='o', label='Cross-validation Accuracy')
plt.axvline(x=best_k, linestyle='--', label=f'Optimal k = {best_k}')

plt.title('K vs Train Acc & CV Acc')
plt.xlabel('K')
plt.ylabel('Train & CV Accuracy')
plt.legend()
plt.show()

# Train KNN model with same k value (compare baseline (mine) vs sk_knn f1 score)
# sk_knn previously done in ROC (will use here):
# Baseline:
f1_baseline = f1_score(y_test_enc, y_pred)
# Sklearn:
sk_y_pred = sk_knn.predict(X_test_scaled)
f1_sk = f1_score(y_test_enc, sk_y_pred)

# Compare two scores in a plot:
print(f'\nF1 Score of baseline: {f1_baseline}')
print(f'F1 Score of sklearn: {f1_sk}\n')

labels =['Baseline KNN', 'Sklearn KNN']
f1_scores = [f1_baseline, f1_sk]

plt.figure(figsize=(8, 8))
plt.bar(labels, f1_scores)
plt.title('F1 Score: Baseline KNN vs. Sklearn KNN')
plt.ylabel('F1 Score')
plt.ylim([0, 1])
plt.show()

