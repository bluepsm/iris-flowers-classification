import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.plot import *
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import seaborn as sns

# pd.options.display.float_format = '{:.2f}'.format

random_seed = 42

iris_dataset = datasets.load_iris()

features = iris_dataset.data
print('Features:')
print(features)

feature_names = iris_dataset.feature_names
print('Feature Names:')
print(feature_names)

labels = iris_dataset.target
print('Labels:')
print(labels)

classes_name = iris_dataset.target_names
print('Class names:')
print(classes_name)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)
# print('x_train')
# print(x_train)
# print('x_test')
# print(x_test)
# print('y_train')
# print(y_train)
# print('y_test')
# print(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
# print('After scaling data:')
# print(x_train)
# print(x_test)

# plot_scatter(x_train[:, 0], x_train[:, 1], feature_names[0], feature_names[1], y_train, classes_name)
# plot_scatter(x_train[:, 2], x_train[:, 3], feature_names[2], feature_names[3], y_train, classes_name)

# x_train_reduced = PCA(n_components=3).fit_transform(x_train)
# plot_scatter_3d(
#     x_train_reduced[:, 0],
#     x_train_reduced[:, 1],
#     x_train_reduced[:, 2],
#     y_train,
#     classes_name,
# )

# x_train_sepal = x_train[:, :2]
# logreg_model = LogisticRegression(C=1e5)
# logreg_model.fit(x_train_sepal, y_train)
# plot_scatter_logreg(logreg_model, x_train_sepal, 'Sepal length', 'Sepal width', y_train)
# model_filename = 'saved_models/iris_flower_sepal_logreg_model.sav'
# joblib.dump(logreg_model, model_filename)

x_train_petal = x_train[:, 2:4]
logreg_model = LogisticRegression(C=1e5)
logreg_model.fit(x_train_petal, y_train)
plot_scatter_logreg(logreg_model, x_train_petal, 'Petal length', 'Petal width', y_train)
model_filename = 'saved_models/iris_flower_petal_logreg_model.sav'
joblib.dump(logreg_model, model_filename)

# loaded_model = joblib.load('saved_models/iris_flower_sepal_logreg_model.sav')
# result = loaded_model.score(x_test[:, :2], y_test)
# print('Sepal Features - Prediction on Test Set:')
# print(result)

loaded_model = joblib.load('saved_models/iris_flower_petal_logreg_model.sav')
result = loaded_model.score(x_test[:, 2:4], y_test)
print('Petal Features - Prediction on Test Set:')
print(result)

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy: ', accuracy)

k_values = [i for i in range(1, len(x_train))]
scores = []

features = scaler.fit_transform(features[:, 2:4])

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, features, labels, cv=5)
    scores.append(np.mean(score))

# sns.lineplot(x=k_values, y=scores, markers='o')
# plt.xlabel('K Values')
# plt.ylabel('Accuracy Score')
# plt.show()

best_index = np.argmax(scores)
best_k = k_values[best_index]
print('Best k Value = ', best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train[:, 2:4], y_train)

y_pred = knn.predict(x_test[:, 2:4])
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
print('Report:', report)



