# support vector machine (SVM) machine learning model
# goal: predict a binary outcome based on multiple features
import pandas as pd
from sklearn import datasets
cancer = datasets.load_breast_cancer()
X, y = datasets.load_breast_cancer(return_X_y=True)
# print feature names (parameters)
print("Features: ", cancer.feature_names)
 # print target names (malignant vs. benign)
print("Labels: ", cancer.target_names)
 # print target labels (0 = benign, 1 = malignant)
print(cancer.target)
# split data into training (70%) vs. test (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=113)
# run SVM model with default parameters as a baseline
from sklearn import svm
clf = svm.SVC()
# train the model using training datasets
clf.fit(X_train, y_train)
# predict the outcome (1|0) for the test dataset
y_pred = clf.predict(X_test)
# assess model performance
from sklearn import metrics
# the model predicts accurately X% of the time overall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# of the samples that we predicted malignant, X% were actually malignant
print("Precision:",metrics.precision_score(y_test, y_pred))
# of the samples that were actually malignant, we accurately predicted X%
print("Recall:",metrics.recall_score(y_test, y_pred))
# overall accuracy is 90% but we generally want >=95%
# precision is 88% meaning false-positives are lowering our precision
# recall is very high (97%)
##### tuning hyperparameters:
# kernel: this parameter transforms the data input into the required form (linear, poly, etc.)
# regularization: this parameter (C) represents the error term (i.e. the size of the hyperplane margin)
# gamma: this parameter determines how closely to fit to the training dataset (lower = looser fit; beware over-fitting!)
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# improved accuracy (94%), improved precision (92%), and maintained recall (97%)
# set a larger hyperplane margin (C = 10 vs. C = 1)
clf = svm.SVC(kernel='linear', C=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# improved precision to 93% while maintaining accuracy and recall
##### scaling data:
# scaling each x attribute to [0,1] or standardizing it to have mean = 0 and variance = 1 may improve performance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# create a pipeline object
pipe = make_pipeline(StandardScaler(),LogisticRegression())
# train the model using scaled training datasets
pipe.fit(X_train, y_train)
# predict the response (1|0) for the test dataset
y_pred = pipe.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# achieved 98% accuracy, 97% precision, and 99% recall, but beware overfitting
##### cross-validation:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# knowledge about the test set can "leak" into the model when tuning hyperparameters, leading to overfitting.
# one way to avoid overfitting is to divide data into training, validation, and test sets, but this method reduces sample size.
# an alternative is to use k-fold cross-validation where the training set is split into k subsets (k-folds)
# for each k-fold, the model is trained using (k - 1) folds of the training set and validated using the remainder
# so accuracy, precision, and recall for a k-fold cross-validation are now the averages of those values computed in the CV loop
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=10, random_state=113)
scores = cross_val_score(clf, X, y, cv=10)
print("Cross-Validation Accuracy:", scores.mean())
# achieved 95% cross-validation accuracy