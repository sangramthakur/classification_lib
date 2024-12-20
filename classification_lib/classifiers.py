# author= Sangram Biplab Manabendra Thakur
# author_email= sangramaimlds@gmail.com

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier, LogisticRegressionCV, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.dummy import DummyClassifier

def evaluate_classifiers(df, target_column, test_size=0.2, random_state=42):
    # Drop rows with any NaN values
    df_cleaned = df.dropna()

    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target column

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # List of models to evaluate
    models = [
        ('LinearSVC', LinearSVC()),
        ('SGDClassifier', SGDClassifier()),
        ('MLPClassifier', MLPClassifier()),
        ('Perceptron', Perceptron()),
        ('LogisticRegression', LogisticRegression()),
        ('LogisticRegressionCV', LogisticRegressionCV()),
        ('SVC', SVC()),
        ('CalibratedClassifierCV', CalibratedClassifierCV()),
        ('PassiveAggressiveClassifier', PassiveAggressiveClassifier()),
        ('LabelPropagation', LabelPropagation()),
        ('LabelSpreading', LabelSpreading()),
        ('RandomForestClassifier', RandomForestClassifier()),
        ('GradientBoostingClassifier', GradientBoostingClassifier()),
        ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()),
        ('HistGradientBoostingClassifier', HistGradientBoostingClassifier()),
        ('RidgeClassifierCV', RidgeClassifierCV()),
        ('RidgeClassifier', RidgeClassifier()),
        ('AdaBoostClassifier', AdaBoostClassifier()),
        ('ExtraTreesClassifier', ExtraTreesClassifier()),
        ('KNeighborsClassifier', KNeighborsClassifier()),
        ('BaggingClassifier', BaggingClassifier()),
        ('BernoulliNB', BernoulliNB()),
        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        ('GaussianNB', GaussianNB()),
        ('NuSVC', NuSVC()),
        ('DecisionTreeClassifier', DecisionTreeClassifier()),
        ('NearestCentroid', NearestCentroid()),
        ('ExtraTreeClassifier', ExtraTreeClassifier()),
        ('DummyClassifier', DummyClassifier())
    ]

    # Dictionary to store results
    results = []

    # Evaluate each model
    for name, model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confusion Matrix and Percentages
        cm = confusion_matrix(y_test, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to get percentage
        
        # Extract True Positive, True Negative, False Positive, False Negative
        tp, fp, fn, tn = cm.ravel()
        tp_percent = (tp / (tp + fn)) * 100
        tn_percent = (tn / (tn + fp)) * 100
        fp_percent = (fp / (tn + fp)) * 100
        fn_percent = (fn / (tp + fn)) * 100
        
        time_taken = time.time() - start_time

        # Append results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy,
            'ROC AUC': roc_auc,
            'F1 Score': f1,
            'Time Taken': time_taken,
            'Confusion Matrix': cm,
            'Confusion Matrix Percentages': cm_percentage,
            'TP Percentage': tp_percent,
            'FP Percentage': fp_percent,
            'FN Percentage': fn_percent,
            'TN Percentage': tn_percent
        })

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)

    # Sort by accuracy to identify the best performing models
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    return results_df
