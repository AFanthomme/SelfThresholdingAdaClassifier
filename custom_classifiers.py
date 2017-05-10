import copy
import logging
from operator import itemgetter
import matplotlib.pyplot as p
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

debug = True

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n /= b
    return digits

def n_dimensional_interator(dimension, support=(0, 1), n_points=10):
    plop = np.linspace(support[0], support[1], n_points)
    for flat_idx in range(n_points**dimension):
        coordinates = numberToBase(flat_idx, n_points)
        while len(coordinates) < dimension:
            coordinates.append(0)
        yield [plop[i] for i in coordinates]


class SelfThresholdingAdaClassifier:
    def __init__(self, base_estimator, n_estimators=50):
        self.UncalibratedAdaBoost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
        self.CalibratedAdaBoost = self.UncalibratedAdaBoost
        self.n_classes_ = None
        self.is_fitted = False
        self.is_optimized = False
        self.ground_truth = None
        self.thresholds = None
        self.history = {}
        self.scores = None

    @staticmethod
    def filter_scores(scores_array, thresholds):
        # If an event has category 0 score above background_threshold, goes to category 0
        background_threshold = 1. - thresholds[0]
        thresholds[0] = 0
        background_like = np.where(scores_array[:, 0] > background_threshold)
        scores_array[background_like, 0] = 1.01

        # Rectify scores : if score for category i is lower than thresholds[i], set it to 0
        scores_filter = ((scores_array - thresholds) >= 0).astype(int)
        filtered_scores = np.multiply(scores_array, scores_filter)
        return filtered_scores


    def __confusion_score(self, predictions, ground_truth, weights=None):
        # [ [false_positive, false_negative] for cat in categories]
        if not weights:
            weights = np.ones((self.n_classes_, 2))
            weights[0, 0] = 2
            weights[0, 1] = 4

        confusion_matrix = np.zeros((self.n_classes_, 2))
        mask = predictions != ground_truth

        misclassified_from = ground_truth[mask]
        misclassified_into = predictions[mask]

        for cat in range(self.n_classes_):
            confusion_matrix[cat, 0] = np.sum(misclassified_into == cat)
            confusion_matrix[cat, 1] = np.sum(misclassified_from == cat)

        confusion_matrix /= len(predictions)
        return np.sum(np.multiply(confusion_matrix, weights))

    def fit(self, X, y):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5)
        self.UncalibratedAdaBoost.fit(X_train, y_train)
        self.n_classes_ = self.UncalibratedAdaBoost.n_classes_
        self.thresholds = tuple([0 for _ in range(self.n_classes_)])
        logging.info('\tRaw adaboosted model trained')
        self.CalibratedAdaBoost = CalibratedClassifierCV(self.UncalibratedAdaBoost, cv="prefit", method="sigmoid")
        self.CalibratedAdaBoost = self.UncalibratedAdaBoost.fit(X_cal, y_cal)
        logging.info('\tModel calibrated')
        self.is_fitted = True
        return self

    def predict_proba(self, X_test):
        if not self.is_fitted:
            logging.error('Trying to predict probas with an unfitted base model')
            raise AttributeError
        proba = self.CalibratedAdaBoost.predict_proba(X_test)
        self.scores = proba
        return proba

    def predict(self, X_test):
        if not self.is_optimized:
            logging.warning('Predicting without thresholding')
        scores = self.CalibratedAdaBoost.predict_proba(X_test)
        filtered_scores = self.filter_scores(scores, self.thresholds)
        return self.CalibratedAdaBoost.classes_[np.argmax(filtered_scores, axis=1)]


    def __assess_thresholds(self, thresholds):
        predictions = self.CalibratedAdaBoost.classes_[np.argmax(self.filter_scores(self.scores, thresholds), axis=1)]
        confusion_score = self.__confusion_score(predictions, self.ground_truth)
        self.history[tuple(thresholds)] = confusion_score
        return self


    def explore_thresholds(self, X_test, y_test, limits=(0., 0.5), n_points=10):
        self.scores = self.predict_proba(X_test)
        self.ground_truth = y_test

        for thresholds in n_dimensional_interator(self.n_classes_, support=limits, n_points=n_points):
            # logging.info(thresholds)
            self.__assess_thresholds(thresholds)


    def explore_history(self):
        sorted_history = sorted(self.history.items(), key=itemgetter(1), reverse=True)
        weights = [pair[0] for pair in sorted_history]
        scores = [pair[1] for pair in sorted_history]
        mean, std = np.mean(scores), np.std(scores)
        significances = (scores - mean) / std
        p.hist(significances)
        p.title('Mean : ' + str(mean) + ' , Std : ' + str(std))
        p.show()
        self.thresholds = weights[0]
        self.is_optimized = True