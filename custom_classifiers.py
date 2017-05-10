import copy
import logging
from operator import itemgetter
import matplotlib.pyplot as p
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class SelfThresholdingAdaClassifier:
    def __init__(self, base_learner, ensemble_size=50):
        self.UncalibratedAdaBoost = AdaBoostClassifier(base_estimator=base_learner, n_estimators=ensemble_size)
        self.CalibratedAdaBoost = self.UncalibratedAdaBoost
        self.is_fitted = False
        self.thresholds = None
        self.is_optimized = False
        self.history = {}
        self.scores = None

    def fit(self, X, y):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5)
        self.UncalibratedAdaBoost.fit(X_train, y_train)
        logging.info('\tRaw adaboosted model trained')
        self.CalibratedAdaBoost = CalibratedClassifierCV(self.UncalibratedAdaBoost, cv="prefit", method="sigmoid")
        self.CalibratedAdaBoost = self.CalibratedAdaBoost.fit(X_cal, y_cal)
        logging.info('\tModel calibrated')
        self.is_fitted = True
        return self

    def predict_proba(self, X_test):
        if not self.is_fitted:
            logging.error('Trying to predict probas with an unfitted base model')
        proba = self.CalibratedAdaBoost.predict_proba(X_test)
        self.scores = proba
        return proba

    def predict(self, X_test):
        scores = self.CalibratedAdaBoost.predict_proba(X_test)
        scores_filter = ((scores - self.thresholds) >= 0).astype(int)
        filtered_scores = np.multiply(scores, scores_filter)
        possible_contaminations = np.where(scores[:, 0] > self.contamination_thr)
        scores[possible_contaminations, 0] = 1
        return self.CalibratedAdaBoost.classes_[np.argmax(filtered_scores, axis=1)]

    def assess_thresholds(self, thresholds, evaluator=roc_auc_score):
        return 0 #roc_auc_score()


    def thresholding(self, X_test, cont_param_limits=(0.3, 0.6), limits=(0.3, 0.6), n_points=4):
        scores_ref = self.CalibratedAdaBoost.predict_proba(X_test)

        for plop in np.linspace(limits[0], limits[1], n_points):
            for contamination_thresh in np.linspace(cont_param_limits[0], cont_param_limits[1], n_points):
                scores = copy.deepcopy(scores_ref)
                thresholds = np.array([0, plop, plop, plop, plop])
                scores_filter = ((scores - thresholds) >= 0).astype(int)
                possible_contaminations = np.where(scores[:, 0] > contamination_thresh)
                np.multiply(scores, scores_filter)
                scores[possible_contaminations, 0] = 1
                predictions = self.CalibratedAdaBoost.classes_[np.argmax(scores, axis=1)]

    def explore_history(self):
        sorted_history = sorted(self.history.items(), key=itemgetter(1), reverse=True)
        weights = [pair[0] for pair in sorted_history]
        scores = [pair[1] for pair in sorted_history]
        mean, std = np.mean(scores), np.std(scores)
        significances = (scores - mean) / std
        p.hist(significances)
        p.show()
        self.thresholds = weights[0]
        self.is_optimized = True