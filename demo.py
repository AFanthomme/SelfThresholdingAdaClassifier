import logging
import cProfile
import numpy as np
from custom_classifiers import SelfThresholdingAdaClassifier
from sklearn.tree import DecisionTreeClassifier

# The class outputs are done through logging for later purposes, so we need to set this up
logging.basicConfig(level=logging.INFO)


# Initiate an instance of the self_thresholding adaboost classifier with a stump as weak learner
# This is a pretty good off-the-shelf solution to begin with
decision_stump = DecisionTreeClassifier(max_depth=1)
my_classifier = SelfThresholdingAdaClassifier(base_estimator=decision_stump, n_estimators=300)


# Here, load training and test data sets (please provide your own scaled dataset)

# mask = np.random.randint(0, len(np.loadtxt('datasets/full_training_set.txt')), 1000)
training_set = np.loadtxt('datasets/full_training_set.txt')
training_labels = np.loadtxt('datasets/full_training_labels.txt')

plop = np.loadtxt('datasets/full_test_set.txt')
test_set, optimization_set = plop[:np.ma.size(plop, 0)//2], plop[np.ma.size(plop, 0)//2:]

plop = np.loadtxt('datasets/full_test_labels.txt')
test_labels, optimization_labels = plop[:np.ma.size(plop, 0)//2], plop[np.ma.size(plop, 0)//2:]

# Do the initial fit and calibration of the adaboost ensemble
my_classifier.fit(training_set, training_labels)

# Start exploring the space of thresholds.
# Here, could implement some kind of task dispatch

# my_classifier.explore_thresholds(optimization_set, optimization_labels, n_points=3)
# my_classifier.explore_history()
# my_classifier.predict(test_set)

cProfile.run('my_classifier.explore_thresholds(test_set, test_labels, n_points=6)')

