import threading
from random import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics

MAX_PARALLEL_CLASSIFIERS = 100

CLASSIFIERS = {
    "Nearest Neighbors" : KNeighborsClassifier(n_neighbors = 250),
    "Linear SVM" : SVC(kernel="linear", C=0.025),
    "RBF SVM" : SVC(gamma=2, C=1),
    # "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree" : DecisionTreeClassifier(max_depth=500),
    "Random Forest" : RandomForestClassifier(max_depth=100, n_estimators=25, max_features='auto'),
    "Neural Net" : MLPClassifier(),
    "AdaBoost" : AdaBoostClassifier(n_estimators=250),
    "Naive Bayes" : GaussianNB(),
    # "QDA" : QuadraticDiscriminantAnalysis()
    }


class Classifier():
    def __init__(self, name, classifier, train_dataset, train_labels, test_dataset, expected):
        self.name = name
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.test_dataset = test_dataset
        self.expected = expected

    def report(self):
        print("Classification report for classifier %s:\n%s\n" % (self.classifier,
                                                                  metrics.classification_report(self.expected, self.predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.expected, self.predicted))

    def run_classifier(self):
        print("Running %s classifier, fitting on train dataset" % (self.name))

        self.classifier.fit(self.train_dataset, self.train_labels)

        print("%s classifier predicting on test dataset" % (self.name))
        self.predicted = self.classifier.predict(self.test_dataset)

        # self.report()

        counter = 0
        initial_index = self.expected.index[0]
        for index, prediction in enumerate(self.predicted):
            if prediction == self.expected[initial_index + index]:
                counter += 1
        print("%s: %s out of %s" % (self.name, counter, len(self.predicted)))


class ClassificationData():
    def __init__(self, train_dataset, train_labels, test_dataset, test_labels):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels


class ClassifierRunner(threading.Thread):
    def __init__(self, classifier_name, classifier, classification_data):
        threading.Thread.__init__(self)
        self.classifier = Classifier(classifier_name,
                                     classifier,
                                     classification_data.train_dataset,
                                     classification_data.train_labels,
                                     classification_data.test_dataset,
                                     classification_data.test_labels)

    def run(self):
        # import pdb; pdb.set_trace()
        # print("RUN %s" % (threading.current_thread()))
        self.classifier.run_classifier()
        # print ("Starting " + self.classifier_name)
        # print ("Exiting " + self.classifier_name)


class ParallelClassifiersRunner():
    def __init__(self, labels_list, test_labels, classification_data):
        self.labels_list = labels_list
        self.test_labels = test_labels
        self.classification_data = classification_data

    def run(self):
        # RUNNING ML CLASSIFIERS
        print("RUNNING ML CLASSIFIERS")

        # Calculate precision for simply guessing the most popular reference for all the entries
        naive_monkey_ratio = self.labels_list.count(max(set(self.labels_list), key=self.labels_list.count)) / len(self.labels_list)
        naive_monkey_successful_guesses = int(naive_monkey_ratio * len(self.test_labels))
        print("Dataset consists of %s instances, %s of them belong to the most popular label" % (len(self.labels_list),
                                                               self.labels_list.count(max(set(self.labels_list), key=self.labels_list.count))))
        print("Therefore, a naive classifier should get right %s out of %s" % (naive_monkey_successful_guesses, len(self.test_labels)))


        # Start a thread for each classifier
        threads = []

        classifiers_names = list(CLASSIFIERS.keys())
        shuffle(classifiers_names)
        for classifier_name in classifiers_names[:MAX_PARALLEL_CLASSIFIERS]:
            runner = ClassifierRunner(classifier_name,
                                               CLASSIFIERS[classifier_name],
                                               self.classification_data)
            runner.start()
            threads.append(runner)

        # Wait for all threads to complete
        print("Wait for all threads to complete")
        for t in threads:
            t.join()
        print("Exiting Main Thread")
