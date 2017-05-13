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
MAX_SUCCESS_EXAMPLES = 5
MAX_FAILURE_EXAMPLES = 5

CLASSIFIERS = {
    # "Nearest Neighbors" : KNeighborsClassifier(n_neighbors = 250),
    # "Linear SVM" : SVC(kernel="linear", C=0.025),
    # "RBF SVM" : SVC(gamma=2, C=1),
    # "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree" : DecisionTreeClassifier(max_depth=500),
    "Random Forest" : RandomForestClassifier(max_depth=100, n_estimators=25, max_features='auto'),
    "Neural Net" : MLPClassifier(),
    # "AdaBoost" : AdaBoostClassifier(n_estimators=250),
    # "Naive Bayes" : GaussianNB(),
    # "QDA" : QuadraticDiscriminantAnalysis()
    }


class Classifier():
    def __init__(self, name, classifier, train_dataset, train_labels, test_dataset, expected, train_references, test_references, labels_list, results, acknowledged_sources):
        self.name = name
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.test_dataset = test_dataset
        self.expected = expected
        self.train_references = train_references
        self.test_references = test_references
        self.results = results
        self.successful = None
        self.successes = []
        self.failures = []
        self.labels_list = labels_list
        self.acknowledged_sources = acknowledged_sources

    def get_failures_by_label(self):
        report = ""
        sorted_labels = list(self.label_failures.keys())
        sorted_labels.sort(key=lambda label: len(self.label_failures[label][0]) + len(self.label_failures[label][1]), reverse=True)
        for label in sorted_labels:
            false_positives = self.label_failures[label][0]
            false_negatives = self.label_failures[label][1]
            report += "\nlabel %s:\nfalse positives: %3s, false negatives: %3s, successes: %3s\n" % (label, len(false_positives), len(false_negatives), len(self.label_successes[label]))
        return report

    def get_report(self):
        report = "Classification report for classifier %s (%s out of %s):" % (self.classifier, self.successful, len(self.predicted))
        report += "\n%s" % metrics.classification_report(self.expected, self.predicted)
        report += "\nConfusion matrix:\n%s" % metrics.confusion_matrix(self.expected, self.predicted)
        report += "\n%s" % (self.get_successes())
        report += "\n%s" % (self.get_failures())
        report += "\n%s" % (self.get_failures_by_label())
        return report

    def get_successful(self):
        return self.successful

    def get_successes(self):
        successes_report = ""
        for success in self.successes[:MAX_SUCCESS_EXAMPLES]:
            successes_report += "\nIdentified \"%s\" successfully as %s (%s)" % (" ".join(success.get_bag_of_words()), success.prediction, success.label)
        return successes_report

    def get_failures(self):
        failures_report = ""
        for failure in self.failures[:MAX_FAILURE_EXAMPLES]:
            failures_report += "\nIdentified \"%s\" as %s instead of %s" % (" ".join(failure.get_bag_of_words()), failure.prediction, failure.label)
        return failures_report

    def report(self):
        print(self.get_report())

    def run_classifier(self):
        print("Running %s classifier, fitting on train dataset" % (self.name))

        self.classifier.fit(self.train_dataset, self.train_labels)

        print("%s classifier predicting on test dataset" % (self.name))
        self.predicted = self.classifier.predict(self.test_dataset)

        # self.report()

        counter = 0
        initial_index = self.expected.index[0]
        for index, prediction in enumerate(self.predicted):
            reference = self.test_references[index]
            reference.prediction = self.acknowledged_sources[prediction]
            if prediction == self.expected[initial_index + index]:
                counter += 1
                self.successes.append(reference)
            else:
                self.failures.append(reference)

        self.successful = counter

        self.label_failures = {}
        self.label_successes = {}
        for label in self.acknowledged_sources:
            false_positives = []
            false_negatives = []
            for failure in self.failures:
                if failure.label == label:
                    false_negatives.append(failure)
                if failure.prediction == label:
                    false_positives.append(failure)
            successful_identifications = []
            for success in self.successes:
                if success.label == label:
                    successful_identifications.append(success)
            self.label_failures[label] = (false_positives, false_negatives)
            self.label_successes[label] = successful_identifications

        print("%s: %s out of %s" % (self.name, self.successful, len(self.predicted)))
        self.results[self.name] = self.get_report()


class ClassificationData():
    def __init__(self, train_dataset, train_labels, train_references, test_dataset, test_labels, test_references):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.train_references = train_references
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.test_references = test_references


class ClassifierRunner(threading.Thread):
    def __init__(self, classifier_name, classifier, classification_data, labels_list, results, acknowledged_sources):
        threading.Thread.__init__(self)
        self.classifier = Classifier(classifier_name,
                                     classifier,
                                     classification_data.train_dataset,
                                     classification_data.train_labels,
                                     classification_data.test_dataset,
                                     classification_data.test_labels,
                                     classification_data.train_references,
                                     classification_data.test_references,
                                     labels_list,
                                     results,
                                     acknowledged_sources)

    def run(self):
        # import pdb; pdb.set_trace()
        # print("RUN %s" % (threading.current_thread()))
        self.classifier.run_classifier()
        # print ("Starting " + self.classifier_name)
        # print ("Exiting " + self.classifier_name)


class ParallelClassifiersRunner():
    def __init__(self, labels_list, test_labels, classification_data, acknowledged_sources):
        self.labels_list = labels_list
        self.test_labels = test_labels
        self.classification_data = classification_data
        self.acknowledged_sources = acknowledged_sources

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
        results = {}
        classifiers_runners = {}

        classifiers_names = list(CLASSIFIERS.keys())
        shuffle(classifiers_names)
        for classifier_name in classifiers_names[:MAX_PARALLEL_CLASSIFIERS]:
            runner = ClassifierRunner(
                                        classifier_name,
                                        CLASSIFIERS[classifier_name],
                                        self.classification_data,
                                        self.labels_list,
                                        results,
                                        self.acknowledged_sources
                                     )
            runner.start()
            threads.append(runner)
            classifiers_runners[classifier_name] = runner

        # Wait for all threads to complete
        print("Wait for all threads to complete")
        for t in threads:
            t.join()
        print("Exiting Main Thread")
        print("FULL REPORT:")
        sorted_keys = list(results.keys())
        sorted_keys.sort(key=lambda classifier_name: classifiers_runners[classifier_name].classifier.get_successful(), reverse=True)
        for classifier_name in sorted_keys[:3]:
            print("\n#####\nClassifier: %s" % classifier_name)
            print("%s" % results[classifier_name])
