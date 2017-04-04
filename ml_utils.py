import threading

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

CLASSIFIERS = {
    "Nearest Neighbors" : KNeighborsClassifier(3),
    "Linear SVM" : SVC(kernel="linear", C=0.025),
    "RBF SVM" : SVC(gamma=2, C=1),
    # "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree" : DecisionTreeClassifier(max_depth=5),
    "Random Forest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net" : MLPClassifier(alpha=1),
    "AdaBoost" : AdaBoostClassifier(),
    "Naive Bayes" : GaussianNB(),
    "QDA" : QuadraticDiscriminantAnalysis()
    }

def run_classifier(classifier_name, classifier, train_dataset, train_labels, test_dataset, expected):
    print("Running %s classifier" % (classifier_name))

    classifier.fit(train_dataset, train_labels)
    predicted = classifier.predict(test_dataset)

    # print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    counter = 0
    initial_index = expected.index[0]
    for index, prediction in enumerate(predicted):
        if prediction == expected[initial_index + index]:
            counter += 1
    print("%s: %s out of %s" % (classifier_name, counter, len(predicted)))

class ClassifierRunner (threading.Thread):
   def __init__(self, classifier_name, classifier, train_dataset, train_labels, test_dataset, expected):
       threading.Thread.__init__(self)
       self.classifier_name = classifier_name
       self.classifier = classifier
       self.train_dataset = train_dataset
       self.train_labels = train_labels
       self.test_dataset = test_dataset
       self.expected = expected

   def run(self):
      # print ("Starting " + self.classifier_name)
      run_classifier(self.classifier_name, self.classifier, self.train_dataset, self.train_labels, self.test_dataset, self.expected)
      # print ("Exiting " + self.classifier_name)
