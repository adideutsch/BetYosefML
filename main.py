import pandas, re
from sklearn import datasets, metrics

import parsing_utils, ml_utils

# CONSTS
BETYOSEF_FILENAME = "BetYosefData/BetYosef-AllText.txt"
DEMO_FACTOR = 0.25
BAG_SIZE = 3
TESTSET_FACTOR = 0.85
MINIMUM_LABEL_FREQUENCY_PERCENTAGE = 0.1
MINIMUM_LABEL_FREQUENCY = 50


def main():
    # GETTING THE DATA
    data = parsing_utils.get_raw_data(BETYOSEF_FILENAME)
    print("Reading... %d characters in data" % len(data))

    train_dataset, train_labels, test_dataset, test_labels, labels_list = parsing_utils.parse_data_to_matrices(
                                                data[:int(len(data) * DEMO_FACTOR)],
                                                MINIMUM_LABEL_FREQUENCY_PERCENTAGE,
                                                MINIMUM_LABEL_FREQUENCY,
                                                BAG_SIZE,
                                                TESTSET_FACTOR,
                                                )

    # RUNNING ML CLASSIFIERS
    print("RUNNING ML CLASSIFIERS")

    # Calculate precision for simply guessing the most popular reference for all the entries
    naive_monkey_ratio = labels_list.count(max(set(labels_list), key=labels_list.count)) / len(labels_list)
    naive_monkey_successful_guesses = int(naive_monkey_ratio * len(test_labels))
    print("\"MONKEY\" Classifier would get %s out of %s" % (naive_monkey_successful_guesses, len(test_labels)))

    # Start a thread for each classifier
    threads = []
    for classifier_name in ml_utils.CLASSIFIERS.keys():
        runner = ml_utils.ClassifierRunner(classifier_name, ml_utils.CLASSIFIERS[classifier_name], train_dataset, train_labels, test_dataset, test_labels)
        runner.start()
        threads.append(runner)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")

if "__main__" == __name__:
    main()