from utils import parsing_utils, ml_utils

# CONSTS
BETYOSEF_FILENAME = "BetYosefData/BetYosef-AllText.txt"
DEMO_FACTOR = 1
BAG_SIZE = 10
TESTSET_FACTOR = 0.85
MINIMUM_LABEL_FREQUENCY_PERCENTAGE = 0.005
MINIMUM_LABEL_FREQUENCY = 100

def main():
    # GETTING THE DATA
    data = parsing_utils.get_raw_data(BETYOSEF_FILENAME)

    train_dataset, train_labels, test_dataset, test_labels, labels_list = parsing_utils.parse_data_to_matrices(
                                                data[:int(len(data) * DEMO_FACTOR)],
                                                MINIMUM_LABEL_FREQUENCY_PERCENTAGE,
                                                MINIMUM_LABEL_FREQUENCY,
                                                BAG_SIZE,
                                                TESTSET_FACTOR,
                                                )

    classification_data = ml_utils.ClassificationData(train_dataset, train_labels, test_dataset, test_labels)
    classifiers_runner = ml_utils.ParallelClassifiersRunner(labels_list, test_labels, classification_data)
    classifiers_runner.run()

if "__main__" == __name__:
    main()
