from utils import parsing_utils, ml_utils
import Reference

# CONSTANTS
BETYOSEF_FILENAME = "BetYosefData/BetYosef-AllText.txt"
DEMO_FACTOR = 1 # Between 0 to 1
BAG_SIZE = 5 # Between 0 to 20
TESTSET_FACTOR = 0.85 # Between 0 to 1
MINIMUM_LABEL_FREQUENCY_PERCENTAGE = 0.1 # Between 0 to 1
MINIMUM_LABEL_FREQUENCY = 100 # Between 0 to infinity


def main():
    # PREPARING THE DATA
    data = parsing_utils.get_raw_data(BETYOSEF_FILENAME)

    # ANALYSE DATA INTO DATASET MATRICES
    train_dataset, train_labels, train_references, test_dataset, test_labels, test_references, labels_list, acknowledged_sources = parsing_utils.parse_data_to_matrices(
                                                data[:int(len(data) * DEMO_FACTOR)],
                                                MINIMUM_LABEL_FREQUENCY_PERCENTAGE,
                                                MINIMUM_LABEL_FREQUENCY,
                                                BAG_SIZE,
                                                TESTSET_FACTOR,
                                                Reference.REFERENCES_BLACKLIST
                                                )

    # CREATE CLASSIFICATION DATA OBJECT FOR CLASSIFIERS
    classification_data = ml_utils.ClassificationData(train_dataset, train_labels, train_references, test_dataset, test_labels, test_references)

    # RUNNING ALL THE CLASSIFIERS
    classifiers_runner = ml_utils.ParallelClassifiersRunner(labels_list, test_labels, classification_data, acknowledged_sources)
    classifiers_runner.run()

if "__main__" == __name__:
    main()
