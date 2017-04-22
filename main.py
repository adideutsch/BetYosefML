from utils import parsing_utils, ml_utils

# CONSTS
BETYOSEF_FILENAME = "BetYosefData/BetYosef-AllText.txt"
DEMO_FACTOR = 1 # Between 0 to 1
BAG_SIZE = 5 # Between 0 to 20
TESTSET_FACTOR = 0.85 # Between 0 to 1
MINIMUM_LABEL_FREQUENCY_PERCENTAGE = 1 # Between 0 to 1
MINIMUM_LABEL_FREQUENCY = 50 # Between 0 to infinity
"""
first five are blacklisted as they appear so many times they tend to "distract" the classifiers
"""
REFERENCES_BLACKLIST = ["ברכות",
                        "חולין",
                        "פסחים",
                        "לאוין",
                        "עירובין",
                        "שו\"ת",
                        "הי\"א",
                        "סי\"ב",
                        "פט\"ו",
                        "בד\"ה",
                        "סי\"ד",
                        "ני\"ז",
                        "וש\"נ",
                        "סי\"ג",
                        "פי\"ד",
                        "הכ\"ו",
                        "ומיהו",
                        "וסי\'",
                        "הדשן",
                        "מהדו\'",
                        "סי\"א",
                        "דאמר",
                        "דעים",
                        "בחי\'",
                        "הכ\"ה",
                        "דאמרינן",
                        "ההוא",
                        "פי\"ג",
                        "נכ\"ו",
                        "פי\"ב",
                        "ני\"ב",
                        "הכ\"ד",
                        "הכ\"א",
                        "סעיף",
                        "הכ\"ב",
                        "וד\"ה",
                        "הכ\"ג",
                        "פי\"א",
                        "הי\"ח",
                        "הי\"ט",
                        "הי\"ז",
                        "הט\"ו",
                        "ומ\"ש",
                        "הי\"ב",
                        "עבה\"ק",
                        "הי\"ג",
                        "הי\"ד",
                        "הט\"ז",
                        "קיז:",
                        "דפו\'",
                        "חו\"מ",
                        "קיד",
                        "קנא.",
                        "קטו",
                        "עיי\'",
                        "ני\"ט",
                        "תורת",
                        "קמז.",
                        "נט\"ו",
                        "וכתב",
                        "סוע\"ב",
                        "מאכ\"א",
                        "סוע\"א",
                        "דבור",
                        "וקצר",
                        "קיא.",
                        "קיג.",
                        "קיב.",
                        "קיב:",
                        "קיא:",
                        "קיד.",
                        "פי\"ז"
                        ]


def main():
    # PREPARING THE DATA
    data = parsing_utils.get_raw_data(BETYOSEF_FILENAME)

    train_dataset, train_labels, test_dataset, test_labels, labels_list = parsing_utils.parse_data_to_matrices(
                                                data[:int(len(data) * DEMO_FACTOR)],
                                                MINIMUM_LABEL_FREQUENCY_PERCENTAGE,
                                                MINIMUM_LABEL_FREQUENCY,
                                                BAG_SIZE,
                                                TESTSET_FACTOR,
                                                REFERENCES_BLACKLIST
                                                )
    classification_data = ml_utils.ClassificationData(train_dataset, train_labels, test_dataset, test_labels)

    # RUNNING ALL THE CLASSIFIERS
    classifiers_runner = ml_utils.ParallelClassifiersRunner(labels_list, test_labels, classification_data)
    classifiers_runner.run()

if "__main__" == __name__:
    main()
