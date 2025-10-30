import pandas as pd
import numpy as np
import pandas.api.types as ptypes


from sklearn.base import BaseEstimator

QIS = ["COW", "WKHP", "SCHL", "MAR", "OCCP", "POBP"]

POSSIBLE_AGES = list(range(1, 100))
POSSIBLE_WKHP = list(range(1, 100))
POSSIBLE_OCCP = list(range(1, 10_000))
POSSIBLE_POBP = list(range(1, 600))


# Copyright 2024 Judith Sainz-Pardo Diaz
def apply_hierarchy(data, hierarchies, level):
    num_level = len(hierarchies.keys()) - 1
    if level > num_level:
        raise ValueError("Error, invalid hierarchy level")
    # if not isinstance(hierarchies[level], pd.Series):
    #     hierarchies[level] = pd.Series(hierarchies[level])
    # if not isinstance(hierarchies[level - 1], pd.Series):
    #     hierarchies[level - 1] = pd.Series(hierarchies[level - 1])

    # create dict mapping values on level - 1 to level 1
    hierarchy_map = {}
    previous_level = hierarchies[level - 1]
    new_level = hierarchies[level]

    if len(previous_level) != len(new_level):
        print(previous_level, new_level)
        raise ValueError("Levels should have same length")

    for i in range(len(previous_level)):
        hierarchy_map[previous_level[i]] = new_level[i]

    # print(hierarchy_map)

    # pos = []
    # for elem in data:
    #     pos.append(np.where(hierarchies[level - 1].values == elem)[0][0])

    # data_anon = data.z
    # print(data)
    return data.map(hierarchy_map)


# Copyright 2024 Judith Sainz-Pardo Diaz
def generate_intervals(quasi_ident, inf, sup, step):
    values = np.arange(inf, sup + 1, step)
    interval = []
    for num in quasi_ident:
        lower = np.searchsorted(values, num)
        if lower == 0:
            lower = 1
        interval.append(f"[{values[lower - 1]}, {values[lower]})")

    return interval


HIERARCHIES = {
    "COW": {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [
            "company employee",
            "company employee",
            "government employee",
            "government employee",
            "government employee",
            "self-employed",
            "self-employed",
            "unpaid",
            "unpaid",
        ],
        2: [
            "in organization",
            "in organization",
            "in organization",
            "in organization",
            "in organization",
            "not in organization",
            "not in organization",
            "not in organization",
            "not in organization",
        ],
        3: ["*", "*", "*", "*", "*", "*", "*", "*", "*"],
    },
    "WKHP": {
        0: POSSIBLE_WKHP,
        1: generate_intervals(POSSIBLE_AGES, 0, 100, 2),
        2: generate_intervals(POSSIBLE_AGES, 0, 100, 5),
        3: generate_intervals(POSSIBLE_AGES, 0, 100, 10),
        4: generate_intervals(POSSIBLE_AGES, 0, 100, 25),
        5: ["*"] * len(POSSIBLE_WKHP),
    },
    "SCHL": {
        0: list(range(1, 25)),
        1: ["NO SCHOOL"]
        + ["PRESCHOOL"] * 2
        + ["PRIMARY SCHOOL"] * 14
        + ["COLLEGE - NO DEGREE"] * 2
        + ["COLLEGE - DEGREE"] * 5,
        2: ["NO SCHOOL"] + ["PRECOLLEGE"] * 16 + ["COLLEGE"] * 7,
        3: ["*"] * 24,
    },
    "MAR": {
        0: list(range(1, 6)),
        1: ["TOGETHER", "ALONE", "ALONE", "ALONE", "NEVER MARRIED"],
        2: ["*"] * 5,
    },
    "OCCP": {
        0: POSSIBLE_OCCP,
        1: generate_intervals(POSSIBLE_OCCP, 0, 10_000, 10),
        2: generate_intervals(POSSIBLE_OCCP, 0, 10_000, 100),
        3: generate_intervals(POSSIBLE_OCCP, 0, 10_000, 1000),
        4: ["*"] * len(POSSIBLE_OCCP),
    },
    "POBP": {
        0: POSSIBLE_POBP,
        1: generate_intervals(POSSIBLE_POBP, 0, 600, 100),
        2: ["*"] * len(POSSIBLE_POBP),
    },
}


class anonymizeData(BaseEstimator):
    def __init__(self, transformation):
        # self.k = k
        self.transformation = transformation
        self.hierarchies = HIERARCHIES
        self.QIS = QIS

    def transform(self, X, y=None):
        anon_X = X.copy()

        for i, QI in enumerate(QIS):
            for j in range(1, self.transformation[i] + 1):
                anon_X[QI] = apply_hierarchy(anon_X[QI], self.hierarchies[QI], j)

        if not ptypes.is_numeric_dtype(anon_X["WKHP"]):
            anon_X.rename(columns={"WKHP": "WKH_GEN"}, inplace=True)
            self.QIS = ["COW", "WKH_GEN", "SCHL", "MAR", "OCCP", "POBP"]

        return anon_X

    def fit(self, X, y=None):
        # data_anon = k_anonymity(
        #     X,
        #     ident=[],
        #     quasi_ident=QIS,
        #     k=self.k,
        #     supp_level=0,
        #     hierarchies=self.hierarchies
        # )

        # self.transformation = utils.get_transformation(data_anon, QIS, self.hierarchies)

        # print("FOUND TRANSFORMATION:", self.transformation)

        return self
