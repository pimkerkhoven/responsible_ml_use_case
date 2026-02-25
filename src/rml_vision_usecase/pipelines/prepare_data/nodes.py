"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 1.0.0
"""

import logging
import math
import sys
import tempfile

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def download_data(year, state):
    with tempfile.TemporaryDirectory() as temp_dir:
        # logger = logging.getLogger(__name__)
        # logger.info(f"Temporary directory created at: {temp_dir}")

        data_source = ACSDataSource(
            survey_year=f"{year}",
            horizon="1-Year",
            survey="person",
            root_dir=temp_dir,
        )
        acs_data = data_source.get_data(states=[state], download=True)
        data, y, _ = ACSIncome.df_to_pandas(acs_data)

        # add target label to data
        data["TARGET"] = y["PINCP"]

        return data


def make_data_splits(data_2014, data_2015):
    data = pd.concat([data_2014, data_2015], ignore_index=True)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    return train_data, val_data, test_data


def create_occ_to_sal(occ_to_soc, soc_to_sal, state):
    soc_to_sal = soc_to_sal[soc_to_sal["ST"] == state]

    result = {}

    for _, row in occ_to_soc.iterrows():
        occ_code = row["2013-2017 ACS/PRCS OCC code"]
        if occ_code == 0.0 or math.isnan(occ_code):
            continue

        soc_code = row["2013-2017 ACS/PRCS OCCSOC code"]
        soc_code = soc_code[:2] + "-" + soc_code[2:]

        mapping = soc_to_sal[soc_to_sal["OCC_CODE"] == soc_code][["A_MEAN", "H_MEAN"]]
        index = -1
        while len(mapping) == 0 and index >= -4:  # noqa: PLR2004
            soc_code = list(soc_code)
            soc_code[index] = "."
            soc_code = "".join(soc_code)

            mapping = soc_to_sal[
                soc_to_sal["OCC_CODE"].str.contains(soc_code, regex=True)
            ][["A_MEAN", "H_MEAN"]]
            index -= 1

        salary = np.nan
        hour_rate = np.nan
        if len(mapping) > 0:
            potential_salary = mapping["A_MEAN"].iloc[0].replace(",", "")
            if not potential_salary == "*":
                salary = int(potential_salary)

            potential_hour_rate = mapping["H_MEAN"].iloc[0].replace(",", "")
            if not potential_hour_rate == "*":
                hour_rate = float(potential_hour_rate)

        result[occ_code] = salary, hour_rate

    return result


# TODO: move to train pipeline as a step
# def augment_with_salary_data(data, occ_to_sal, augment_features):
#     if augment_features is None:
#         return data
#
#     if "MEAN_SALARY" in augment_features:
#         data["MEAN_SALARY"] = (
#             data["OCCP"]
#             .astype(int)
#             .map(lambda x: occ_to_sal[x][0] if x in occ_to_sal else np.nan)
#         )
#
#     if "HOUR_RATE" in augment_features:
#         data["HOUR_RATE"] = (
#             data["OCCP"]
#             .astype(int)
#             .map(lambda x: occ_to_sal[x][1] if x in occ_to_sal else np.nan)
#             * data["WKHP"]
#         )
#
#     return data
