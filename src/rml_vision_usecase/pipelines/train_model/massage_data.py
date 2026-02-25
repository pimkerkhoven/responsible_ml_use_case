import pandas as pd


def massage_data(X, y, sensitive_feature, f_perc, ranker):
    y = y.copy()
    num_massages = round(len(y) * f_perc)
    # print("N_massages:", num_massages)

    # ranker = LogisticRegression(max_iter=10_000).fit(X, y)
    probas = ranker.predict_proba(X)

    df = pd.DataFrame()
    df["TARGET"] = y
    df["ranking"] = probas[:, 1]
    df["SENSITIVE"] = sensitive_feature

    df = df.sort_values("ranking")

    # print(df)

    true_male_samples_indices = (
        df[(df["TARGET"] == True) & (df["SENSITIVE"] == 1.0)].head(num_massages).index  # noqa: E712
    )
    false_female_samples_indices = (
        df[(df["TARGET"] == False) & (df["SENSITIVE"] == 2.0)].tail(num_massages).index  # noqa: E712, PLR2004
    )

    y[true_male_samples_indices] = False
    y[false_female_samples_indices] = True

    return y
