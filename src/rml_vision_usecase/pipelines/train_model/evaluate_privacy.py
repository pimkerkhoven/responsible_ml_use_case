from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def evaluate_privacy(X, sex, model):
    predict_X = X.copy()

    X["target"] = model.predict(predict_X)
    y = sex

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = LogisticRegression(max_iter=10_000)
    clf.fit(X_train, y_train)

    return 2 * abs(0.5 - clf.score(X_test, y_test))
