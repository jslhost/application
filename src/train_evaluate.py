from sklearn.metrics import confusion_matrix


def model_evaluation(model, X_train, y_train, X_test, y_test):
    model_score_test = model.score(X_test, y_test)
    model_score_train = model.score(X_train, y_train)
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))

    return model_score_train, model_score_test, conf_matrix
