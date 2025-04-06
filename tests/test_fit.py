from flyingsquid.label_model import LabelModel
import numpy as np


def test_fit_and_predict():
    """
    Test whether fit and predict_proba don't raise any exceptions
    """

    n_lfs = 5
    n_classes = 2
    n_datapoints = 100
    label_matrix = np.random.choice([-1, 0, 1], size=[n_datapoints, n_lfs])

    label_model = LabelModel(n_lfs)
    label_model.fit(label_matrix)
    preds = label_model.predict_proba(label_matrix)

    assert preds.shape == (n_datapoints, n_classes)
