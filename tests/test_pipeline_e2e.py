import joblib


MODEL_PATH = "models/latest/model.pkl"


def test_corrected_pipeline_predicts_known_complaints():
    model = joblib.load(MODEL_PATH)
    texts_and_labels = [
        (
            "I was charged twice for the same checking account transaction "
            "and the bank refused a refund.",
            "Checking or savings account",
        ),
        ("My mortgage payment was applied to the wrong account.", "Mortgage"),
        (
            "My student loan servicer changed and my balance is incorrect.",
            "Student loan",
        ),
    ]

    predictions = model.predict([text for text, _ in texts_and_labels])

    assert list(predictions) == [label for _, label in texts_and_labels]
