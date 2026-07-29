from pathlib import Path

import joblib
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)


# Locate the saved model
MODEL_PATH = Path(__file__).with_name(
    "student_outcome_model.joblib"
)


@st.cache_resource
def load_model_bundle():
    """Load the trained model and supporting information."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model_bundle()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


model = model_bundle["model"]
metrics = model_bundle["metrics"]


st.title("🎓 Student Outcome Predictor")

st.write(
    """
    This application uses Semester 1, demographic and financial
    information to predict whether a student is likely to Dropout,
    remain Enrolled or Graduate.
    """
)

st.success("Trained model loaded successfully.")

st.subheader("Final Model Performance")

column_1, column_2, column_3 = st.columns(3)

column_1.metric(
    "Test Accuracy",
    f"{metrics['test_accuracy']:.3f}"
)

column_2.metric(
    "Macro F1",
    f"{metrics['test_macro_f1']:.3f}"
)

column_3.metric(
    "Dropout Recall",
    f"{metrics['test_dropout_recall']:.3f}"
)