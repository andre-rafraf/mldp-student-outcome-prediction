from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Outcome Predictor",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("student_outcome_model.joblib")


@st.cache_resource
def load_model_bundle():
    """Load the trained pipeline and the information saved with it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def clamp(value, minimum, maximum):
    """Keep a saved default inside the limits accepted by an input widget."""
    return min(max(value, minimum), maximum)


# -----------------------------------------------------------------------------
# Load and validate the saved model bundle
# -----------------------------------------------------------------------------
try:
    model_bundle = load_model_bundle()

    required_keys = {
        "model",
        "feature_columns",
        "feature_defaults",
        "metrics",
    }
    missing_keys = required_keys.difference(model_bundle)
    if missing_keys:
        raise KeyError(
            "The model bundle is missing: " + ", ".join(sorted(missing_keys))
        )
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]
feature_defaults = model_bundle["feature_defaults"]
metrics = model_bundle["metrics"]


# -----------------------------------------------------------------------------
# Header and model information
# -----------------------------------------------------------------------------
st.title("Student Outcome Predictor")
st.write(
    "This decision-support application uses information available at the end "
    "of Semester 1 to estimate whether a student is likely to **Dropout**, "
    "remain **Enrolled**, or **Graduate**."
)
st.caption(
    "The result is a model estimate and should support—not replace—professional "
    "academic judgement."
)

with st.expander("View final model performance"):
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
    metric_2.metric("Macro F1", f"{metrics['test_macro_f1']:.3f}")
    metric_3.metric("Dropout Recall", f"{metrics['test_dropout_recall']:.3f}")

st.info(
    "Complete the fields below and select **Predict student outcome**. "
    "Features not shown in the form use typical values from the training data."
)


# -----------------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------------
yes_no = {"No": 0, "Yes": 1}
gender_codes = {"Female": 0, "Male": 1}

with st.form("student_prediction_form"):
    st.subheader("1. Student background")
    background_1, background_2, background_3 = st.columns(3)

    with background_1:
        age = st.number_input(
            "Age at enrollment",
            min_value=17,
            max_value=70,
            value=int(
                clamp(float(feature_defaults.get("Age at enrollment", 20)), 17, 70)
            ),
            step=1,
        )

        gender_default = int(feature_defaults.get("Gender", 0))
        gender_label = st.selectbox(
            "Gender",
            options=list(gender_codes),
            index=1 if gender_default == 1 else 0,
        )

        displaced_default = int(feature_defaults.get("Displaced", 1))
        displaced_label = st.selectbox(
            "Displaced student",
            options=list(yes_no),
            index=1 if displaced_default == 1 else 0,
        )

    with background_2:
        admission_grade = st.number_input(
            "Admission grade (95–190)",
            min_value=95.0,
            max_value=190.0,
            value=float(
                clamp(float(feature_defaults.get("Admission grade", 126.0)), 95, 190)
            ),
            step=0.1,
            format="%.1f",
        )

        previous_grade = st.number_input(
            "Previous qualification grade (95–190)",
            min_value=95.0,
            max_value=190.0,
            value=float(
                clamp(
                    float(
                        feature_defaults.get(
                            "Previous qualification (grade)", 133.1
                        )
                    ),
                    95,
                    190,
                )
            ),
            step=0.1,
            format="%.1f",
        )

        international_default = int(feature_defaults.get("International", 0))
        international_label = st.selectbox(
            "International student",
            options=list(yes_no),
            index=1 if international_default == 1 else 0,
        )

    with background_3:
        debtor_default = int(feature_defaults.get("Debtor", 0))
        debtor_label = st.selectbox(
            "Has outstanding debt",
            options=list(yes_no),
            index=1 if debtor_default == 1 else 0,
        )

        tuition_default = int(feature_defaults.get("Tuition fees up to date", 1))
        tuition_label = st.selectbox(
            "Tuition fees up to date",
            options=list(yes_no),
            index=1 if tuition_default == 1 else 0,
        )

        scholarship_default = int(feature_defaults.get("Scholarship holder", 0))
        scholarship_label = st.selectbox(
            "Scholarship holder",
            options=list(yes_no),
            index=1 if scholarship_default == 1 else 0,
        )

    st.divider()
    st.subheader("2. Semester 1 academic performance")
    academic_1, academic_2, academic_3 = st.columns(3)

    with academic_1:
        credited = st.number_input(
            "Credited curricular units",
            min_value=0,
            max_value=20,
            value=int(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (credited)", 0
                        )
                    ),
                    0,
                    20,
                )
            ),
            step=1,
        )

        enrolled = st.number_input(
            "Enrolled curricular units",
            min_value=0,
            max_value=26,
            value=int(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (enrolled)", 6
                        )
                    ),
                    0,
                    26,
                )
            ),
            step=1,
        )

    with academic_2:
        evaluations = st.number_input(
            "Number of evaluations",
            min_value=0,
            max_value=45,
            value=int(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (evaluations)", 8
                        )
                    ),
                    0,
                    45,
                )
            ),
            step=1,
        )

        approved = st.number_input(
            "Approved curricular units",
            min_value=0,
            max_value=26,
            value=int(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (approved)", 5
                        )
                    ),
                    0,
                    26,
                )
            ),
            step=1,
        )

    with academic_3:
        semester_grade = st.number_input(
            "Average Semester 1 grade (0–20)",
            min_value=0.0,
            max_value=20.0,
            value=float(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (grade)", 12.32
                        )
                    ),
                    0,
                    20,
                )
            ),
            step=0.1,
            format="%.2f",
        )

        without_evaluations = st.number_input(
            "Units without evaluations",
            min_value=0,
            max_value=12,
            value=int(
                clamp(
                    float(
                        feature_defaults.get(
                            "Curricular units 1st sem (without evaluations)", 0
                        )
                    ),
                    0,
                    12,
                )
            ),
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict student outcome",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Validate inputs and produce the prediction
# -----------------------------------------------------------------------------
if submitted:
    validation_errors = []

    if approved > enrolled:
        validation_errors.append(
            "Approved curricular units cannot be greater than enrolled units."
        )

    if approved > evaluations:
        validation_errors.append(
            "Approved curricular units cannot be greater than the number of evaluations."
        )

    if credited > enrolled:
        validation_errors.append(
            "Credited curricular units cannot be greater than enrolled units."
        )

    if without_evaluations > enrolled:
        validation_errors.append(
            "Units without evaluations cannot be greater than enrolled units."
        )

    if enrolled == 0 and any(
        [
            credited > 0,
            evaluations > 0,
            approved > 0,
            without_evaluations > 0,
        ]
    ):
        validation_errors.append(
            "All Semester 1 unit values must be 0 when enrolled units are 0."
        )

    if validation_errors:
        st.session_state.pop("prediction_result", None)

        for message in validation_errors:
            st.error(message)

    else:
        # Begin with the training-data defaults for every feature, then replace
        # the values collected through the user interface.
        input_values = feature_defaults.copy()
        input_values.update(
            {
                "Age at enrollment": int(age),
                "Gender": gender_codes[gender_label],
                "Displaced": yes_no[displaced_label],
                "Admission grade": float(admission_grade),
                "Previous qualification (grade)": float(previous_grade),
                "International": yes_no[international_label],
                "Debtor": yes_no[debtor_label],
                "Tuition fees up to date": yes_no[tuition_label],
                "Scholarship holder": yes_no[scholarship_label],
                "Curricular units 1st sem (credited)": int(credited),
                "Curricular units 1st sem (enrolled)": int(enrolled),
                "Curricular units 1st sem (evaluations)": int(evaluations),
                "Curricular units 1st sem (approved)": int(approved),
                "Curricular units 1st sem (grade)": float(semester_grade),
                "Curricular units 1st sem (without evaluations)": int(
                    without_evaluations
                ),
            }
        )

        input_values["Sem1 Pass Rate"] = (
            approved / enrolled if enrolled > 0 else 0.0
        )
        input_values["Sem1 Completion Gap"] = enrolled - approved

        input_frame = pd.DataFrame([input_values]).reindex(columns=feature_columns)

        try:
            prediction = str(model.predict(input_frame)[0])
            probabilities = model.predict_proba(input_frame)[0]
            probability_by_class = {
                str(class_name): float(probability)
                for class_name, probability in zip(model.classes_, probabilities)
            }

            st.session_state["prediction_result"] = {
                "prediction": prediction,
                "probabilities": probability_by_class,
                "pass_rate": float(input_values["Sem1 Pass Rate"]),
                "completion_gap": int(input_values["Sem1 Completion Gap"]),
            }
        except Exception as error:
            st.session_state.pop("prediction_result", None)
            st.error("A prediction could not be generated.")
            st.exception(error)


# -----------------------------------------------------------------------------
# Display prediction and probabilities
# -----------------------------------------------------------------------------
result = st.session_state.get("prediction_result")

if result:
    st.divider()
    st.subheader("3. Prediction result")

    predicted_outcome = result["prediction"]
    confidence = result["probabilities"][predicted_outcome]

    if predicted_outcome == "Dropout":
        st.error(
            f"Predicted outcome: **Dropout** — model confidence: "
            f"**{confidence:.1%}**"
        )
        st.write(
            "This profile may benefit from early academic advising, financial "
            "support, and closer progress monitoring."
        )
    elif predicted_outcome == "Enrolled":
        st.warning(
            f"Predicted outcome: **Enrolled** — model confidence: "
            f"**{confidence:.1%}**"
        )
        st.write(
            "The student is predicted to remain enrolled. Continued monitoring "
            "may help support progression toward graduation."
        )
    else:
        st.success(
            f"Predicted outcome: **Graduate** — model confidence: "
            f"**{confidence:.1%}**"
        )
        st.write(
            "The profile indicates a positive likelihood of graduation while "
            "continued academic support remains valuable."
        )

    summary_1, summary_2 = st.columns(2)
    summary_1.metric("Semester 1 pass rate", f"{result['pass_rate']:.1%}")
    summary_2.metric("Completion gap", result["completion_gap"])

    st.markdown("#### Outcome probabilities")
    ordered_outcomes = ["Dropout", "Enrolled", "Graduate"]
    for outcome in ordered_outcomes:
        probability = result["probabilities"].get(outcome, 0.0)
        st.write(f"**{outcome}: {probability:.1%}**")
        st.progress(probability)

    st.caption(
        "A probability is not a guarantee. The model should be used as an "
        "early-support indicator rather than as the sole basis for decisions."
    )

with st.expander("How this app prepares the model input"):
    st.write(
        "The app uses the values entered above and fills the remaining model "
        "features with typical training-data values saved in the model bundle. "
        "It then calculates Semester 1 pass rate and completion gap before "
        "passing one complete row to the trained scikit-learn pipeline."
    )