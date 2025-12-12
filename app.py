import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import altair as alt
import joblib
import time
from typing import Tuple
from math import ceil

# Page config
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>

.st-emotion-cache-pbi6hp {
    margin-bottom: -1rem;
    margin-top: -1rem;
}

.st-ch {
    height: 2rem;
    -webkit-box-align: center;
    align-items: center;
}

.st-emotion-cache-1fnxbr3 {
    height: 2rem;
}

.st-emotion-cache-pbi6hp p {
    margin-block-end: 0.5em;
    margin-block-start: 0.5em;
}

label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    font-size: 0.9rem !important;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stSlider"],
div[data-testid="stRadio"] {
    margin-bottom: 0rem !important;
}

.main .block-container {
    max-width: 900px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts() -> Tuple[object, object, object, object]:
    
    artifacts = joblib.load("model/model.joblib")

    # Expecting a dict with keys: 'model', 'preprocessor', 'scaler_interactions', 'interaction_creator'
    preprocessor = artifacts["preprocessor"]
    interaction_creator = artifacts["interaction_creator"]
    scaler_interactions = artifacts["scaler_interactions"]
    model = artifacts["model"]

    return preprocessor, interaction_creator, scaler_interactions, model

@st.cache_data
def load_data():
    return pd.read_csv("data/Exam_Score_Prediction.csv")

def create_interaction_features_production(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw input data with columns: study_hours, sleep_hours, class_attendance,
        sleep_quality, exam_difficulty, study_method

    Returns:
    --------
    pd.DataFrame : DataFrame with interaction features
    """
    interactions = pd.DataFrame(index=df_raw.index)

    # 1. study_hours × sleep_hours (Study-rest balance)
    interactions['study_x_sleep'] = df_raw['study_hours'] * df_raw['sleep_hours']

    # 2. study_hours × class_attendance (Self-study + class engagement)
    interactions['study_x_attendance'] = df_raw['study_hours'] * df_raw['class_attendance']

    # 3. study_hours² (Diminishing returns on studying)
    interactions['study_squared'] = df_raw['study_hours'] ** 2

    # 4. sleep_hours × sleep_quality (Sleep effectiveness)
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    sleep_quality_num = df_raw['sleep_quality'].map(sleep_quality_map)
    interactions['sleep_x_quality'] = df_raw['sleep_hours'] * sleep_quality_num

    # 5. study_hours × exam_difficulty (Preparation vs challenge)
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    difficulty_num = df_raw['exam_difficulty'].map(difficulty_map)
    interactions['study_x_difficulty'] = df_raw['study_hours'] * difficulty_num

    # 6. class_attendance × study_method (Learning style)
    method_map = {'self-study': 0, 'group study': 1, 'online videos': 2, 'coaching': 3, 'mixed': 4}
    method_num = df_raw['study_method'].map(method_map)
    interactions['attendance_x_method'] = df_raw['class_attendance'] * method_num

    return interactions


def categorical_avg_chart(data: pd.DataFrame, col: str, categories: list[str]):
    """
    Bar chart of average exam score by category for a categorical feature.
    Lets Altair auto-zoom the y-axis (zero=False) so small differences are visible.
    """
    if {col, "exam_score"}.issubset(data.columns) and not data.empty:
        df_mean = (
            data.groupby(col, as_index=False)["exam_score"]
            .mean()
            .rename(columns={"exam_score": "avg_score"})
        )

        if df_mean.empty:
            return None

        chart = (
            alt.Chart(df_mean)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{col}:N",
                    sort=categories,  # use your specified ordering
                    title=col.replace("_", " ").title(),
                ),
                y=alt.Y(
                    "avg_score:Q",
                    title="Average exam score (%)",
                    scale=alt.Scale(zero=False),  # auto-zoom; no forced zero baseline
                ),
                tooltip=[col, "avg_score"],
            )
            .properties(
                width=350,
                height=300,
            )
        )

        return chart

    return None


preprocessor, interaction_creator, scaler_interactions, model = load_artifacts()


NUMERIC_FEATURES = [
    ("age", "Age", 17, 24, 20),
    ("study_hours", "Hours studied per week", 0.08, 7.91, 4.0),
    ("class_attendance", "Attendance rate (%)", 40.6, 99.4, 70.0),
    ("sleep_hours", "Average sleep hours per night", 4.1, 9.9, 7.5)
]

CATEGORICAL_FEATURES = {
    "gender": ["male", "female", "other"],
    "course": ["bca", "ba", "b.sc", "b.com", "bba", "diploma", "b.tech"],
    "internet_access": ["yes", "no"],
    "sleep_quality": ["poor", "average", "good"],
    "study_method": ["self-study", "online videos", "coaching", "group study", "mixed"],
    "facility_rating": ["low", "medium", "high"],
    "exam_difficulty": ["easy", "moderate", "hard"]
}


tab_predict, tab_explore = st.tabs(["🔮 Predict Score", "📊 Explore the Data"])

with tab_predict:

    st.header("📚 Student Exam Score Predictor")
    st.write("Fill out the form in the sidebar with the student's information to estimate their exam score.")

    numeric_inputs: dict[str, float] = {}
    categorical_inputs: dict[str, str] = {}
    submitted = False

    with st.sidebar:
        st.markdown("### Student input features")

        with st.form("prediction_form_sidebar"):
            st.markdown("**Numeric features**")
            for feat_name, label, min_val, max_val, default_val in NUMERIC_FEATURES:
                numeric_inputs[feat_name] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                )

            st.markdown("---")
            st.markdown("**Categorical features**")
            for feat_name, categories in CATEGORICAL_FEATURES.items():
                categorical_inputs[feat_name] = st.selectbox(
                    feat_name.replace("_", " ").title(),
                    categories,
                )

            submitted = st.form_submit_button(
                "Predict exam score",
                type="primary",
                use_container_width=True,
            )

    st.markdown("### Prediction result")
    result_container = st.container()

    if submitted:

        input_data = {**numeric_inputs, **categorical_inputs}
        input_df = pd.DataFrame([input_data])

        X_base = preprocessor.transform(input_df)

        if hasattr(interaction_creator, "transform"):
            X_int = interaction_creator.transform(input_df)
        else:
            X_int = interaction_creator(input_df)

        X_int_scaled = scaler_interactions.transform(X_int)

        X_full = np.hstack([X_base, X_int_scaled])

        y_pred = model.predict(X_full)[0]
        score = float(y_pred)

        if score < 50:
            category = "At risk of failing"
            advice = "Consider increasing study hours, improving sleep, or seeking extra academic support."
        elif score < 65:
            category = "Below average"
            advice = "More consistent studying and better rest could noticeably improve performance."
        elif score < 80:
            category = "Average"
            advice = "You seem on track, but there’s room to strengthen study habits for a higher score."
        else:
            category = "Strong performance expected"
            advice = "Your current habits look supportive of good performance. Keep it up!"

        with result_container:
            c1, c2 = st.columns(2)
            c1.metric("Predicted score", f"{score:.1f} %")
            c2.metric("Performance category", category)

            st.progress(max(0.0, min(1.0, score / 100.0)))
            st.info(advice)

            st.caption(
                "This estimate is generated by a trained machine learning model and should be "
                "interpreted as a probabilistic forecast, not a guarantee."
            )
    else:
        with result_container:
            st.info("Adjust the inputs in the sidebar and click **Predict exam score** to see the result here.")



with tab_explore:
    st.header("Explore the training data")

    try:
        data = load_data()
    except FileNotFoundError:
        st.warning(
            "No data file found at `data/student_scores_sample.csv`.\n\n"
            "Add a sample of your training data there to enable this tab."
        )
    else:
        # Top metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Number of students", f"{len(data):,}")

        c5.metric("Median age", f"{data['age'].median():.1f}")
        c2.metric("Average exam score", f"{data['exam_score'].mean():.1f} %")
        c3.metric("Avg. study hours/week", f"{data['study_hours'].mean():.1f}")
        c4.metric("Avg. attendance rate", f"{data['class_attendance'].mean():.1f} %")


        st.markdown("---")

        st.markdown("### Numerical Data")

        col_1a, col_2a, col_3a = st.columns(3)

        # Score distribution
        if "exam_score" in data.columns:
            with col_1a:
                st.markdown("#### Exam score distribution")
                
                hist = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        alt.X("exam_score:Q", bin=alt.Bin(maxbins=20), title="Exam score (%)"),
                        alt.Y("count():Q", title="Number of students"),
                    )
                    .properties(
                        width=450,
                        height=260,
                    )
                )

                st.altair_chart(hist, use_container_width=False)


        if {"age", "exam_score"}.issubset(data.columns):
            with col_2a:
                st.markdown("#### Age vs exam score")
                
                age_mean = (
                    data.groupby("age", as_index=False)["exam_score"]
                    .mean()
                    .rename(columns={"exam_score": "avg_score"})
                )

                # Optional: keep this while debugging
                # st.write("Debug - age_mean:", age_mean)

                if not age_mean.empty:
                    chart = (
                        alt.Chart(age_mean)
                        .mark_line(point=True, color="#4f9cf9")
                        .encode(
                            x=alt.X("age:O", title="Age"),
                            y=alt.Y(
                                "avg_score:Q",
                                title="Average score (%)",
                                # Let Altair choose a non-zero baseline that fits the data
                                scale=alt.Scale(zero=False),
                            ),
                            tooltip=["age", "avg_score"],
                        )
                        .properties(width=450, height=260)
                    )

                    st.altair_chart(chart, use_container_width=False)
                else:
                    st.info("No data available to plot age vs exam score.")

        # Study hours vs score
        if {"study_hours", "exam_score"}.issubset(data.columns):
            with col_3a:
                st.markdown("#### Study hours vs exam score")

                xcol = "study_hours"
                ycol = "exam_score"

                x_min = float(data[xcol].min())
                x_max = float(data[xcol].max())
                y_min = float(data[ycol].min())
                y_max = float(data[ycol].max())

                x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
                y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1

                chart = (
                    alt.Chart(data)
                    .mark_circle(size=20, opacity=0.8)
                    .encode(
                        x=alt.X(
                            xcol,
                            scale=alt.Scale(domain=[x_min - x_pad, x_max + x_pad]),
                            title="Study hours per week",
                        ),
                        y=alt.Y(
                            ycol,
                            scale=alt.Scale(domain=[y_min - y_pad, y_max + y_pad]),
                            title="Exam score (%)",
                        ),
                        tooltip=[xcol, ycol],
                    )
                    .properties(
                        width=450,
                        height=280,
                    )
                )

                st.altair_chart(chart, use_container_width=False)

        randomname, col_1b, col_2b, randomname2 = st.columns([1, 2, 2, 1])

        with randomname:
            st.empty()

        if {"class_attendance", "exam_score"}.issubset(data.columns):
            with col_1b:
                st.markdown("#### Attendance rate vs exam score")
                
                xcol = "class_attendance"
                ycol = "exam_score"

                x_min = float(data[xcol].min())
                x_max = float(data[xcol].max())
                y_min = float(data[ycol].min())
                y_max = float(data[ycol].max())

                x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
                y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1

                chart = (
                    alt.Chart(data)
                    .mark_circle(size=20, opacity=0.8)
                    .encode(
                        x=alt.X(
                            xcol,
                            scale=alt.Scale(domain=[x_min - x_pad, x_max + x_pad]),
                            title="Attendance Rate (%)",
                        ),
                        y=alt.Y(
                            ycol,
                            scale=alt.Scale(domain=[y_min - y_pad, y_max + y_pad]),
                            title="Exam score (%)",
                        ),
                        tooltip=[xcol, ycol],
                    )
                    .properties(
                        width=450,
                        height=280,
                    )
                )

                st.altair_chart(chart, use_container_width=False)

        if {"sleep_hours", "exam_score"}.issubset(data.columns):
            with col_2b:
                st.markdown("#### Hours slept vs exam score")
                
                sleep_mean = (
                    data.groupby("sleep_hours", as_index=False)["exam_score"]
                    .mean()
                    .rename(columns={"exam_score": "avg_score"})
                )

                if not sleep_mean.empty:
                    chart = (
                        alt.Chart(sleep_mean)
                        .mark_line(point=True, color="#4f9cf9")
                        .encode(
                            x=alt.X(
                                "sleep_hours:Q",
                                title="Hours slept per night",
                                scale=alt.Scale(zero=False),       # <-- important for zoomed view
                            ),
                            y=alt.Y(
                                "avg_score:Q",
                                title="Average exam score (%)",
                                scale=alt.Scale(zero=False),       # <-- auto-zoom around the data
                            ),
                            tooltip=["sleep_hours", "avg_score"],
                        )
                        .properties(width=450, height=260)
                    )

                    st.altair_chart(chart, use_container_width=False)
                else:
                    st.info("No data to plot.")

        with randomname2:
            st.empty()

        st.markdown("---")
        st.markdown("### Categorical Data")

        cat_1, cat_2 = st.columns(2)

        with cat_1:
            st.markdown("#### Avg. by gender")
            chart = categorical_avg_chart(data, "gender", CATEGORICAL_FEATURES["gender"])
            if chart: st.altair_chart(chart, use_container_width=True)
        with cat_2:
            st.markdown("#### Avg. by course")
            chart = categorical_avg_chart(data, "course", CATEGORICAL_FEATURES["course"])
            if chart: st.altair_chart(chart, use_container_width=True)

        cat_3, cat_4 = st.columns(2)

        with cat_3:
            st.markdown("#### Avg. (online v. in-person)")
            chart = categorical_avg_chart(data, "internet_access", CATEGORICAL_FEATURES["internet_access"])
            if chart: st.altair_chart(chart, use_container_width=True)
        with cat_4:
            st.markdown("#### Avg. by sleep quality")
            chart = categorical_avg_chart(data, "sleep_quality", CATEGORICAL_FEATURES["sleep_quality"])
            if chart: st.altair_chart(chart, use_container_width=True)

        cat_5, cat_6 = st.columns(2)

        with cat_5:
            st.markdown("#### Avg. by study method")
            chart = categorical_avg_chart(data, "study_method", CATEGORICAL_FEATURES["study_method"])
            if chart: st.altair_chart(chart, use_container_width=True)
        with cat_6:
            st.markdown("#### Avg. by facility rating")
            chart = categorical_avg_chart(data, "facility_rating", CATEGORICAL_FEATURES["facility_rating"])
            if chart: st.altair_chart(chart, use_container_width=True)

        randomcat, cat_7, randomcat2 = st.columns([1, 2, 1])

        with cat_7:
            st.markdown("#### Avg. by exam difficulty")
            chart = categorical_avg_chart(data, "exam_difficulty", CATEGORICAL_FEATURES["exam_difficulty"])
            if chart: st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Training data sample")


        with st.expander("Show sample of training data"):
            st.dataframe(data.head(100))
