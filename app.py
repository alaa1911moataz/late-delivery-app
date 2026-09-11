```python
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)


# =========================================================
# Dark Mode
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# =========================================================
# Custom Theme
# =========================================================

if st.session_state.dark_mode:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        .stApp p,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #FAFAFA !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #262730 !important;
            color: #FFFFFF !important;
        }

        .stSelectbox div[data-baseweb="select"] span {
            color: #FFFFFF !important;
        }

        section[data-testid="stFileUploader"] {
            background-color: #262730;
            border-radius: 10px;
            padding: 10px;
        }

        .stButton button {
            background-color: #262730;
            color: #FFFFFF;
            border: 1px solid #555555;
            border-radius: 8px;
        }

        .stButton button:hover {
            border-color: #FFFFFF;
        }

        [data-testid="stDataFrame"] {
            background-color: #262730;
        }

        [data-testid="stMetric"] {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Dark Mode Button
# =========================================================

col1, col2 = st.columns([9, 1])

with col2:

    if st.button(
        "☀️" if st.session_state.dark_mode else "🌙",
        help="Toggle Dark Mode"
    ):

        st.session_state.dark_mode = (
            not st.session_state.dark_mode
        )

        st.rerun()


# =========================================================
# Load Model From Hugging Face
# =========================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="alaa1911/late-delivery-model",
        filename="late_delivery_model.pkl"
    )

    return joblib.load(model_path)


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Could not load the prediction model."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# Title
# =========================================================

st.title("🚚 Late Delivery Risk Prediction")

st.write(
    "Upload an Excel or CSV file containing order information. "
    "The model will predict late delivery risk for every order."
)


# =========================================================
# Required Columns
# =========================================================

required_columns = [

    "Type",

    "Days for shipment (scheduled)",

    "Benefit per order",

    "Sales per customer",

    "Category Id",

    "Category Name",

    "Customer City",

    "Customer Country",

    "Customer Segment",

    "Customer State",

    "Department Id",

    "Department Name",

    "Latitude",

    "Longitude",

    "Market",

    "Order City",

    "Order Country",

    "Order Item Discount",

    "Order Item Discount Rate",

    "Order Item Profit Ratio",

    "Order Item Quantity",

    "Sales",

    "Order Item Total",

    "Order Region",

    "Order State",

    "Product Card Id",

    "Product Name",

    "Product Price",

    "Shipping Mode",

    "Order Date",

    "Order Time"

]


# =========================================================
# File Upload
# =========================================================

st.subheader("📂 Upload Your File")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=[
        "xlsx",
        "xls",
        "csv"
    ]
)


# =========================================================
# Process Uploaded File
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # Read File
        # -------------------------------------------------

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )


        # -------------------------------------------------
        # Success Message
        # -------------------------------------------------

        st.success(
            f"✅ File uploaded successfully: "
            f"{uploaded_file.name}"
        )


        # -------------------------------------------------
        # Basic Information
        # -------------------------------------------------

        st.write(
            f"**Rows:** {df.shape[0]}  |  "
            f"**Columns:** {df.shape[1]}"
        )


        # -------------------------------------------------
        # Preview
        # -------------------------------------------------

        st.subheader(
            "📋 Uploaded Data Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # =================================================
        # Check Required Columns
        # =================================================

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]


        if missing_columns:

            st.error(
                "❌ Some required columns are missing."
            )

            st.write(
                "Please make sure your file contains:"
            )

            for column in missing_columns:

                st.write(
                    f"- `{column}`"
                )

            st.stop()


        st.success(
            "✅ All required columns are available."
        )


        # =================================================
        # Prediction Button
        # =================================================

        if st.button(
            "🔮 Predict Late Delivery Risk",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Making predictions..."
                ):

                    # -------------------------------------
                    # Copy Original Data
                    # -------------------------------------

                    prediction_data = df.copy()


                    # -------------------------------------
                    # Convert Order Date
                    # -------------------------------------

                    prediction_data["Order Date"] = (
                        pd.to_datetime(
                            prediction_data["Order Date"],
                            errors="coerce"
                        )
                    )


                    # -------------------------------------
                    # Create Date Features
                    # -------------------------------------

                    prediction_data["Order Year"] = (
                        prediction_data["Order Date"].dt.year
                    )

                    prediction_data["Order Month"] = (
                        prediction_data["Order Date"].dt.month
                    )

                    prediction_data["Order Day"] = (
                        prediction_data["Order Date"].dt.day
                    )

                    prediction_data["Order DayOfWeek"] = (
                        prediction_data["Order Date"].dt.dayofweek
                    )


                    # -------------------------------------
                    # Convert Order Time
                    # -------------------------------------

                    prediction_data["Order Time"] = (
                        pd.to_datetime(
                            prediction_data["Order Time"],
                            format="%H:%M:%S",
                            errors="coerce"
                        )
                    )


                    # -------------------------------------
                    # Create Hour Feature
                    # -------------------------------------

                    prediction_data["Order Hour"] = (
                        prediction_data["Order Time"].dt.hour
                    )


                    # -------------------------------------
                    # Check Date / Time
                    # -------------------------------------

                    invalid_dates = (
                        prediction_data["Order Year"].isna().sum()
                    )

                    invalid_times = (
                        prediction_data["Order Hour"].isna().sum()
                    )


                    if invalid_dates > 0:

                        st.warning(
                            f"⚠️ {invalid_dates} rows have "
                            "invalid Order Date values."
                        )


                    if invalid_times > 0:

                        st.warning(
                            f"⚠️ {invalid_times} rows have "
                            "invalid Order Time values."
                        )


                    # -------------------------------------
                    # Remove Columns Not Used By Model
                    # -------------------------------------

                    columns_to_remove = [

                        "Order Date",

                        "Order Time",

                        "Order Item Id",

                        "Order Id",

                        "Customer Id",

                        "Days for shipping (real)",

                        "Delivery Status",

                        "Order Status",

                        "Shipping Date",

                        "Shipping Time",

                        "Late_delivery_risk"

                    ]


                    prediction_data = (
                        prediction_data.drop(
                            columns=[
                                column

                                for column in columns_to_remove

                                if column in prediction_data.columns
                            ],
                            errors="ignore"
                        )
                    )


                    # -------------------------------------
                    # Get Exact Model Features
                    # -------------------------------------

                    model_features = (
                        model
                        .named_steps[
                            "preprocessor"
                        ]
                        .feature_names_in_
                    )


                    # -------------------------------------
                    # Keep Same Features As Training
                    # -------------------------------------

                    prediction_data = (
                        prediction_data[
                            model_features
                        ]
                    )


                    # -------------------------------------
                    # Make Predictions
                    # -------------------------------------

                    predictions = (
                        model.predict(
                            prediction_data
                        )
                    )


                    probabilities = (
                        model.predict_proba(
                            prediction_data
                        )
                    )


                    # -------------------------------------
                    # Find Class 1
                    # -------------------------------------

                    class_1_index = list(
                        model.classes_
                    ).index(1)


                    late_probabilities = (
                        probabilities[
                            :,
                            class_1_index
                        ]
                    )


                    # =====================================
                    # Add Predictions To Original Data
                    # =====================================

                    df[
                        "Late_Delivery_Prediction"
                    ] = [

                        "Late"
                        if prediction == 1
                        else "On Time"

                        for prediction in predictions

                    ]


                    df[
                        "Late_Delivery_Probability"
                    ] = (

                        late_probabilities * 100

                    ).round(2)


                    # -------------------------------------
                    # Risk Level
                    # -------------------------------------

                    df[
                        "Risk_Level"
                    ] = [

                        "High"

                        if probability >= 0.70

                        else "Medium"

                        if probability >= 0.40

                        else "Low"

                        for probability
                        in late_probabilities

                    ]


                # =================================================
                # Prediction Completed
                # =================================================

                st.success(
                    "✅ Prediction completed successfully!"
                )


                # =================================================
                # Statistics
                # =================================================

                st.subheader(
                    "📊 Prediction Summary"
                )


                total_orders = len(
                    df
                )


                late_orders = int(
                    sum(
                        predictions == 1
                    )
                )


                on_time_orders = (
                    total_orders
                    - late_orders
                )


                high_risk = int(
                    sum(
                        late_probabilities >= 0.70
                    )
                )


                medium_risk = int(
                    sum(
                        (
                            late_probabilities >= 0.40
                        )
                        &
                        (
                            late_probabilities < 0.70
                        )
                    )
                )


                low_risk = int(
                    sum(
                        late_probabilities < 0.40
                    )
                )


                # ---------------------------------------------
                # Metrics
                # ---------------------------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Total Orders",
                        total_orders
                    )


                with col2:

                    st.metric(
                        "Predicted Late",
                        late_orders
                    )


                with col3:

                    st.metric(
                        "Predicted On Time",
                        on_time_orders
                    )


                # ---------------------------------------------
                # Risk Metrics
                # ---------------------------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.error(
                        f"🔴 High Risk: {high_risk}"
                    )


                with col2:

                    st.warning(
                        f"🟡 Medium Risk: {medium_risk}"
                    )


                with col3:

                    st.success(
                        f"🟢 Low Risk: {low_risk}"
                    )


                # =================================================
                # Results
                # =================================================

                st.subheader(
                    "📋 Prediction Results"
                )


                st.dataframe(
                    df,
                    use_container_width=True
                )


                # =================================================
                # Download Excel
                # =================================================

                st.subheader(
                    "📥 Download Results"
                )


                output_file = (
                    "late_delivery_predictions.xlsx"
                )


                # ---------------------------------------------
                # Create Excel In Memory
                # ---------------------------------------------

                from io import BytesIO


                output = BytesIO()


                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:

                    df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Predictions"
                    )


                output.seek(0)


                # ---------------------------------------------
                # Download Button
                # ---------------------------------------------

                st.download_button(
                    label="📥 Download Excel File",
                    data=output,
                    file_name=output_file,
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )


            except Exception as e:

                st.error(
                    "❌ Prediction Error"
                )

                st.code(
                    str(e)
                )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "🚚 Late Delivery Risk Prediction | "
    "Machine Learning Application"
)
```
