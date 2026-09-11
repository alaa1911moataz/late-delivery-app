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
# Load Model from Hugging Face
# =========================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="alaa1911/late-delivery-model",
        filename="late_delivery_model.pkl"
    )

    return joblib.load(model_path)


model = load_model()


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
    type=["xlsx", "xls", "csv"]
)


# =========================================================
# Process File
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # Read File
        # -------------------------------------------------

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)


        st.success(
            f"✅ File uploaded successfully: {uploaded_file.name}"
        )


        # -------------------------------------------------
        # Show Original Data
        # -------------------------------------------------

        st.subheader("📋 Uploaded Data")

        st.write(
            f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # -------------------------------------------------
        # Check Required Columns
        # -------------------------------------------------

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]


        if missing_columns:

            st.error(
                "❌ Some required columns are missing."
            )

            st.write("Missing columns:")

            for col in missing_columns:
                st.write(f"- `{col}`")

            st.stop()


        st.success(
            "✅ All required columns are available."
        )


        # =================================================
        # Predict Button
        # =================================================

        if st.button(
            "🔮 Predict Late Delivery Risk",
            type="primary"
        ):

            with st.spinner(
                "Making predictions..."
            ):

                # -----------------------------------------
                # Copy Data
                # -----------------------------------------

                prediction_data = df.copy()


                # -----------------------------------------
                # Convert Order Date
                # -----------------------------------------

                prediction_data["Order Date"] = pd.to_datetime(
                    prediction_data["Order Date"],
                    errors="coerce"
                )


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


                # -----------------------------------------
                # Convert Order Time
                # -----------------------------------------

                prediction_data["Order Time"] = pd.to_datetime(
                    prediction_data["Order Time"],
                    format="%H:%M:%S",
                    errors="coerce"
                )


                prediction_data["Order Hour"] = (
                    prediction_data["Order Time"].dt.hour
                )


                # -----------------------------------------
                # Remove Columns Not Used By Model
                # -----------------------------------------

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

                prediction_data = prediction_data.drop(
                    columns=[
                        col
                        for col in columns_to_remove
                        if col in prediction_data.columns
                    ],
                    errors="ignore"
                )


                # -----------------------------------------
                # Keep Exactly Model Features
                # -----------------------------------------

                model_features = (
                    model.named_steps["preprocessor"]
                    .feature_names_in_
                )

                prediction_data = prediction_data[
                    model_features
                ]


                # -----------------------------------------
                # Prediction
                # -----------------------------------------

                predictions = model.predict(
                    prediction_data
                )

                probabilities = model.predict_proba(
                    prediction_data
                )


                # -----------------------------------------
                # Find Class 1 Probability
                # -----------------------------------------

                class_1_index = list(
                    model.classes_
                ).index(1)

                late_probabilities = (
                    probabilities[:, class_1_index]
                )


                # -----------------------------------------
                # Add Results To Original Data
                # -----------------------------------------

                df["Late_Delivery_Prediction"] = [
                    "Late" if prediction == 1
                    else "On Time"
                    for prediction in predictions
                ]

                df["Late_Delivery_Probability"] = (
                    late_probabilities * 100
                ).round(2)


                # -----------------------------------------
                # Risk Level
                # -----------------------------------------

                df["Risk_Level"] = [
                    "High"
                    if probability >= 0.70
                    else "Medium"
                    if probability >= 0.40
                    else "Low"
                    for probability in late_probabilities
                ]


            # =================================================
            # Results
            # =================================================

            st.success(
                "✅ Prediction completed successfully!"
            )


            st.subheader("📊 Prediction Results")


            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            total_orders = len(df)

            late_orders = sum(
                predictions == 1
            )

            on_time_orders = total_orders - late_orders


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
            # Results Table
            # ---------------------------------------------

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


            output_file = "late_delivery_predictions.xlsx"


            df.to_excel(
                output_file,
                index=False
            )


            with open(
                output_file,
                "rb"
            ) as file:

                st.download_button(
                    label="📥 Download Excel File",
                    data=file,
                    file_name=output_file,
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    )
                )


    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )
```
