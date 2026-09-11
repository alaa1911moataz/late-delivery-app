import streamlit as st
import pandas as pd
import joblib


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("late_delivery_model.pkl")


model = load_model()


# =========================================================
# Title
# =========================================================

st.title("🚚 Late Delivery Risk Prediction")

st.write(
    "Enter the order information below to predict whether "
    "the order is at risk of late delivery."
)


# =========================================================
# Input Form
# =========================================================

with st.form("prediction_form"):

    st.subheader("📦 Order Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        Type = st.selectbox(
            "Type",
            ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]
        )

        shipping_mode = st.selectbox(
            "Shipping Mode",
            [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day"
            ]
        )

        days_scheduled = st.number_input(
            "Days for shipment (scheduled)",
            min_value=0,
            max_value=10,
            value=3
        )

        order_quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=1
        )

        customer_segment = st.selectbox(
            "Customer Segment",
            [
                "Consumer",
                "Corporate",
                "Home Office"
            ]
        )

    with col2:

        benefit_per_order = st.number_input(
            "Benefit per order",
            value=0.0
        )

        sales_customer = st.number_input(
            "Sales per customer",
            min_value=0.0,
            value=100.0
        )

        order_item_discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            value=0.0
        )

        order_item_discount_rate = st.number_input(
            "Order Item Discount Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

        order_item_profit_ratio = st.number_input(
            "Order Item Profit Ratio",
            value=0.0
        )

    with col3:

        sales = st.number_input(
            "Sales",
            min_value=0.0,
            value=100.0
        )

        order_item_total = st.number_input(
            "Order Item Total",
            min_value=0.0,
            value=100.0
        )

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=100.0
        )

        category_id = st.number_input(
            "Category Id",
            min_value=0,
            value=1
        )

        department_id = st.number_input(
            "Department Id",
            min_value=0,
            value=1
        )

        product_card_id = st.number_input(
            "Product Card Id",
            min_value=0,
            value=1
        )


    # =====================================================
    # Customer Information
    # =====================================================

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        customer_city = st.text_input(
            "Customer City",
            value="Caguas"
        )

        customer_country = st.text_input(
            "Customer Country",
            value="Puerto Rico"
        )

        customer_state = st.text_input(
            "Customer State",
            value="PR"
        )

        customer_id = st.number_input(
            "Customer Id",
            min_value=0,
            value=1
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            value=18.0
        )

        longitude = st.number_input(
            "Longitude",
            value=-66.0
        )

        market = st.text_input(
            "Market",
            value="LATAM"
        )

    with col3:

        order_region = st.text_input(
            "Order Region",
            value="Caribbean"
        )

        order_state = st.text_input(
            "Order State",
            value="PR"
        )

        order_city = st.text_input(
            "Order City",
            value="Caguas"
        )

        order_country = st.text_input(
            "Order Country",
            value="Puerto Rico"
        )


    # =====================================================
    # Category / Product Information
    # =====================================================

    st.subheader("🛍️ Product Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        category_name = st.text_input(
            "Category Name",
            value="Sporting Goods"
        )

        department_name = st.text_input(
            "Department Name",
            value="Fitness"
        )

    with col2:

        product_name = st.text_input(
            "Product Name",
            value="Example Product"
        )

    with col3:

        order_id = st.number_input(
            "Order Id",
            min_value=0,
            value=1
        )

        order_item_id = st.number_input(
            "Order Item Id",
            min_value=0,
            value=1
        )


    # =====================================================
    # Date & Time
    # =====================================================

    st.subheader("📅 Order Date & Time")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        order_year = st.number_input(
            "Order Year",
            min_value=2000,
            max_value=2030,
            value=2017
        )

    with col2:

        order_month = st.number_input(
            "Order Month",
            min_value=1,
            max_value=12,
            value=6
        )

    with col3:

        order_day = st.number_input(
            "Order Day",
            min_value=1,
            max_value=31,
            value=15
        )

    with col4:

        order_dayofweek = st.number_input(
            "Order DayOfWeek",
            min_value=0,
            max_value=6,
            value=2
        )


    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=12
    )


    # =====================================================
    # Prediction Button
    # =====================================================

    predict_button = st.form_submit_button(
        "🔮 Predict Late Delivery Risk"
    )


# =========================================================
# Prediction
# =========================================================

if predict_button:

    # Create DataFrame with EXACT same column names
    input_data = pd.DataFrame({

        "Type": [Type],

        "Days for shipment (scheduled)": [
            days_scheduled
        ],

        "Benefit per order": [
            benefit_per_order
        ],

        "Sales per customer": [
            sales_customer
        ],

        "Category Id": [
            category_id
        ],

        "Category Name": [
            category_name
        ],

        "Customer City": [
            customer_city
        ],

        "Customer Country": [
            customer_country
        ],

        "Customer Id": [
            customer_id
        ],

        "Customer Segment": [
            customer_segment
        ],

        "Customer State": [
            customer_state
        ],

        "Department Id": [
            department_id
        ],

        "Department Name": [
            department_name
        ],

        "Latitude": [
            latitude
        ],

        "Longitude": [
            longitude
        ],

        "Market": [
            market
        ],

        "Order City": [
            order_city
        ],

        "Order Country": [
            order_country
        ],

        "Order Id": [
            order_id
        ],

        "Order Item Discount": [
            order_item_discount
        ],

        "Order Item Discount Rate": [
            order_item_discount_rate
        ],

        "Order Item Id": [
            order_item_id
        ],

        "Order Item Profit Ratio": [
            order_item_profit_ratio
        ],

        "Order Item Quantity": [
            order_quantity
        ],

        "Sales": [
            sales
        ],

        "Order Item Total": [
            order_item_total
        ],

        "Order Region": [
            order_region
        ],

        "Order State": [
            order_state
        ],

        "Product Card Id": [
            product_card_id
        ],

        "Product Name": [
            product_name
        ],

        "Product Price": [
            product_price
        ],

        "Shipping Mode": [
            shipping_mode
        ],

        "Order Year": [
            order_year
        ],

        "Order Month": [
            order_month
        ],

        "Order Day": [
            order_day
        ],

        "Order DayOfWeek": [
            order_dayofweek
        ],

        "Order Hour": [
            order_hour
        ]
    })


    # =====================================================
    # Prediction
    # =====================================================

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        # Probability of class 1
        class_1_index = list(model.classes_).index(1)

        late_probability = probability[class_1_index]


        # =================================================
        # Results
        # =================================================

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH RISK OF LATE DELIVERY"
                )

            else:

                st.success(
                    "✅ LOW RISK OF LATE DELIVERY"
                )


        with col2:

            st.metric(
                "Late Delivery Probability",
                f"{late_probability * 100:.2f}%"
            )


        # =================================================
        # Probability Bar
        # =================================================

        st.write("### Risk Level")

        st.progress(
            float(late_probability)
        )


        if late_probability >= 0.70:

            st.error(
                "🔴 High Risk"
            )

        elif late_probability >= 0.40:

            st.warning(
                "🟡 Medium Risk"
            )

        else:

            st.success(
                "🟢 Low Risk"
            )


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )