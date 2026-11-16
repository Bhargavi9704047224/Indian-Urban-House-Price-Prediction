
import streamlit as st
import requests

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- VERCEL FASTAPI BACKEND ----------------

API_URL = "https://indian-urban-house-price-prediction-d9jcuz2i3-bhargavi-aeea.vercel.app/predict"


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #172554, #2563eb);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 35px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
}

.section-heading {
    color: #1e3a8a;
    font-size: 23px;
    font-weight: 700;
    padding: 12px 0;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    border: none;
    width: 100%;
}

div.stButton > button:hover {
    background: #1e40af;
    color: white;
}

div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stFormSubmitButton > button:hover {
    background: #1e40af;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dbeafe;
}

div[data-testid="stExpander"] {
    background: white;
    border-radius: 12px;
    border: 1px solid #dbeafe;
}

.result-card {
    background: linear-gradient(120deg, #eff6ff, #dbeafe);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #bfdbfe;
}

.result-price {
    color: #1d4ed8;
    font-size: 32px;
    font-weight: bold;
}

.result-label {
    color: #475569;
    font-size: 15px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🏠 Property Predictor")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Project Overview")

st.sidebar.write("**Project:** Indian Urban House Price Prediction")
st.sidebar.write("**Model:** Decision Tree Regressor")
st.sidebar.write("**Dataset:** 100,000 Properties")
st.sidebar.write("**Backend:** FastAPI (Vercel)")
st.sidebar.write("**Frontend:** Streamlit (Render)")

st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 How to Use")

st.sidebar.write("1. Enter property details.")
st.sidebar.write("2. Select property facilities.")
st.sidebar.write("3. Click Predict House Price.")
st.sidebar.write("4. View the estimated price.")

st.sidebar.markdown("---")

st.sidebar.success("Prediction API: Vercel Backend")


# ---------------- PAGE HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🏡 Indian Urban House Price Prediction</h1>
    <p>Smart Machine Learning Based Real Estate Price Estimation</p>
    <p>Estimate property prices using location, size, amenities and other features.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- QUICK INFORMATION ----------------

st.markdown("### 📌 Property Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.info("🏙️ 10 Indian Cities")

with info2:
    st.info("🏠 Multiple Property Types")

with info3:
    st.info("🤖 ML-Based Prediction")


st.markdown("---")


# ---------------- INPUT FORM ----------------

st.markdown(
    '<p class="section-heading">📝 Enter Property Details</p>',
    unsafe_allow_html=True
)

with st.form("house_prediction_form"):

    # BASIC INFORMATION

    with st.expander("📍 Basic Property Information", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            Property_ID = st.text_input(
                "Property ID",
                "PROP-000001"
            )

            City = st.selectbox(
                "City",
                [
                    "Kolkata", "Ahmedabad", "Bangalore", "Delhi",
                    "Mumbai", "Pune", "Chennai", "Chandigarh",
                    "Hyderabad", "Jaipur"
                ]
            )

        with c2:
            Locality_Type = st.selectbox(
                "Locality Type",
                ["Mid-Range", "Affordable", "Premium"]
            )

            Property_Type = st.selectbox(
                "Property Type",
                ["Apartment", "Independent House", "Villa", "Builder Floor"]
            )

        with c3:
            BHK = st.number_input("BHK", 1, 6, 2)

            Bathrooms = st.number_input("Bathrooms", 1, 7, 2)

        c1, c2 = st.columns(2)

        with c1:
            Distance_to_Metro_km = st.number_input(
                "Distance to Metro (km)", 0.1, 25.0, 2.0
            )

        with c2:
            Distance_to_City_Center_km = st.number_input(
                "Distance to City Center (km)", 0.2, 40.0, 5.0
            )


    # PROPERTY SIZE

    with st.expander("📐 Property Size and Structure", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            Super_Area_SqFt = st.number_input(
                "Super Area (SqFt)", 250.0, 3941.9, 1000.0
            )

            Carpet_Area_SqFt = st.number_input(
                "Carpet Area (SqFt)", 166.4, 3259.2, 800.0
            )

        with c2:
            Floor_Number = st.number_input(
                "Floor Number", 0, 30, 2
            )

            Total_Floors = st.number_input(
                "Total Floors", 1, 35, 5
            )

        with c3:
            Age_of_Property = st.number_input(
                "Property Age", 0, 40, 5
            )

            Furnishing_Status = st.selectbox(
                "Furnishing Status",
                ["Semi-Furnished", "Unfurnished", "Furnished"]
            )


    # AMENITIES

    with st.expander("🛋️ Amenities and Facilities", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            Parking = st.selectbox(
                "Parking", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Lift_Available = st.selectbox(
                "Lift Available", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Gated_Community = st.selectbox(
                "Gated Community", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Balcony_Count = st.number_input(
                "Balcony Count", 0, 4, 2
            )

        with c2:
            Water_Supply = st.selectbox(
                "Water Supply",
                ["Corporation", "Both", "Borewell"]
            )

            Power_Backup = st.selectbox(
                "Power Backup",
                ["None", "Full", "Partial"]
            )

            Security = st.selectbox(
                "Security", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Gym_Available = st.selectbox(
                "Gym Available", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with c3:
            Swimming_Pool = st.selectbox(
                "Swimming Pool", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Clubhouse = st.selectbox(
                "Clubhouse", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Children_Play_Area = st.selectbox(
                "Children Play Area", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Park_Nearby = st.selectbox(
                "Park Nearby", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )


    # LOCATION

    with st.expander("📍 Location and Connectivity", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            Hospital_Distance_km = st.number_input(
                "Hospital Distance (km)", 0.0, 23.33, 2.0
            )

            School_Distance_km = st.number_input(
                "School Distance (km)", 0.0, 19.53, 2.0
            )

        with c2:
            Shopping_Mall_Distance_km = st.number_input(
                "Shopping Mall Distance (km)", 0.01, 29.65, 3.0
            )

            Road_Width_ft = st.number_input(
                "Road Width (ft)", 12, 80, 30
            )

        with c3:
            Nearby_Metro = st.selectbox(
                "Nearby Metro", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            Near_Main_Road = st.selectbox(
                "Near Main Road", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )


    # OTHER DETAILS

    with st.expander("🏢 Other Property Details", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            Facing = st.selectbox(
                "Facing",
                [
                    "North", "East", "North-East",
                    "West", "South", "South-East"
                ]
            )

            Ownership_Type = st.selectbox(
                "Ownership Type",
                ["Leasehold", "Co-operative", "Freehold"]
            )

            Construction_Status = st.selectbox(
                "Construction Status",
                ["Ready to Move", "Under Construction"]
            )

        with c2:
            Transaction_Type = st.selectbox(
                "Transaction Type", ["New", "Resale"]
            )

            Air_Quality_Index = st.number_input(
                "Air Quality Index", 25, 300, 100
            )

            Noise_Level_dB = st.number_input(
                "Noise Level (dB)", 25.0, 95.0, 50.0
            )

        with c3:
            Green_Cover_Percent = st.number_input(
                "Green Cover (%)", 2.0, 65.0, 30.0
            )

            Property_Tax_Annual_INR = st.number_input(
                "Annual Property Tax (INR)", 3002, 79999, 10000
            )

            Maintenance_Monthly_INR = st.number_input(
                "Monthly Maintenance (INR)", 500, 17999, 2000
            )

        Price_Per_SqFt_INR = st.number_input(
            "Price Per SqFt (INR)", 524.97, 38708.0, 5000.0
        )


    st.markdown("---")

    st.markdown("### 🚀 Ready to Estimate Your Property Price?")

    submitted = st.form_submit_button(
        "🔮 Predict House Price"
    )


# ---------------- PREDICTION ----------------

if submitted:

    data = {
        "Property_ID": Property_ID,
        "City": City,
        "Locality_Type": Locality_Type,
        "Distance_to_Metro_km": Distance_to_Metro_km,
        "Distance_to_City_Center_km": Distance_to_City_Center_km,
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Bathrooms": Bathrooms,
        "Super_Area_SqFt": Super_Area_SqFt,
        "Carpet_Area_SqFt": Carpet_Area_SqFt,
        "Floor_Number": Floor_Number,
        "Total_Floors": Total_Floors,
        "Age_of_Property": Age_of_Property,
        "Furnishing_Status": Furnishing_Status,
        "Parking": Parking,
        "Lift_Available": Lift_Available,
        "Gated_Community": Gated_Community,
        "Balcony_Count": Balcony_Count,
        "Water_Supply": Water_Supply,
        "Power_Backup": Power_Backup,
        "Security": Security,
        "Gym_Available": Gym_Available,
        "Swimming_Pool": Swimming_Pool,
        "Clubhouse": Clubhouse,
        "Children_Play_Area": Children_Play_Area,
        "Park_Nearby": Park_Nearby,
        "Hospital_Distance_km": Hospital_Distance_km,
        "School_Distance_km": School_Distance_km,
        "Shopping_Mall_Distance_km": Shopping_Mall_Distance_km,
        "Road_Width_ft": Road_Width_ft,
        "Facing": Facing,
        "Ownership_Type": Ownership_Type,
        "Construction_Status": Construction_Status,
        "Transaction_Type": Transaction_Type,
        "Nearby_Metro": Nearby_Metro,
        "Near_Main_Road": Near_Main_Road,
        "Air_Quality_Index": Air_Quality_Index,
        "Noise_Level_dB": Noise_Level_dB,
        "Green_Cover_Percent": Green_Cover_Percent,
        "Property_Tax_Annual_INR": Property_Tax_Annual_INR,
        "Maintenance_Monthly_INR": Maintenance_Monthly_INR,
        "Price_Per_SqFt_INR": Price_Per_SqFt_INR
    }

    try:

        with st.spinner("🏠 Analyzing property details..."):

            response = requests.post(
                API_URL,
                json=data,
                timeout=60
            )

        response.raise_for_status()

        result = response.json()

        st.markdown("---")

        st.markdown(
            '<p class="section-heading">🎯 Your Prediction Results</p>',
            unsafe_allow_html=True
        )

        st.success("House price prediction completed successfully!")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    ESTIMATED PROPERTY PRICE
                </div>
                <div class="result-price">
                    ₹ {result['predicted_price_in_lakhs']:,.2f} Lakhs
                </div>
                <div class="result-label">
                    Property ID: {result['Property_ID']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Price in Lakhs",
                f"₹ {result['predicted_price_in_lakhs']:,.2f}"
            )

        with col2:
            st.metric(
                "Estimated Price in INR",
                f"₹ {result['predicted_price_in_INR']:,.2f}"
            )

        st.markdown("---")

        st.markdown("### 🏡 Property Summary")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("City", City)

        with s2:
            st.metric("Property Type", Property_Type)

        with s3:
            st.metric("BHK", BHK)

        with s4:
            st.metric("Area", f"{Super_Area_SqFt:,.0f} SqFt")

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to Vercel FastAPI backend. "
            "Please check your backend deployment."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The prediction request timed out. Please try again."
        )

    except requests.exceptions.HTTPError as e:
        st.error(f"Backend HTTP error: {e}")
        st.code(response.text)

    except (KeyError, ValueError) as e:
        st.error(f"Invalid prediction response from backend: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("""
<div class="footer">
    🏠 Indian Urban House Price Prediction |
    Machine Learning + FastAPI + Streamlit
    <br>
    Built as a Machine Learning Deployment Project
</div>
""", unsafe_allow_html=True)
