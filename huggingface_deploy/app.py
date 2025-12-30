"""
Travel ML Dashboard - Hugging Face Spaces Version
Interactive visualization and model inference
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Page config
st.set_page_config(
    page_title="Travel ML Dashboard",
    page_icon="plane",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - adjusted for HF Spaces flat structure
DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.dirname(__file__)


# ==================== DATA LOADING ====================

@st.cache_data
def load_data():
    """Load all datasets"""
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    flights = pd.read_csv(os.path.join(DATA_DIR, 'flights.csv'))
    hotels = pd.read_csv(os.path.join(DATA_DIR, 'hotels.csv'))

    flights['date'] = pd.to_datetime(flights['date'])
    hotels['date'] = pd.to_datetime(hotels['date'])

    return users, flights, hotels


@st.cache_resource
def load_models():
    """Load ML models"""
    models = {}

    try:
        models['flight'] = joblib.load(os.path.join(MODELS_DIR, 'flight_price_model.pkl'))
    except:
        models['flight'] = None

    try:
        models['gender'] = joblib.load(os.path.join(MODELS_DIR, 'gender_classifier.pkl'))
    except:
        models['gender'] = None

    try:
        models['hotel'] = joblib.load(os.path.join(MODELS_DIR, 'hotel_recommender.pkl'))
    except:
        models['hotel'] = None

    return models


# ==================== SIDEBAR ====================

st.sidebar.title("Travel ML Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select Page",
    ["Data Overview", "Flight Price Prediction", "Gender Classification",
     "Hotel Recommendations", "Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard provides insights from travel data and "
    "allows interaction with ML models for predictions and recommendations."
)


# ==================== PAGES ====================

def data_overview_page():
    """Data overview and statistics page"""
    st.title("Data Overview")

    users, flights, hotels = load_data()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(users):,}")
    col2.metric("Total Flights", f"{len(flights):,}")
    col3.metric("Total Hotel Bookings", f"{len(hotels):,}")
    col4.metric("Avg Flight Price", f"${flights['price'].mean():.2f}")

    st.markdown("---")

    # Tabs for different datasets
    tab1, tab2, tab3 = st.tabs(["Users", "Flights", "Hotels"])

    with tab1:
        st.subheader("Users Dataset")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(users, names='gender', title='Gender Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(users, x='age', nbins=20, title='Age Distribution',
                              color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)

        company_counts = users['company'].value_counts().head(10)
        fig = px.bar(x=company_counts.index, y=company_counts.values,
                    title='Top 10 Companies', labels={'x': 'Company', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(users.head(20), use_container_width=True)

    with tab2:
        st.subheader("Flights Dataset")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(flights, names='flightType', title='Flight Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(flights, x='flightType', y='price', title='Price by Flight Type')
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(flights.sample(min(500, len(flights))),
                        x='distance', y='price', color='flightType',
                        title='Price vs Distance', opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

        flights['month'] = flights['date'].dt.month
        monthly_avg = flights.groupby('month')['price'].mean().reset_index()
        fig = px.line(monthly_avg, x='month', y='price',
                     title='Average Flight Price by Month', markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(flights.head(20), use_container_width=True)

    with tab3:
        st.subheader("Hotels Dataset")

        col1, col2 = st.columns(2)

        with col1:
            top_hotels = hotels['name'].value_counts().head(10)
            fig = px.bar(x=top_hotels.values, y=top_hotels.index, orientation='h',
                        title='Top 10 Hotels by Bookings')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(hotels, x='price', nbins=30,
                              title='Hotel Price Per Day Distribution')
            st.plotly_chart(fig, use_container_width=True)

        top_places = hotels['place'].value_counts().head(10)
        fig = px.bar(x=top_places.index, y=top_places.values,
                    title='Top 10 Destinations', labels={'x': 'Place', 'y': 'Bookings'})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(hotels.head(20), use_container_width=True)


def flight_prediction_page():
    """Flight price prediction page"""
    st.title("Flight Price Prediction")

    users, flights, hotels = load_data()
    models = load_models()

    st.markdown("Enter flight details to predict the price:")

    col1, col2 = st.columns(2)

    with col1:
        # Get list of existing cities
        origin_cities = sorted(flights['from'].unique().tolist())
        dest_cities = sorted(flights['to'].unique().tolist())
        agencies = sorted(flights['agency'].unique().tolist())

        # Origin selection
        st.markdown("**From (Origin)**")
        origin_method = st.radio("Choose origin:", ["Select from list", "Type custom city"],
                                 key="origin_method", horizontal=True)
        if origin_method == "Select from list":
            origin = st.selectbox("Select origin city:", origin_cities, key="origin_dropdown")
        else:
            origin = st.text_input("Enter city name:", placeholder="e.g., Delhi, Mumbai, London",
                                  key="origin_city")

        # Destination selection
        st.markdown("**To (Destination)**")
        dest_method = st.radio("Choose destination:", ["Select from list", "Type custom city"],
                               key="dest_method", horizontal=True)
        if dest_method == "Select from list":
            destination = st.selectbox("Select destination city:", dest_cities, key="dest_dropdown")
        else:
            destination = st.text_input("Enter city name:", placeholder="e.g., London, Singapore, Paris",
                                       key="dest_city")

        flight_type = st.selectbox("Flight Type", flights['flightType'].unique())

        # Agency selection
        st.markdown("**Agency**")
        agency_method = st.radio("Choose agency:", ["Select from list", "Type custom"],
                                 key="agency_method", horizontal=True)
        if agency_method == "Select from list":
            agency = st.selectbox("Select agency:", agencies, key="agency_dropdown")
        else:
            agency = st.text_input("Enter agency name:", placeholder="e.g., Air India, Delta",
                                  key="agency_input")
            if not agency:
                agency = "Unknown"

    with col2:
        distance = st.slider("Distance (miles)", 100, 10000, 2000)
        time_hours = st.slider("Flight Duration (hours)", 0.5, 20.0, 4.0, 0.5)
        month = st.selectbox("Month", list(range(1, 13)))
        day_of_week = st.selectbox("Day of Week",
                                   ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"])

    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    is_weekend = 1 if day_map[day_of_week] >= 5 else 0

    # Show info for custom entries
    known_cities = set(flights['from'].unique().tolist() + flights['to'].unique().tolist())
    if origin and origin not in known_cities:
        st.info(f"Note: '{origin}' is a custom city. Prediction will use distance and duration as primary factors.")
    if destination and destination not in known_cities:
        st.info(f"Note: '{destination}' is a custom city. Prediction will use distance and duration as primary factors.")

    if st.button("Predict Price", type="primary"):
        if models['flight'] is not None:
            try:
                model = models['flight']['model']
                scaler = models['flight']['scaler']
                label_encoders = models['flight']['label_encoders']
                feature_cols = models['flight']['feature_columns']

                features = []
                for col in feature_cols:
                    if col == 'time':
                        features.append(time_hours)
                    elif col == 'distance':
                        features.append(distance)
                    elif col == 'from_encoded':
                        try:
                            features.append(label_encoders['from'].transform([origin])[0])
                        except:
                            features.append(0)
                    elif col == 'to_encoded':
                        try:
                            features.append(label_encoders['to'].transform([destination])[0])
                        except:
                            features.append(0)
                    elif col == 'flightType_encoded':
                        try:
                            features.append(label_encoders['flightType'].transform([flight_type])[0])
                        except:
                            features.append(0)
                    elif col == 'agency_encoded':
                        try:
                            features.append(label_encoders['agency'].transform([agency])[0])
                        except:
                            features.append(0)
                    elif col == 'month':
                        features.append(month)
                    elif col == 'day_of_week':
                        features.append(day_map[day_of_week])
                    elif col == 'is_weekend':
                        features.append(is_weekend)

                X = np.array(features).reshape(1, -1)
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]

                st.success(f"### Predicted Price: ${prediction:.2f}")

                # Show prediction details
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Price", f"${prediction:.2f}")
                col2.metric("Distance", f"{distance} miles")
                col3.metric("Duration", f"{time_hours} hours")

                # Try to find similar flights for comparison (only if origin exists in data)
                similar = flights[
                    (flights['from'] == origin) &
                    (flights['flightType'] == flight_type)
                ]['price']

                if len(similar) > 0:
                    st.markdown("#### Comparison with Similar Flights:")
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    comp_col1.metric("Your Prediction", f"${prediction:.2f}")
                    comp_col2.metric("Avg Similar Flights", f"${similar.mean():.2f}")
                    comp_col3.metric("Difference", f"${prediction - similar.mean():.2f}")
                else:
                    # For custom cities, show comparison with same flight type overall
                    same_type = flights[flights['flightType'] == flight_type]['price']
                    if len(same_type) > 0:
                        st.markdown(f"#### Reference: Average {flight_type} class price is ${same_type.mean():.2f}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Flight price model not loaded. Please ensure model files are uploaded.")


def gender_classification_page():
    """Gender classification page"""
    st.title("Gender Classification")

    users, flights, hotels = load_data()
    models = load_models()

    st.markdown("Enter user travel patterns to predict gender:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        company = st.selectbox("Company", sorted(users['company'].unique()))
        avg_flight_price = st.number_input("Avg Flight Price ($)", 100, 2000, 500)
        total_flight_spent = st.number_input("Total Flight Spending ($)", 500, 50000, 5000)
        flight_count = st.number_input("Number of Flights", 1, 100, 10)

    with col2:
        avg_distance = st.number_input("Avg Flight Distance", 500, 5000, 2000)
        avg_hotel_total = st.number_input("Avg Hotel Spending ($)", 100, 2000, 500)
        hotel_count = st.number_input("Number of Hotel Bookings", 1, 100, 10)
        avg_stay_days = st.slider("Avg Stay Duration (days)", 1, 14, 3)
        first_class_count = st.number_input("First Class Flights", 0, 50, 2)

    if st.button("Predict Gender", type="primary"):
        if models['gender'] is not None:
            try:
                model = models['gender']['model']
                scaler = models['gender']['scaler']
                label_encoders = models['gender']['label_encoders']
                target_encoder = models['gender']['target_encoder']
                feature_cols = models['gender']['feature_columns']

                features = []
                for col in feature_cols:
                    if col == 'age':
                        features.append(age)
                    elif col == 'company_encoded':
                        try:
                            features.append(label_encoders['company'].transform([company])[0])
                        except:
                            features.append(0)
                    elif col == 'avg_flight_price' or col == 'avg_price':
                        features.append(avg_flight_price)
                    elif col == 'total_flight_spent' or col == 'total_spent':
                        features.append(total_flight_spent)
                    elif col == 'flight_count':
                        features.append(flight_count)
                    elif col == 'avg_distance':
                        features.append(avg_distance)
                    elif col == 'avg_hotel_total' or col == 'avg_hotel':
                        features.append(avg_hotel_total)
                    elif col == 'hotel_booking_count' or col == 'hotel_count':
                        features.append(hotel_count)
                    elif col == 'avg_stay_days' or col == 'avg_days':
                        features.append(avg_stay_days)
                    elif col == 'first_class_count':
                        features.append(first_class_count)
                    else:
                        features.append(0)

                X = np.array(features).reshape(1, -1)
                X_scaled = scaler.transform(X)

                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0]

                gender = target_encoder.inverse_transform([prediction])[0]
                confidence = max(probability) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"### Predicted Gender: {gender.capitalize()}")
                    st.metric("Confidence", f"{confidence:.1f}%")

                with col2:
                    probs_df = pd.DataFrame({
                        'Gender': target_encoder.classes_,
                        'Probability': probability
                    })
                    fig = px.bar(probs_df, x='Gender', y='Probability',
                                title='Prediction Probabilities')
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Gender classifier not loaded. Please ensure model files are uploaded.")


def hotel_recommendations_page():
    """Hotel recommendations page"""
    st.title("Hotel Recommendations")

    users, flights, hotels = load_data()
    models = load_models()

    tab1, tab2 = st.tabs(["By User", "By Preferences"])

    with tab1:
        st.subheader("Get Recommendations for a User")
        user_code = st.number_input("User Code", 1, users['code'].max(), 100)
        n_recs = st.slider("Number of Recommendations", 3, 10, 5)

        if st.button("Get Recommendations", type="primary", key="user_recs"):
            if models['hotel'] is not None:
                try:
                    user_history = hotels[hotels['userCode'] == user_code]

                    if len(user_history) > 0:
                        st.markdown("#### User's Booking History:")
                        st.dataframe(user_history[['name', 'place', 'days', 'total', 'date']].head(5))

                    hotel_info = models['hotel']['hotel_info']
                    nn_model = models['hotel']['nn_model']
                    scaler = models['hotel']['scaler']

                    if len(user_history) > 0:
                        pref_vector = np.array([
                            user_history['price'].mean(),
                            user_history['days'].mean(),
                            user_history['total'].mean(),
                            0
                        ]).reshape(1, -1)
                        pref_scaled = scaler.transform(pref_vector)

                        distances, indices = nn_model.kneighbors(pref_scaled)

                        st.markdown("#### Recommended Hotels:")
                        recs = []
                        for idx, dist in zip(indices[0][:n_recs], distances[0][:n_recs]):
                            info = hotel_info.iloc[idx]
                            recs.append({
                                'Hotel': info['hotel_id'].split('_')[0],
                                'Location': info['place'],
                                'Avg Price/Day': f"${info['price']:.2f}",
                                'Match Score': f"{(1-dist)*100:.1f}%"
                            })
                        st.dataframe(pd.DataFrame(recs), use_container_width=True)
                    else:
                        st.info("No booking history found. Try the preferences tab.")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Hotel recommender not loaded.")

    with tab2:
        st.subheader("Get Recommendations by Preferences")

        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("Budget per Night ($)", 50, 500, 150)
            stay_days = st.slider("Planned Stay (days)", 1, 14, 3)
        with col2:
            destination = st.selectbox("Preferred Destination",
                                       ["Any"] + sorted(hotels['place'].unique().tolist()))

        if st.button("Get Recommendations", type="primary", key="pref_recs"):
            if models['hotel'] is not None:
                try:
                    hotel_info = models['hotel']['hotel_info']
                    nn_model = models['hotel']['nn_model']
                    scaler = models['hotel']['scaler']

                    pref_vector = np.array([
                        budget,
                        stay_days,
                        budget * stay_days,
                        0
                    ]).reshape(1, -1)
                    pref_scaled = scaler.transform(pref_vector)

                    distances, indices = nn_model.kneighbors(pref_scaled, n_neighbors=10)

                    st.markdown("#### Recommended Hotels:")
                    recs = []
                    for idx, dist in zip(indices[0], distances[0]):
                        info = hotel_info.iloc[idx]
                        if destination == "Any" or info['place'] == destination:
                            recs.append({
                                'Hotel': info['hotel_id'].split('_')[0],
                                'Location': info['place'],
                                'Avg Price/Day': f"${info['price']:.2f}",
                                'Match Score': f"{(1-dist)*100:.1f}%"
                            })

                    if recs:
                        st.dataframe(pd.DataFrame(recs[:5]), use_container_width=True)
                    else:
                        st.info("No matching hotels found. Try adjusting your preferences.")

                except Exception as e:
                    st.error(f"Error: {e}")


def analytics_page():
    """Analytics and insights page"""
    st.title("Travel Analytics")

    users, flights, hotels = load_data()

    col1, col2, col3, col4 = st.columns(4)

    total_revenue = flights['price'].sum() + hotels['total'].sum()
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Avg Ticket Price", f"${flights['price'].mean():.2f}")
    col3.metric("Avg Hotel Spend", f"${hotels['total'].mean():.2f}")
    col4.metric("Active Users", f"{users['code'].nunique():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        revenue_by_type = flights.groupby('flightType')['price'].sum().reset_index()
        fig = px.pie(revenue_by_type, values='price', names='flightType',
                    title='Revenue by Flight Type')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        flights['route'] = flights['from'] + ' -> ' + flights['to']
        top_routes = flights.groupby('route')['price'].agg(['count', 'sum']).reset_index()
        top_routes.columns = ['Route', 'Bookings', 'Revenue']
        top_routes = top_routes.nlargest(10, 'Bookings')

        fig = px.bar(top_routes, x='Route', y='Bookings', title='Top 10 Routes')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Booking Trends Over Time")

    flights['month_year'] = flights['date'].dt.to_period('M').astype(str)
    monthly_bookings = flights.groupby('month_year').agg({
        'travelCode': 'count',
        'price': 'sum'
    }).reset_index()
    monthly_bookings.columns = ['Month', 'Bookings', 'Revenue']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly_bookings['Month'], y=monthly_bookings['Bookings'],
               name="Bookings", marker_color='#636EFA'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=monthly_bookings['Month'], y=monthly_bookings['Revenue'],
                   name="Revenue", marker_color='#EF553B', mode='lines+markers'),
        secondary_y=True
    )
    fig.update_layout(title="Monthly Bookings and Revenue")
    fig.update_yaxes(title_text="Bookings", secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("User Segmentation")

    user_spending = flights.groupby('userCode')['price'].sum().reset_index()
    user_spending.columns = ['userCode', 'total_spent']

    user_analysis = users.merge(user_spending, left_on='code', right_on='userCode', how='left')
    user_analysis['total_spent'] = user_analysis['total_spent'].fillna(0)

    user_analysis['age_group'] = pd.cut(user_analysis['age'],
                                        bins=[0, 25, 35, 45, 55, 100],
                                        labels=['18-25', '26-35', '36-45', '46-55', '55+'])

    col1, col2 = st.columns(2)

    with col1:
        age_spending = user_analysis.groupby('age_group')['total_spent'].mean().reset_index()
        fig = px.bar(age_spending, x='age_group', y='total_spent',
                    title='Average Spending by Age Group',
                    labels={'age_group': 'Age Group', 'total_spent': 'Avg Spending ($)'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_spending = user_analysis.groupby('gender')['total_spent'].mean().reset_index()
        fig = px.bar(gender_spending, x='gender', y='total_spent',
                    title='Average Spending by Gender',
                    labels={'gender': 'Gender', 'total_spent': 'Avg Spending ($)'})
        st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN ====================

if page == "Data Overview":
    data_overview_page()
elif page == "Flight Price Prediction":
    flight_prediction_page()
elif page == "Gender Classification":
    gender_classification_page()
elif page == "Hotel Recommendations":
    hotel_recommendations_page()
elif page == "Analytics":
    analytics_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for MLOps Capstone Project")
