"""
Airflow DAG for Travel Data Pipeline
Automates data processing, model training, and deployment
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import joblib
import os

# Default arguments
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['mlops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Paths
DATA_DIR = '/opt/airflow/data'
MODELS_DIR = '/opt/airflow/models'


def validate_data(**context):
    """Validate incoming data quality"""
    print("🔍 Validating data quality...")
    
    users = pd.read_csv(f'{DATA_DIR}/users.csv')
    flights = pd.read_csv(f'{DATA_DIR}/flights.csv')
    hotels = pd.read_csv(f'{DATA_DIR}/hotels.csv')
    
    validations = {
        'users_not_empty': len(users) > 0,
        'flights_not_empty': len(flights) > 0,
        'hotels_not_empty': len(hotels) > 0,
        'users_no_nulls': users[['code', 'name', 'gender']].isnull().sum().sum() == 0,
        'flights_no_nulls': flights[['travelCode', 'userCode', 'price']].isnull().sum().sum() == 0,
        'prices_positive': (flights['price'] > 0).all(),
    }
    
    all_passed = all(validations.values())
    for check, passed in validations.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    
    if not all_passed:
        raise ValueError("Data validation failed!")
    
    context['ti'].xcom_push(key='data_stats', value={
        'users': len(users), 'flights': len(flights), 'hotels': len(hotels)
    })
    return True


def preprocess_data(**context):
    """Preprocess and clean data"""
    print("🔄 Preprocessing data...")
    
    os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
    
    users = pd.read_csv(f'{DATA_DIR}/users.csv')
    flights = pd.read_csv(f'{DATA_DIR}/flights.csv')
    hotels = pd.read_csv(f'{DATA_DIR}/hotels.csv')
    
    flights['date'] = pd.to_datetime(flights['date'])
    flights = flights.dropna(subset=['price']).query('price > 0')
    
    hotels['date'] = pd.to_datetime(hotels['date'])
    hotels = hotels.dropna(subset=['total']).query('total > 0')
    
    users.to_csv(f'{DATA_DIR}/processed/users_processed.csv', index=False)
    flights.to_csv(f'{DATA_DIR}/processed/flights_processed.csv', index=False)
    hotels.to_csv(f'{DATA_DIR}/processed/hotels_processed.csv', index=False)
    
    print(f"✅ Processed: {len(users)} users, {len(flights)} flights, {len(hotels)} hotels")
    return True


def train_flight_model(**context):
    """Train flight price prediction model"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("🎯 Training flight price model...")
    
    flights = pd.read_csv(f'{DATA_DIR}/processed/flights_processed.csv')
    
    label_encoders = {}
    for col in ['from', 'to', 'flightType', 'agency']:
        le = LabelEncoder()
        flights[f'{col}_encoded'] = le.fit_transform(flights[col].astype(str))
        label_encoders[col] = le
    
    flights['date'] = pd.to_datetime(flights['date'])
    flights['month'] = flights['date'].dt.month
    flights['day_of_week'] = flights['date'].dt.dayofweek
    
    feature_cols = ['time', 'distance', 'from_encoded', 'to_encoded', 
                    'flightType_encoded', 'agency_encoded', 'month', 'day_of_week']
    
    X = flights[feature_cols].values
    y = flights['price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: ${rmse:.2f}, R²: {r2:.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        'model': model, 'scaler': scaler, 
        'label_encoders': label_encoders, 'feature_columns': feature_cols
    }, f'{MODELS_DIR}/flight_price_model.pkl')
    
    context['ti'].xcom_push(key='flight_metrics', value={'rmse': rmse, 'r2': r2})
    print("✅ Flight model saved!")
    return True


def train_gender_model(**context):
    """Train gender classification model"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    print("🎯 Training gender classifier...")
    
    users = pd.read_csv(f'{DATA_DIR}/processed/users_processed.csv')
    flights = pd.read_csv(f'{DATA_DIR}/processed/flights_processed.csv')
    hotels = pd.read_csv(f'{DATA_DIR}/processed/hotels_processed.csv')
    
    flight_agg = flights.groupby('userCode').agg({
        'price': ['mean', 'sum', 'count'], 'distance': 'mean'
    }).reset_index()
    flight_agg.columns = ['userCode', 'avg_price', 'total_spent', 'flight_count', 'avg_distance']
    
    hotel_agg = hotels.groupby('userCode').agg({
        'total': ['mean', 'count'], 'days': 'mean'
    }).reset_index()
    hotel_agg.columns = ['userCode', 'avg_hotel', 'hotel_count', 'avg_days']
    
    df = users.merge(flight_agg, left_on='code', right_on='userCode', how='left')
    df = df.merge(hotel_agg, on='userCode', how='left').fillna(0)
    
    le_company = LabelEncoder()
    df['company_encoded'] = le_company.fit_transform(df['company'])
    
    le_gender = LabelEncoder()
    y = le_gender.fit_transform(df['gender'])
    
    feature_cols = ['age', 'company_encoded', 'avg_price', 'total_spent', 
                    'flight_count', 'avg_distance', 'avg_hotel', 'hotel_count']
    X = df[feature_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    joblib.dump({
        'model': model, 'scaler': scaler,
        'label_encoders': {'company': le_company},
        'target_encoder': le_gender, 'feature_columns': feature_cols
    }, f'{MODELS_DIR}/gender_classifier.pkl')
    
    context['ti'].xcom_push(key='gender_metrics', value={'accuracy': accuracy, 'f1': f1})
    print("✅ Gender classifier saved!")
    return True


def train_recommender(**context):
    """Train hotel recommendation model"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.neighbors import NearestNeighbors
    
    print("🎯 Training hotel recommender...")
    
    users = pd.read_csv(f'{DATA_DIR}/processed/users_processed.csv')
    hotels = pd.read_csv(f'{DATA_DIR}/processed/hotels_processed.csv')
    
    hotels['hotel_id'] = hotels['name'] + '_' + hotels['place']
    unique_hotels = hotels['hotel_id'].unique()
    
    hotel_mapping = {h: i for i, h in enumerate(unique_hotels)}
    reverse_mapping = {i: h for h, i in hotel_mapping.items()}
    
    n_users = users['code'].max() + 1
    n_hotels = len(unique_hotels)
    
    user_item_matrix = np.zeros((n_users, n_hotels))
    for _, row in hotels.iterrows():
        user_item_matrix[row['userCode'], hotel_mapping[row['hotel_id']]] += row['total']
    
    row_sums = user_item_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    user_item_matrix = user_item_matrix / row_sums
    
    hotel_agg = hotels.groupby('hotel_id').agg({
        'price': 'mean', 'days': 'mean', 'total': 'mean', 'place': 'first'
    }).reset_index()
    
    le_place = LabelEncoder()
    hotel_agg['place_encoded'] = le_place.fit_transform(hotel_agg['place'])
    
    scaler = StandardScaler()
    hotel_features = scaler.fit_transform(hotel_agg[['price', 'days', 'total', 'place_encoded']])
    
    nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn_model.fit(hotel_features)
    
    joblib.dump({
        'user_item_matrix': user_item_matrix,
        'hotel_features': hotel_features,
        'hotel_info': hotel_agg,
        'scaler': scaler,
        'label_encoders': {'place': le_place},
        'hotel_mapping': hotel_mapping,
        'reverse_hotel_mapping': reverse_mapping,
        'nn_model': nn_model
    }, f'{MODELS_DIR}/hotel_recommender.pkl')
    
    print(f"✅ Recommender saved! {n_hotels} hotels indexed")
    return True


def generate_report(**context):
    """Generate training report"""
    ti = context['ti']
    
    data_stats = ti.xcom_pull(key='data_stats', task_ids='validate_data')
    flight_metrics = ti.xcom_pull(key='flight_metrics', task_ids='train_flight_model')
    gender_metrics = ti.xcom_pull(key='gender_metrics', task_ids='train_gender_model')
    
    report = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║              TRAVEL ML PIPELINE REPORT                       ║
    ║              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  DATA STATISTICS                                             ║
    ║    Users:   {data_stats['users']:,}                                          ║
    ║    Flights: {data_stats['flights']:,}                                        ║
    ║    Hotels:  {data_stats['hotels']:,}                                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  FLIGHT PRICE MODEL                                          ║
    ║    RMSE: ${flight_metrics['rmse']:.2f}                                       ║
    ║    R²:   {flight_metrics['r2']:.4f}                                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  GENDER CLASSIFIER                                           ║
    ║    Accuracy: {gender_metrics['accuracy']:.4f}                                ║
    ║    F1 Score: {gender_metrics['f1']:.4f}                                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    print(report)
    
    with open(f'{DATA_DIR}/reports/pipeline_report_{datetime.now().strftime("%Y%m%d")}.txt', 'w') as f:
        f.write(report)
    
    return True


# Define DAG
with DAG(
    'travel_ml_pipeline',
    default_args=default_args,
    description='Travel ML Data Pipeline - Train and Deploy Models',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'travel', 'training'],
) as dag:
    
    # Start
    start = EmptyOperator(task_id='start')
    
    # Data Validation
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )
    
    # Preprocessing
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    
    # Model Training (parallel)
    train_flight = PythonOperator(
        task_id='train_flight_model',
        python_callable=train_flight_model,
    )
    
    train_gender = PythonOperator(
        task_id='train_gender_model',
        python_callable=train_gender_model,
    )
    
    train_hotel = PythonOperator(
        task_id='train_recommender',
        python_callable=train_recommender,
    )
    
    # Report Generation
    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )
    
    # End
    end = EmptyOperator(task_id='end')
    
    # Define dependencies
    start >> validate >> preprocess
    preprocess >> [train_flight, train_gender, train_hotel]
    [train_flight, train_gender, train_hotel] >> report >> end
