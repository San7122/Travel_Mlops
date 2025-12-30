"""
Generate synthetic travel datasets for MLOps capstone project
Datasets: users.csv, flights.csv, hotels.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 500
NUM_FLIGHTS = 2000
NUM_HOTELS = 1500

# Sample data
COMPANIES = ['TechCorp', 'FinanceHub', 'HealthCare Inc', 'EduWorld', 'RetailMax', 
             'MediaGroup', 'ConsultPro', 'ManufactureX', 'LogiTrans', 'FoodChain']
FIRST_NAMES_MALE = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 
                    'Joseph', 'Thomas', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark']
FIRST_NAMES_FEMALE = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth', 
                      'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Margaret']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
              'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson']
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
          'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
          'Fort Worth', 'Columbus', 'Charlotte', 'Seattle', 'Denver', 'Boston', 'Miami', 'Atlanta']
FLIGHT_TYPES = ['economic', 'firstClass', 'business']
AGENCIES = ['Rainbow', 'CloudFy', 'FlyingDrops', 'Cloudy', 'SkyWay', 'JetSet', 'AirGo']
HOTEL_NAMES = ['Grand Hotel', 'Royal Inn', 'Comfort Suites', 'Plaza Hotel', 'Park Hyatt',
               'Marriott', 'Hilton', 'Holiday Inn', 'Best Western', 'Sheraton',
               'Radisson', 'Westin', 'InterContinental', 'Crown Plaza', 'Ramada']

def generate_users():
    """Generate users dataset"""
    users = []
    for i in range(1, NUM_USERS + 1):
        gender = random.choice(['male', 'female'])
        if gender == 'male':
            first_name = random.choice(FIRST_NAMES_MALE)
        else:
            first_name = random.choice(FIRST_NAMES_FEMALE)
        last_name = random.choice(LAST_NAMES)
        
        users.append({
            'code': i,
            'company': random.choice(COMPANIES),
            'name': f"{first_name} {last_name}",
            'gender': gender,
            'age': random.randint(22, 65)
        })
    
    return pd.DataFrame(users)

def generate_flights(users_df):
    """Generate flights dataset"""
    flights = []
    user_codes = users_df['code'].tolist()
    
    for i in range(1, NUM_FLIGHTS + 1):
        from_city = random.choice(CITIES)
        to_city = random.choice([c for c in CITIES if c != from_city])
        flight_type = random.choice(FLIGHT_TYPES)
        
        # Calculate realistic distance (random but consistent)
        distance = random.randint(500, 5000)
        
        # Calculate time based on distance (approx 500 miles/hour)
        time_hours = round(distance / 500 + random.uniform(0.5, 2), 2)
        
        # Calculate price based on distance, type, and some randomness
        base_price = distance * 0.15
        if flight_type == 'firstClass':
            price = base_price * random.uniform(2.5, 4.0)
        elif flight_type == 'business':
            price = base_price * random.uniform(1.5, 2.5)
        else:
            price = base_price * random.uniform(0.8, 1.2)
        
        # Random date in 2023-2024
        start_date = datetime(2023, 1, 1)
        random_days = random.randint(0, 730)
        flight_date = start_date + timedelta(days=random_days)
        
        flights.append({
            'travelCode': i,
            'userCode': random.choice(user_codes),
            'from': from_city,
            'to': to_city,
            'flightType': flight_type,
            'price': round(price, 2),
            'time': time_hours,
            'distance': distance,
            'agency': random.choice(AGENCIES),
            'date': flight_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(flights)

def generate_hotels(flights_df):
    """Generate hotels dataset linked to flights"""
    hotels = []
    
    # Use existing travel codes and user codes from flights
    flight_records = flights_df[['travelCode', 'userCode', 'to']].values.tolist()
    
    for i in range(NUM_HOTELS):
        if i < len(flight_records):
            travel_code, user_code, place = flight_records[i]
        else:
            # Generate additional hotel bookings
            idx = random.randint(0, len(flight_records) - 1)
            travel_code, user_code, place = flight_records[idx]
            travel_code = NUM_FLIGHTS + i + 1
        
        days = random.randint(1, 14)
        price_per_day = random.uniform(80, 500)
        total = round(days * price_per_day, 2)
        
        # Random date
        start_date = datetime(2023, 1, 1)
        random_days = random.randint(0, 730)
        hotel_date = start_date + timedelta(days=random_days)
        
        hotels.append({
            'travelCode': travel_code,
            'userCode': user_code,
            'name': random.choice(HOTEL_NAMES),
            'place': place,
            'days': days,
            'price': round(price_per_day, 2),
            'total': total,
            'date': hotel_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(hotels)

if __name__ == "__main__":
    print("Generating Users Dataset...")
    users_df = generate_users()
    users_df.to_csv('users.csv', index=False)
    print(f"Generated {len(users_df)} users")
    
    print("\nGenerating Flights Dataset...")
    flights_df = generate_flights(users_df)
    flights_df.to_csv('flights.csv', index=False)
    print(f"Generated {len(flights_df)} flights")
    
    print("\nGenerating Hotels Dataset...")
    hotels_df = generate_hotels(flights_df)
    hotels_df.to_csv('hotels.csv', index=False)
    print(f"Generated {len(hotels_df)} hotels")
    
    print("\n✅ All datasets generated successfully!")
    print("\nDataset Summaries:")
    print(f"\nUsers: {users_df.shape}")
    print(users_df.head())
    print(f"\nFlights: {flights_df.shape}")
    print(flights_df.head())
    print(f"\nHotels: {hotels_df.shape}")
    print(hotels_df.head())
