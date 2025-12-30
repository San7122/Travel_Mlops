"""
Flask REST API for Travel ML Models
Serves flight price prediction, gender classification, and hotel recommendations
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

app = Flask(__name__)
CORS(app)

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FLIGHT_MODEL_PATH = os.path.join(MODELS_DIR, 'flight_price_model.pkl')
GENDER_MODEL_PATH = os.path.join(MODELS_DIR, 'gender_classifier.pkl')
HOTEL_MODEL_PATH = os.path.join(MODELS_DIR, 'hotel_recommender.pkl')

# Global model instances
flight_model = None
gender_model = None
hotel_model = None


def load_models():
    """Load all ML models"""
    global flight_model, gender_model, hotel_model
    
    try:
        if os.path.exists(FLIGHT_MODEL_PATH):
            flight_model = joblib.load(FLIGHT_MODEL_PATH)
            print("[OK] Flight price model loaded")
    except Exception as e:
        print(f"[WARNING] Could not load flight model: {e}")
    
    try:
        if os.path.exists(GENDER_MODEL_PATH):
            gender_model = joblib.load(GENDER_MODEL_PATH)
            print("[OK] Gender classifier loaded")
    except Exception as e:
        print(f"[WARNING] Could not load gender model: {e}")
    
    try:
        if os.path.exists(HOTEL_MODEL_PATH):
            hotel_model = joblib.load(HOTEL_MODEL_PATH)
            print("[OK] Hotel recommender loaded")
    except Exception as e:
        print(f"[WARNING] Could not load hotel model: {e}")


# ==================== ROUTES ====================

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Travel ML API',
        'version': '1.0.0',
        'endpoints': {
            'flight_price': '/api/v1/predict/flight-price',
            'gender': '/api/v1/predict/gender',
            'hotel_recommendations': '/api/v1/recommend/hotels',
            'health': '/health'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'flight_price': flight_model is not None,
            'gender_classifier': gender_model is not None,
            'hotel_recommender': hotel_model is not None
        }
    })


# ==================== FLIGHT PRICE PREDICTION ====================

@app.route('/api/v1/predict/flight-price', methods=['POST'])
def predict_flight_price():
    """
    Predict flight price
    
    Request body:
    {
        "from": "New York",
        "to": "Los Angeles",
        "flightType": "business",
        "agency": "CloudFy",
        "time": 5.5,
        "distance": 2800,
        "month": 6,
        "day_of_week": 2,
        "is_weekend": 0
    }
    """
    if flight_model is None:
        return jsonify({'error': 'Flight price model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['from', 'to', 'flightType', 'distance']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set defaults for optional fields
        data.setdefault('time', data['distance'] / 500)
        data.setdefault('agency', 'Unknown')
        data.setdefault('month', 6)
        data.setdefault('day_of_week', 2)
        data.setdefault('is_weekend', 0)
        
        # Make prediction
        model = flight_model['model']
        scaler = flight_model['scaler']
        label_encoders = flight_model['label_encoders']
        feature_columns = flight_model['feature_columns']
        
        # Encode features
        encoded_features = []
        for col in feature_columns:
            if col.endswith('_encoded'):
                original_col = col.replace('_encoded', '')
                if original_col in data and original_col in label_encoders:
                    try:
                        val = label_encoders[original_col].transform([data[original_col]])[0]
                    except:
                        val = 0
                else:
                    val = 0
                encoded_features.append(val)
            else:
                encoded_features.append(data.get(col, 0))
        
        X = np.array(encoded_features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'currency': 'USD',
            'input': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== GENDER CLASSIFICATION ====================

@app.route('/api/v1/predict/gender', methods=['POST'])
def predict_gender():
    """
    Predict user gender
    
    Request body:
    {
        "age": 35,
        "company": "TechCorp",
        "avg_flight_price": 800,
        "total_flight_spent": 5000,
        "flight_count": 6,
        "avg_distance": 2000,
        "total_distance": 12000,
        "avg_flight_time": 4.5,
        "first_class_count": 2,
        "avg_hotel_total": 500,
        "total_hotel_spent": 3000,
        "hotel_booking_count": 6,
        "avg_stay_days": 3,
        "total_stay_days": 18,
        "avg_hotel_price_per_day": 180
    }
    """
    if gender_model is None:
        return jsonify({'error': 'Gender classifier not loaded'}), 503
    
    try:
        data = request.get_json()
        
        model = gender_model['model']
        scaler = gender_model['scaler']
        label_encoders = gender_model['label_encoders']
        feature_columns = gender_model['feature_columns']
        target_encoder = gender_model['target_encoder']
        
        # Prepare features
        feature_values = []
        for col in feature_columns:
            if col == 'company_encoded' and 'company' in data:
                try:
                    val = label_encoders['company'].transform([data['company']])[0]
                except:
                    val = 0
            else:
                val = data.get(col, 0)
            feature_values.append(val)
        
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        gender = target_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            'predicted_gender': gender,
            'confidence': float(max(probability)),
            'probabilities': dict(zip(target_encoder.classes_.tolist(), probability.tolist()))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== HOTEL RECOMMENDATIONS ====================

@app.route('/api/v1/recommend/hotels', methods=['POST'])
def recommend_hotels():
    """
    Get hotel recommendations
    
    Request body:
    {
        "user_code": 100,
        "method": "hybrid",  // collaborative, content, hybrid
        "n_recommendations": 5,
        "preferences": {
            "price": 150,
            "days": 5,
            "place": "Miami"
        }
    }
    """
    if hotel_model is None:
        return jsonify({'error': 'Hotel recommender not loaded'}), 503
    
    try:
        data = request.get_json()
        
        user_code = data.get('user_code')
        method = data.get('method', 'hybrid')
        n_recommendations = data.get('n_recommendations', 5)
        preferences = data.get('preferences')
        
        # Get user-item matrix and other components
        user_item_matrix = hotel_model['user_item_matrix']
        hotel_features = hotel_model['hotel_features']
        hotel_info = hotel_model['hotel_info']
        scaler = hotel_model['scaler']
        nn_model = hotel_model['nn_model']
        reverse_hotel_mapping = hotel_model['reverse_hotel_mapping']
        label_encoders = hotel_model['label_encoders']
        
        recommendations = []
        
        if method == 'content' and preferences:
            # Content-based with preferences
            pref_vector = np.array([
                preferences.get('price', 200),
                preferences.get('days', 3),
                preferences.get('total', 600),
                0
            ]).reshape(1, -1)
            
            if 'place' in preferences and 'place' in label_encoders:
                try:
                    pref_vector[0, 3] = label_encoders['place'].transform([preferences['place']])[0]
                except:
                    pass
            
            pref_vector_scaled = scaler.transform(pref_vector)
            distances, indices = nn_model.kneighbors(pref_vector_scaled)
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(hotel_info):
                    info = hotel_info.iloc[idx]
                    recommendations.append({
                        'hotel_id': info['hotel_id'],
                        'name': info['hotel_id'].split('_')[0],
                        'place': info['place'],
                        'avg_price': float(info['price']),
                        'similarity': float(1 - dist)
                    })
        
        elif user_code is not None and user_code < user_item_matrix.shape[0]:
            # Collaborative filtering
            from sklearn.metrics.pairwise import cosine_similarity
            
            user_vector = user_item_matrix[user_code].reshape(1, -1)
            user_similarities = cosine_similarity(user_vector, user_item_matrix)[0]
            similar_users = np.argsort(user_similarities)[::-1][1:11]
            
            rec_scores = np.zeros(user_item_matrix.shape[1])
            for similar_user in similar_users:
                similarity = user_similarities[similar_user]
                rec_scores += similarity * user_item_matrix[similar_user]
            
            already_booked = np.where(user_item_matrix[user_code] > 0)[0]
            rec_scores[already_booked] = -1
            
            top_indices = np.argsort(rec_scores)[::-1][:n_recommendations]
            
            for idx in top_indices:
                if rec_scores[idx] > 0:
                    hotel_id = reverse_hotel_mapping[idx]
                    info = hotel_info[hotel_info['hotel_id'] == hotel_id].iloc[0]
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'name': hotel_id.split('_')[0],
                        'place': info['place'],
                        'avg_price': float(info['price']),
                        'score': float(rec_scores[idx])
                    })
        
        return jsonify({
            'recommendations': recommendations[:n_recommendations],
            'method': method,
            'user_code': user_code
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/hotels/popular', methods=['GET'])
def get_popular_hotels():
    """Get most popular hotels"""
    if hotel_model is None:
        return jsonify({'error': 'Hotel recommender not loaded'}), 503
    
    try:
        n = request.args.get('n', 10, type=int)
        hotel_info = hotel_model['hotel_info']
        
        # Return top hotels by average price (as a proxy for popularity)
        top_hotels = hotel_info.nlargest(n, 'total')
        
        results = []
        for _, row in top_hotels.iterrows():
            results.append({
                'hotel_id': row['hotel_id'],
                'name': row['hotel_id'].split('_')[0],
                'place': row['place'],
                'avg_price': float(row['price'])
            })
        
        return jsonify({'popular_hotels': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n[STARTING] Travel ML API...")
    load_models()
    
    print("\n[ENDPOINTS] Available endpoints:")
    print("  - GET  /              : API info")
    print("  - GET  /health        : Health check")
    print("  - POST /api/v1/predict/flight-price : Predict flight price")
    print("  - POST /api/v1/predict/gender       : Predict gender")
    print("  - POST /api/v1/recommend/hotels     : Get hotel recommendations")
    print("  - GET  /api/v1/hotels/popular       : Get popular hotels")
    
    app.run(host='0.0.0.0', port=5001, debug=True)

# app.py (create at project root)
import streamlit.web.cli as stcli
import sys
sys.argv = ["streamlit", "run", "streamlit/app.py", "--server.port", "8501"]
if __name__ == "__main__":
    sys.exit(stcli.main())