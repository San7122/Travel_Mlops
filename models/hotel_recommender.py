"""
Hotel Recommendation Model
Recommend hotels based on user preferences and historical data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import joblib
import warnings
warnings.filterwarnings('ignore')


class HotelRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.hotel_features = None
        self.user_profiles = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.hotel_mapping = {}
        self.reverse_hotel_mapping = {}
        self.nn_model = None
        
    def load_data(self, users_path='../data/users.csv',
                  flights_path='../data/flights.csv',
                  hotels_path='../data/hotels.csv'):
        """Load all datasets"""
        self.users = pd.read_csv(users_path)
        self.flights = pd.read_csv(flights_path)
        self.hotels = pd.read_csv(hotels_path)
        
        print(f"Loaded {len(self.users)} users, {len(self.flights)} flights, {len(self.hotels)} hotels")
        return self.users, self.flights, self.hotels
    
    def build_user_item_matrix(self):
        """Build user-hotel interaction matrix"""
        # Create unique hotel IDs
        self.hotels['hotel_id'] = self.hotels['name'] + '_' + self.hotels['place']
        unique_hotels = self.hotels['hotel_id'].unique()
        
        # Create mappings
        self.hotel_mapping = {hotel: idx for idx, hotel in enumerate(unique_hotels)}
        self.reverse_hotel_mapping = {idx: hotel for hotel, idx in self.hotel_mapping.items()}
        
        # Create user-item matrix (users x hotels)
        n_users = self.users['code'].max() + 1
        n_hotels = len(unique_hotels)
        
        self.user_item_matrix = np.zeros((n_users, n_hotels))
        
        # Fill matrix with booking counts/ratings
        for _, row in self.hotels.iterrows():
            user_idx = row['userCode']
            hotel_idx = self.hotel_mapping[row['hotel_id']]
            # Use total spent as implicit rating
            self.user_item_matrix[user_idx, hotel_idx] += row['total']
        
        # Normalize
        row_sums = self.user_item_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.user_item_matrix = self.user_item_matrix / row_sums
        
        print(f"User-Item matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def build_hotel_features(self):
        """Build feature matrix for hotels"""
        hotel_agg = self.hotels.groupby('hotel_id').agg({
            'price': 'mean',
            'days': 'mean',
            'total': 'mean',
            'place': 'first'
        }).reset_index()
        
        # Encode place
        le = LabelEncoder()
        hotel_agg['place_encoded'] = le.fit_transform(hotel_agg['place'])
        self.label_encoders['place'] = le
        
        # Create feature matrix
        feature_cols = ['price', 'days', 'total', 'place_encoded']
        self.hotel_features = hotel_agg[feature_cols].values
        self.hotel_features = self.scaler.fit_transform(self.hotel_features)
        
        self.hotel_info = hotel_agg
        print(f"Hotel features shape: {self.hotel_features.shape}")
        return self.hotel_features
    
    def build_user_profiles(self):
        """Build user preference profiles"""
        # Aggregate user travel patterns
        user_agg = self.hotels.groupby('userCode').agg({
            'price': ['mean', 'std'],
            'days': ['mean', 'sum'],
            'total': ['mean', 'sum']
        }).reset_index()
        
        user_agg.columns = ['userCode', 'avg_price', 'price_std', 
                           'avg_days', 'total_days', 'avg_total', 'total_spent']
        
        # Merge with user demographics
        self.user_profiles = self.users.merge(
            user_agg, 
            left_on='code', 
            right_on='userCode', 
            how='left'
        ).fillna(0)
        
        print(f"User profiles shape: {self.user_profiles.shape}")
        return self.user_profiles
    
    def train(self):
        """Train the recommendation model"""
        print("\n[INFO] Building recommendation system...")
        
        # Build matrices
        self.build_user_item_matrix()
        self.build_hotel_features()
        self.build_user_profiles()
        
        # Train nearest neighbors for content-based filtering
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.nn_model.fit(self.hotel_features)
        
        print("\n[SUCCESS] Recommendation model trained!")
        return self
    
    def get_collaborative_recommendations(self, user_code, n_recommendations=5):
        """Get recommendations using collaborative filtering"""
        if user_code >= self.user_item_matrix.shape[0]:
            return []
        
        user_vector = self.user_item_matrix[user_code].reshape(1, -1)
        
        # Find similar users
        user_similarities = cosine_similarity(user_vector, self.user_item_matrix)[0]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        
        # Aggregate their preferences
        recommendations = np.zeros(self.user_item_matrix.shape[1])
        for similar_user in similar_users:
            similarity = user_similarities[similar_user]
            recommendations += similarity * self.user_item_matrix[similar_user]
        
        # Remove already booked hotels
        already_booked = np.where(self.user_item_matrix[user_code] > 0)[0]
        recommendations[already_booked] = -1
        
        # Get top recommendations
        top_indices = np.argsort(recommendations)[::-1][:n_recommendations]
        
        results = []
        for idx in top_indices:
            if recommendations[idx] > 0:
                hotel_id = self.reverse_hotel_mapping[idx]
                hotel_info = self.hotel_info[self.hotel_info['hotel_id'] == hotel_id].iloc[0]
                results.append({
                    'hotel_id': hotel_id,
                    'name': hotel_id.split('_')[0],
                    'place': hotel_info['place'],
                    'avg_price': hotel_info['price'],
                    'score': float(recommendations[idx])
                })
        
        return results
    
    def get_content_based_recommendations(self, user_code=None, preferences=None, n_recommendations=5):
        """Get recommendations based on hotel features"""
        if preferences:
            # Build preference vector
            pref_vector = np.array([
                preferences.get('price', 200),
                preferences.get('days', 3),
                preferences.get('total', 600),
                0  # place_encoded (will be overwritten)
            ]).reshape(1, -1)
            
            if 'place' in preferences and 'place' in self.label_encoders:
                try:
                    pref_vector[0, 3] = self.label_encoders['place'].transform([preferences['place']])[0]
                except:
                    pref_vector[0, 3] = 0
            
            pref_vector_scaled = self.scaler.transform(pref_vector)
        elif user_code is not None:
            # Use user's historical preferences
            user_hotels = self.hotels[self.hotels['userCode'] == user_code]
            if len(user_hotels) == 0:
                return []
            
            pref_vector = np.array([
                user_hotels['price'].mean(),
                user_hotels['days'].mean(),
                user_hotels['total'].mean(),
                0
            ]).reshape(1, -1)
            pref_vector_scaled = self.scaler.transform(pref_vector)
        else:
            return []
        
        # Find nearest hotels
        distances, indices = self.nn_model.kneighbors(pref_vector_scaled)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.hotel_info):
                hotel_info = self.hotel_info.iloc[idx]
                results.append({
                    'hotel_id': hotel_info['hotel_id'],
                    'name': hotel_info['hotel_id'].split('_')[0],
                    'place': hotel_info['place'],
                    'avg_price': hotel_info['price'],
                    'similarity': float(1 - dist)
                })
        
        return results[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_code, preferences=None, n_recommendations=5):
        """Get hybrid recommendations combining collaborative and content-based"""
        collab_recs = self.get_collaborative_recommendations(user_code, n_recommendations * 2)
        content_recs = self.get_content_based_recommendations(user_code, preferences, n_recommendations * 2)
        
        # Combine and deduplicate
        all_recs = {}
        
        for rec in collab_recs:
            hotel_id = rec['hotel_id']
            all_recs[hotel_id] = {
                **rec,
                'collab_score': rec.get('score', 0),
                'content_score': 0
            }
        
        for rec in content_recs:
            hotel_id = rec['hotel_id']
            if hotel_id in all_recs:
                all_recs[hotel_id]['content_score'] = rec.get('similarity', 0)
            else:
                all_recs[hotel_id] = {
                    **rec,
                    'collab_score': 0,
                    'content_score': rec.get('similarity', 0)
                }
        
        # Calculate hybrid score
        for hotel_id in all_recs:
            all_recs[hotel_id]['hybrid_score'] = (
                0.6 * all_recs[hotel_id]['collab_score'] +
                0.4 * all_recs[hotel_id]['content_score']
            )
        
        # Sort by hybrid score
        sorted_recs = sorted(all_recs.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def recommend(self, user_code=None, preferences=None, method='hybrid', n_recommendations=5):
        """Main recommendation interface"""
        if method == 'collaborative':
            return self.get_collaborative_recommendations(user_code, n_recommendations)
        elif method == 'content':
            return self.get_content_based_recommendations(user_code, preferences, n_recommendations)
        else:
            return self.get_hybrid_recommendations(user_code, preferences, n_recommendations)
    
    def save_model(self, filepath='hotel_recommender.pkl'):
        """Save the recommendation model"""
        model_artifacts = {
            'user_item_matrix': self.user_item_matrix,
            'hotel_features': self.hotel_features,
            'hotel_info': self.hotel_info,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'hotel_mapping': self.hotel_mapping,
            'reverse_hotel_mapping': self.reverse_hotel_mapping,
            'nn_model': self.nn_model,
            'user_profiles': self.user_profiles
        }
        joblib.dump(model_artifacts, filepath)
        print(f"\n[SUCCESS] Model saved to {filepath}")
    
    def load_model(self, filepath='hotel_recommender.pkl'):
        """Load a saved model"""
        model_artifacts = joblib.load(filepath)
        self.user_item_matrix = model_artifacts['user_item_matrix']
        self.hotel_features = model_artifacts['hotel_features']
        self.hotel_info = model_artifacts['hotel_info']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        self.hotel_mapping = model_artifacts['hotel_mapping']
        self.reverse_hotel_mapping = model_artifacts['reverse_hotel_mapping']
        self.nn_model = model_artifacts['nn_model']
        self.user_profiles = model_artifacts['user_profiles']
        print(f"[SUCCESS] Model loaded from {filepath}")
    
    def get_popular_hotels(self, n=10):
        """Get most popular hotels overall"""
        hotel_counts = self.hotels['hotel_id'].value_counts().head(n)
        
        results = []
        for hotel_id, count in hotel_counts.items():
            hotel_info = self.hotel_info[self.hotel_info['hotel_id'] == hotel_id].iloc[0]
            results.append({
                'hotel_id': hotel_id,
                'name': hotel_id.split('_')[0],
                'place': hotel_info['place'],
                'avg_price': hotel_info['price'],
                'booking_count': int(count)
            })
        
        return results
    
    def get_hotels_by_place(self, place, n=5):
        """Get top hotels in a specific location"""
        place_hotels = self.hotels[self.hotels['place'] == place]
        
        if len(place_hotels) == 0:
            return []
        
        hotel_stats = place_hotels.groupby('hotel_id').agg({
            'total': 'count',
            'price': 'mean'
        }).reset_index()
        
        hotel_stats.columns = ['hotel_id', 'bookings', 'avg_price']
        hotel_stats = hotel_stats.sort_values('bookings', ascending=False).head(n)
        
        results = []
        for _, row in hotel_stats.iterrows():
            results.append({
                'hotel_id': row['hotel_id'],
                'name': row['hotel_id'].split('_')[0],
                'place': place,
                'avg_price': row['avg_price'],
                'booking_count': int(row['bookings'])
            })
        
        return results


if __name__ == "__main__":
    # Initialize recommender
    recommender = HotelRecommender()
    
    # Load data
    recommender.load_data(
        '../data/users.csv',
        '../data/flights.csv',
        '../data/hotels.csv'
    )
    
    # Train model
    recommender.train()
    
    # Save model
    recommender.save_model('hotel_recommender.pkl')
    
    # Test recommendations
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS")
    print("="*60)
    
    # Test for a specific user
    user_code = 100
    print(f"\n[USER] Recommendations for User {user_code}:")
    
    print("\n[COLLABORATIVE FILTERING]")
    collab_recs = recommender.recommend(user_code, method='collaborative')
    for i, rec in enumerate(collab_recs, 1):
        print(f"  {i}. {rec['name']} in {rec['place']} (${rec['avg_price']:.2f}/day)")
    
    print("\n[CONTENT-BASED FILTERING]")
    content_recs = recommender.recommend(user_code, method='content')
    for i, rec in enumerate(content_recs, 1):
        print(f"  {i}. {rec['name']} in {rec['place']} (${rec['avg_price']:.2f}/day)")
    
    print("\n[HYBRID RECOMMENDATIONS]")
    hybrid_recs = recommender.recommend(user_code, method='hybrid')
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"  {i}. {rec['name']} in {rec['place']} (${rec['avg_price']:.2f}/day)")
    
    # Test with preferences
    print("\n[PREFERENCE-BASED RECOMMENDATIONS]")
    prefs = {'price': 150, 'days': 5, 'place': 'Miami'}
    pref_recs = recommender.recommend(preferences=prefs, method='content')
    for i, rec in enumerate(pref_recs, 1):
        print(f"  {i}. {rec['name']} in {rec['place']} (${rec['avg_price']:.2f}/day)")
    
    # Popular hotels
    print("\n[MOST POPULAR HOTELS]")
    popular = recommender.get_popular_hotels(5)
    for i, hotel in enumerate(popular, 1):
        print(f"  {i}. {hotel['name']} in {hotel['place']} - {hotel['booking_count']} bookings")
