"""
Gender Classification Model
Classify user gender based on features from travel data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import joblib
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


class GenderClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_encoder = LabelEncoder()
        
    def load_data(self, users_path='../data/users.csv', 
                  flights_path='../data/flights.csv',
                  hotels_path='../data/hotels.csv'):
        """Load and merge all datasets"""
        users = pd.read_csv(users_path)
        flights = pd.read_csv(flights_path)
        hotels = pd.read_csv(hotels_path)
        
        print(f"Loaded {len(users)} users, {len(flights)} flights, {len(hotels)} hotels")
        
        # Aggregate flight features per user
        flight_agg = flights.groupby('userCode').agg({
            'price': ['mean', 'sum', 'count', 'std'],
            'distance': ['mean', 'sum'],
            'time': ['mean', 'sum'],
            'flightType': lambda x: (x == 'firstClass').sum()  # Count of first class flights
        }).reset_index()
        
        flight_agg.columns = [
            'userCode', 'avg_flight_price', 'total_flight_spent', 'flight_count', 'flight_price_std',
            'avg_distance', 'total_distance', 'avg_flight_time', 'total_flight_time', 'first_class_count'
        ]
        
        # Aggregate hotel features per user
        hotel_agg = hotels.groupby('userCode').agg({
            'total': ['mean', 'sum', 'count'],
            'days': ['mean', 'sum'],
            'price': ['mean']
        }).reset_index()
        
        hotel_agg.columns = [
            'userCode', 'avg_hotel_total', 'total_hotel_spent', 'hotel_booking_count',
            'avg_stay_days', 'total_stay_days', 'avg_hotel_price_per_day'
        ]
        
        # Merge all data
        df = users.merge(flight_agg, left_on='code', right_on='userCode', how='left')
        df = df.merge(hotel_agg, on='userCode', how='left')
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for classification"""
        df = df.copy()
        
        # Encode company
        if 'company' in df.columns:
            le = LabelEncoder()
            df['company_encoded'] = le.fit_transform(df['company'].astype(str))
            self.label_encoders['company'] = le
        
        # Define feature columns
        self.feature_columns = [
            'age', 'company_encoded',
            'avg_flight_price', 'total_flight_spent', 'flight_count',
            'avg_distance', 'total_distance', 'avg_flight_time',
            'first_class_count', 'avg_hotel_total', 'total_hotel_spent',
            'hotel_booking_count', 'avg_stay_days', 'total_stay_days',
            'avg_hotel_price_per_day'
        ]
        
        # Filter to available columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix and target"""
        X = df[self.feature_columns].values
        y = self.target_encoder.fit_transform(df['gender'])
        return X, y
    
    def train(self, df, model_type='random_forest'):
        """Train the classification model"""
        # Preprocess
        df = self.preprocess_data(df)
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        self.model = models.get(model_type, models['random_forest'])
        
        # Train
        print(f"\nTraining {model_type} classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        print("\n[MODEL PERFORMANCE]")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Precision:      {metrics['test_precision']:.4f}")
        print(f"  Recall:         {metrics['test_recall']:.4f}")
        print(f"  F1 Score:       {metrics['test_f1']:.4f}")
        
        # Classification report
        print("\n[CLASSIFICATION REPORT]")
        target_names = self.target_encoder.classes_
        print(classification_report(y_test, y_pred_test, target_names=target_names))
        
        # Confusion matrix
        print("[CONFUSION MATRIX]")
        cm = confusion_matrix(y_test, y_pred_test)
        print(pd.DataFrame(cm, index=target_names, columns=target_names))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n[FEATURE IMPORTANCE]")
            print(importance_df.head(10).to_string(index=False))
        
        return metrics
    
    def predict(self, features_dict):
        """Predict gender for new user"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        feature_values = []
        for col in self.feature_columns:
            if col == 'company_encoded' and 'company' in features_dict:
                try:
                    val = self.label_encoders['company'].transform([features_dict['company']])[0]
                except:
                    val = 0
            else:
                val = features_dict.get(col, 0)
            feature_values.append(val)
        
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        gender = self.target_encoder.inverse_transform([prediction])[0]
        
        return {
            'gender': gender,
            'confidence': float(max(probability)),
            'probabilities': dict(zip(self.target_encoder.classes_, probability.tolist()))
        }
    
    def save_model(self, filepath='gender_classifier.pkl'):
        """Save trained model"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_encoder': self.target_encoder
        }
        joblib.dump(model_artifacts, filepath)
        print(f"\n[SUCCESS] Model saved to {filepath}")
    
    def load_model(self, filepath='gender_classifier.pkl'):
        """Load trained model"""
        model_artifacts = joblib.load(filepath)
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        self.feature_columns = model_artifacts['feature_columns']
        self.target_encoder = model_artifacts['target_encoder']
        print(f"[SUCCESS] Model loaded from {filepath}")
    
    def train_with_mlflow(self, df, model_type='random_forest', experiment_name='gender_classification'):
        """Train with MLflow tracking"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{model_type}_run"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("features", str(self.feature_columns))
            
            metrics = self.train(df, model_type)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(self.model, "model")
            
            print(f"\n[SUCCESS] Model logged to MLflow experiment: {experiment_name}")
        
        return metrics


def compare_classifiers(df):
    """Compare different classification models"""
    model_types = ['logistic', 'random_forest', 'gradient_boosting']
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training: {model_type.upper()}")
        print('='*50)
        
        classifier = GenderClassifier()
        metrics = classifier.train(df, model_type)
        metrics['model'] = model_type
        results.append(metrics)
    
    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("CLASSIFIER COMPARISON SUMMARY")
    print("="*60)
    print(results_df[['model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']].to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    # Initialize classifier
    classifier = GenderClassifier()
    
    # Load data
    df = classifier.load_data(
        '../data/users.csv',
        '../data/flights.csv',
        '../data/hotels.csv'
    )
    
    print(f"\nMerged dataset shape: {df.shape}")
    print(f"Gender distribution:\n{df['gender'].value_counts()}")
    
    # Compare models
    print("\n[INFO] Comparing Different Classifiers...")
    comparison = compare_classifiers(df)
    
    # Train best model
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL (Random Forest)")
    print("="*60)
    
    classifier = GenderClassifier()
    classifier.train(df, 'random_forest')
    
    # Save model
    classifier.save_model('gender_classifier.pkl')
    
    # Test prediction
    print("\n[TEST] Testing Prediction:")
    test_input = {
        'age': 35,
        'company': 'TechCorp',
        'avg_flight_price': 800,
        'total_flight_spent': 5000,
        'flight_count': 6,
        'avg_distance': 2000,
        'total_distance': 12000,
        'avg_flight_time': 4.5,
        'first_class_count': 2,
        'avg_hotel_total': 500,
        'total_hotel_spent': 3000,
        'hotel_booking_count': 6,
        'avg_stay_days': 3,
        'total_stay_days': 18,
        'avg_hotel_price_per_day': 180
    }
    
    result = classifier.predict(test_input)
    print(f"  Predicted Gender: {result['gender']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: {result['probabilities']}")
