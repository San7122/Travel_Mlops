"""
Flight Price Prediction Model
Regression model to predict flight prices based on various features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import warnings
import os
warnings.filterwarnings('ignore')

class FlightPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_data(self, filepath='../data/flights.csv'):
        """Load and return flight data"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} flight records")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the flight data for modeling"""
        df = df.copy()
        
        # Handle categorical columns
        categorical_cols = ['from', 'to', 'flightType', 'agency']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Extract date features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Select features for model
        self.feature_columns = [
            'time', 'distance', 
            'from_encoded', 'to_encoded', 
            'flightType_encoded', 'agency_encoded',
            'month', 'day_of_week', 'is_weekend'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix and target"""
        X = df[self.feature_columns].values
        y = df['price'].values
        return X, y
    
    def train(self, df, model_type='random_forest'):
        """Train the regression model"""
        # Preprocess
        df = self.preprocess_data(df)
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        self.model = models.get(model_type, models['random_forest'])
        
        # Train
        print(f"\nTraining {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        print("\n[MODEL PERFORMANCE]")
        print(f"  Train RMSE: ${metrics['train_rmse']:.2f}")
        print(f"  Test RMSE:  ${metrics['test_rmse']:.2f}")
        print(f"  Train MAE:  ${metrics['train_mae']:.2f}")
        print(f"  Test MAE:   ${metrics['test_mae']:.2f}")
        print(f"  Train R²:   {metrics['train_r2']:.4f}")
        print(f"  Test R²:    {metrics['test_r2']:.4f}")
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n[FEATURE IMPORTANCE]")
            print(importance_df.to_string(index=False))
        
        return metrics
    
    def predict(self, features_dict):
        """Make prediction for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode categorical features
        encoded_features = []
        for col in self.feature_columns:
            if col.endswith('_encoded'):
                original_col = col.replace('_encoded', '')
                if original_col in features_dict:
                    if original_col in self.label_encoders:
                        try:
                            val = self.label_encoders[original_col].transform([features_dict[original_col]])[0]
                        except:
                            val = 0
                    else:
                        val = 0
                    encoded_features.append(val)
            else:
                encoded_features.append(features_dict.get(col, 0))
        
        X = np.array(encoded_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return round(prediction, 2)
    
    def save_model(self, filepath='flight_price_model.pkl'):
        """Save the trained model"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_artifacts, filepath)
        print(f"\n[SUCCESS] Model saved to {filepath}")
    
    def load_model(self, filepath='flight_price_model.pkl'):
        """Load a trained model"""
        model_artifacts = joblib.load(filepath)
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        self.feature_columns = model_artifacts['feature_columns']
        print(f"[SUCCESS] Model loaded from {filepath}")
    
    def train_with_mlflow(self, df, model_type='random_forest', experiment_name='flight_price_prediction'):
        """Train model with MLflow tracking"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{model_type}_run"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("features", self.feature_columns)
            
            # Train model
            metrics = self.train(df, model_type)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            print(f"\n[SUCCESS] Model logged to MLflow experiment: {experiment_name}")
        
        return metrics


def compare_models(df):
    """Compare different regression models"""
    model_types = ['linear', 'ridge', 'random_forest', 'gradient_boosting']
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training: {model_type.upper()}")
        print('='*50)
        
        predictor = FlightPricePredictor()
        metrics = predictor.train(df, model_type)
        metrics['model'] = model_type
        results.append(metrics)
    
    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(results_df[['model', 'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    # Initialize predictor
    predictor = FlightPricePredictor()
    
    # Load data
    df = predictor.load_data('../data/flights.csv')
    
    # Compare models
    print("\n[INFO] Comparing Different Models...")
    comparison = compare_models(df)
    
    # Train best model (Random Forest)
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL (Random Forest)")
    print("="*60)
    
    predictor = FlightPricePredictor()
    predictor.train(df, 'random_forest')
    
    # Save model
    predictor.save_model('flight_price_model.pkl')
    
    # Test prediction
    print("\n[TEST] Testing Prediction:")
    test_input = {
        'from': 'New York',
        'to': 'Los Angeles',
        'flightType': 'business',
        'agency': 'CloudFy',
        'time': 5.5,
        'distance': 2800,
        'month': 6,
        'day_of_week': 2,
        'is_weekend': 0
    }
    
    predicted_price = predictor.predict(test_input)
    print(f"  Input: {test_input}")
    print(f"  Predicted Price: ${predicted_price}")
