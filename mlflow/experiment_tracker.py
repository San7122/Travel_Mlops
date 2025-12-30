"""
MLflow Model Tracking
Track experiments, compare models, and manage model versions
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLflowExperimentTracker:
    def __init__(self, tracking_uri='http://localhost:5001', experiment_name='travel_ml'):
        """Initialize MLflow tracker"""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        print(f"✅ MLflow initialized")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Experiment: {experiment_name}")
    
    def log_data_info(self, df, dataset_name):
        """Log dataset information"""
        with mlflow.start_run(run_name=f"data_info_{dataset_name}", nested=True):
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("num_rows", len(df))
            mlflow.log_param("num_columns", len(df.columns))
            mlflow.log_param("columns", str(df.columns.tolist()))
            
            # Log basic statistics
            stats = df.describe().to_dict()
            for col, col_stats in stats.items():
                for stat_name, value in col_stats.items():
                    if pd.notna(value):
                        mlflow.log_metric(f"{col}_{stat_name}", value)
            
            # Log missing values
            missing = df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    mlflow.log_metric(f"missing_{col}", count)
    
    def train_flight_model_with_tracking(self, flights_df, model_params=None):
        """Train flight price model with MLflow tracking"""
        
        with mlflow.start_run(run_name="flight_price_experiment"):
            # Log data info
            mlflow.log_param("dataset_size", len(flights_df))
            
            # Prepare data
            df = flights_df.copy()
            label_encoders = {}
            
            for col in ['from', 'to', 'flightType', 'agency']:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            feature_cols = ['time', 'distance', 'from_encoded', 'to_encoded',
                           'flightType_encoded', 'agency_encoded', 'month', 'day_of_week']
            
            X = df[feature_cols].values
            y = df['price'].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Log preprocessing params
            mlflow.log_param("features", feature_cols)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("scaler", "StandardScaler")
            
            # Define models to compare
            models = {
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, random_state=42
                )
            }
            
            best_model = None
            best_r2 = -float('inf')
            best_model_name = None
            
            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name, nested=True):
                    # Log model parameters
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_params(model.get_params())
                    
                    # Train
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Log metrics
                    mlflow.log_metric("train_rmse", train_rmse)
                    mlflow.log_metric("test_rmse", test_rmse)
                    mlflow.log_metric("train_mae", train_mae)
                    mlflow.log_metric("test_mae", test_mae)
                    mlflow.log_metric("train_r2", train_r2)
                    mlflow.log_metric("test_r2", test_r2)
                    mlflow.log_metric("cv_r2_mean", cv_mean)
                    mlflow.log_metric("cv_r2_std", cv_std)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"model_{model_name}")
                    
                    print(f"  {model_name}: R²={test_r2:.4f}, RMSE=${test_rmse:.2f}")
                    
                    # Track best model
                    if test_r2 > best_r2:
                        best_r2 = test_r2
                        best_model = model
                        best_model_name = model_name
            
            # Log best model
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_r2", best_r2)
            
            print(f"\n✅ Best model: {best_model_name} with R²={best_r2:.4f}")
            
            return best_model, best_model_name
    
    def register_model(self, run_id, model_name, stage="Staging"):
        """Register a model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_details = mlflow.register_model(model_uri, model_name)
        
        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage=stage
        )
        
        print(f"✅ Model '{model_name}' registered (version {model_details.version}) in {stage}")
        return model_details
    
    def get_best_model(self, metric="test_r2", ascending=False):
        """Get the best model run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            return None
        
        best_run = runs.iloc[0]
        return {
            'run_id': best_run['run_id'],
            metric: best_run[f'metrics.{metric}'],
            'model_type': best_run.get('params.model_type', 'unknown')
        }
    
    def compare_runs(self, metric="test_r2"):
        """Compare all runs in the experiment"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if len(runs) == 0:
            print("No runs found")
            return None
        
        # Select relevant columns
        cols = ['run_id', 'params.model_type', f'metrics.{metric}', 
                'metrics.test_rmse', 'metrics.test_mae']
        available_cols = [c for c in cols if c in runs.columns]
        
        comparison = runs[available_cols].dropna()
        comparison = comparison.sort_values(f'metrics.{metric}', ascending=False)
        
        print("\n📊 Model Comparison:")
        print(comparison.to_string(index=False))
        
        return comparison
    
    def load_model_from_registry(self, model_name, stage="Production"):
        """Load a model from the registry"""
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Loaded model '{model_name}' from {stage}")
        return model
    
    def list_experiments(self):
        """List all experiments"""
        experiments = self.client.search_experiments()
        
        print("\n📋 Experiments:")
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        return experiments


def run_mlflow_tracking():
    """Run complete MLflow tracking example"""
    # Initialize tracker
    tracker = MLflowExperimentTracker(
        tracking_uri='http://localhost:5001',
        experiment_name='flight_price_prediction'
    )
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    flights_df = pd.read_csv(os.path.join(data_dir, 'flights.csv'))
    
    print(f"\n📊 Loaded {len(flights_df)} flight records")
    
    # Train models with tracking
    print("\n🎯 Training models with MLflow tracking...")
    best_model, best_model_name = tracker.train_flight_model_with_tracking(flights_df)
    
    # Compare runs
    tracker.compare_runs()
    
    # Get best model info
    best_run = tracker.get_best_model()
    print(f"\n🏆 Best model: {best_run}")
    
    return tracker


if __name__ == "__main__":
    print("="*60)
    print("MLflow Experiment Tracking")
    print("="*60)
    
    # Check if MLflow server is running
    print("\n⚠️  Make sure MLflow server is running:")
    print("   mlflow server --host 0.0.0.0 --port 5001")
    print("\n   Or use Docker Compose to start all services")
    
    # Run tracking
    try:
        tracker = run_mlflow_tracking()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTo run locally without MLflow server:")
        print("  1. Start MLflow: mlflow ui --port 5001")
        print("  2. Run this script again")
