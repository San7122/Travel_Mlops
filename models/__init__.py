"""
Travel ML Models Package
Contains regression, classification, and recommendation models
"""
from .flight_price_model import FlightPricePredictor
from .gender_classifier import GenderClassifier
from .hotel_recommender import HotelRecommender

__all__ = ['FlightPricePredictor', 'GenderClassifier', 'HotelRecommender']
