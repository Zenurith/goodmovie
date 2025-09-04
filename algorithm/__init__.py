"""
Algorithm package for movie recommendation system.

This package contains various recommendation algorithms including
collaborative filtering and content-based filtering.
"""

from .collaborative_filtering import (
    CollaborativeFilter,
    create_recommender,
    get_recommendations
)

__all__ = [
    'CollaborativeFilter',
    'create_recommender', 
    'get_recommendations'
]