"""
Algorithm package for movie recommendation system.

This package contains various recommendation algorithms including
collaborative filtering and content-based filtering.
"""

from .collaborative_filtering import (
    CollaborativeFilter,
    create_recommender,
    get_recommendations,
    load_users_data,
    get_recommendations_with_users_data
)

__all__ = [
    'CollaborativeFilter',
    'create_recommender', 
    'get_recommendations',
    'load_users_data',
    'get_recommendations_with_users_data'
]