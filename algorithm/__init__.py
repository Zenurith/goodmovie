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

from .content_based import (
    ContentBasedRecommender,
    create_content_based_recommender,
    get_content_based_recommendations
)

from .tfidf_content import (
    TFIDFMovieSearch,
    create_tfidf_search_engine,
    search_movies_tfidf
)

__all__ = [
    'CollaborativeFilter',
    'create_recommender', 
    'get_recommendations',
    'load_users_data',
    'get_recommendations_with_users_data',
    'ContentBasedRecommender',
    'create_content_based_recommender',
    'get_content_based_recommendations',
    'TFIDFMovieSearch',
    'create_tfidf_search_engine',
    'search_movies_tfidf'
]