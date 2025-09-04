#!/usr/bin/env python3
"""
Enhanced Content-Based Movie Recommender System

Improvements over the notebook version:
- Better error handling and validation
- Fuzzy search for movie titles
- Caching for performance
- Modular class-based design
- Configuration management
- Logging support
- API-ready structure
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import difflib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RecommendationConfig:
    """Configuration class for the recommender system"""
    dataset_path: Optional[str] = None
    cache_dir: str = "cache"
    similarity_method: str = "cosine"  # cosine, euclidean
    default_n_recommendations: int = 10
    default_min_rating: float = 5.0
    default_min_votes: int = 50
    fuzzy_search_threshold: int = 70
    cache_enabled: bool = True
    log_level: str = "INFO"


class MovieNotFoundError(Exception):
    """Custom exception for when a movie is not found"""
    pass


class DatasetError(Exception):
    """Custom exception for dataset-related errors"""
    pass


class ContentBasedRecommender:
    """Content-Based Movie Recommender System"""
    
    def __init__(self, config: RecommendationConfig = None):
        """
        Initialize the recommender system
        
        Args:
            config: Configuration object with system settings
        """
        self.config = config or RecommendationConfig()
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.movie_mappings = {}
        self.cache_path = Path(self.config.cache_dir)
        self.cache_path.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load data if dataset path provided
        if self.config.dataset_path:
            self.load_data(self.config.dataset_path)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_dataset(self) -> Optional[str]:
        """
        Find the movie dataset in various possible locations
        
        Returns:
            Path to the dataset if found, None otherwise
        """
        possible_paths = [
            '../data/dataset/movie_features_engineered.csv',
            '../data/dataset/movies_clean.csv',
            '../movie_features_engineered.csv',
            'data/dataset/movie_features_engineered.csv',
            'movie_features_engineered.csv',
            'movies_clean.csv'
        ]
        
        self.logger.info("Searching for movie dataset...")
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found dataset: {path}")
                return path
        
        return None
    
    def load_data(self, dataset_path: str = None) -> bool:
        """
        Load the movie dataset
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DatasetError: If dataset cannot be loaded
        """
        try:
            if dataset_path is None:
                dataset_path = self.find_dataset()
            
            if dataset_path is None:
                raise DatasetError("No movie dataset found in expected locations")
            
            self.movies_df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded: {dataset_path}")
            self.logger.info(f"Shape: {self.movies_df.shape}")
            
            # Validate required columns
            required_cols = ['id', 'title']
            missing_cols = [col for col in required_cols if col not in self.movies_df.columns]
            
            if missing_cols:
                raise DatasetError(f"Missing required columns: {missing_cols}")
            
            # Create movie mappings
            self._create_movie_mappings()
            
            # Prepare features
            self._prepare_features()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise DatasetError(f"Failed to load dataset: {e}")
    
    def _create_movie_mappings(self):
        """Create mappings between movie IDs, indices, and titles"""
        self.movie_mappings = {
            'id_to_idx': {movie_id: idx for idx, movie_id in enumerate(self.movies_df['id'])},
            'idx_to_id': {idx: movie_id for movie_id, idx in 
                         {movie_id: idx for idx, movie_id in enumerate(self.movies_df['id'])}.items()},
            'title_to_idx': {title.lower(): idx for idx, title in enumerate(self.movies_df['title'])},
            'titles': list(self.movies_df['title'])
        }
    
    def _prepare_features(self):
        """Prepare feature matrix from the dataset"""
        try:
            # Get feature columns (exclude non-feature columns)
            exclude_cols = ['id', 'title', 'overview', 'tagline', 'release_date']
            feature_cols = [col for col in self.movies_df.columns if col not in exclude_cols]
            
            self.logger.info(f"Using {len(feature_cols)} features for similarity computation")
            
            # Fill missing values and scale features
            feature_data = self.movies_df[feature_cols].fillna(0)
            self.feature_matrix = self.scaler.fit_transform(feature_data)
            
            self.logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            raise DatasetError(f"Failed to prepare features: {e}")
    
    def _get_cache_filename(self, cache_type: str) -> str:
        """Generate cache filename based on data characteristics"""
        data_hash = hash(str(self.movies_df.shape) + str(self.movies_df.columns.tolist()))
        return f"{cache_type}_{abs(data_hash)}.pkl"
    
    def _load_cache(self, cache_type: str) -> Optional[np.ndarray]:
        """Load cached data if available and valid"""
        if not self.config.cache_enabled:
            return None
        
        cache_file = self.cache_path / self._get_cache_filename(cache_type)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.logger.info(f"Loaded {cache_type} from cache")
                    return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_type}: {e}")
        
        return None
    
    def _save_cache(self, data: np.ndarray, cache_type: str):
        """Save data to cache"""
        if not self.config.cache_enabled:
            return
        
        cache_file = self.cache_path / self._get_cache_filename(cache_type)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info(f"Saved {cache_type} to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_type}: {e}")
    
    def compute_similarity_matrix(self, method: str = None, force_recompute: bool = False):
        """
        Compute similarity matrix between all movies
        
        Args:
            method: Similarity method ('cosine' or 'euclidean')
            force_recompute: Force recomputation even if cache exists
        """
        if method is None:
            method = self.config.similarity_method
        
        if not force_recompute:
            self.similarity_matrix = self._load_cache(f'similarity_{method}')
            if self.similarity_matrix is not None:
                return
        
        self.logger.info(f"Computing {method} similarity matrix...")
        
        if method == 'cosine':
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
        elif method == 'euclidean':
            distances = euclidean_distances(self.feature_matrix)
            self.similarity_matrix = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        self.logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")
        
        # Cache the result
        self._save_cache(self.similarity_matrix, f'similarity_{method}')
    
    def fuzzy_search_movie(self, title: str, threshold: float = None) -> Tuple[str, int, int]:
        """
        Find movie using fuzzy search
        
        Args:
            title: Movie title to search for
            threshold: Minimum similarity threshold (0-100)
            
        Returns:
            Tuple of (matched_title, match_score, movie_index)
            
        Raises:
            MovieNotFoundError: If no suitable match is found
        """
        if threshold is None:
            threshold = self.config.fuzzy_search_threshold / 100.0
        else:
            threshold = threshold / 100.0
        
        # Try exact match first
        title_lower = title.lower()
        if title_lower in self.movie_mappings['title_to_idx']:
            idx = self.movie_mappings['title_to_idx'][title_lower]
            return self.movies_df.iloc[idx]['title'], 100, idx
        
        # Use difflib for fuzzy matching
        matches = []
        for movie_title in self.movie_mappings['titles']:
            similarity = difflib.SequenceMatcher(None, title.lower(), movie_title.lower()).ratio()
            if similarity >= threshold:
                matches.append((movie_title, int(similarity * 100), self.movie_mappings['titles'].index(movie_title)))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if matches:
            matched_title, score, idx = matches[0]
            self.logger.info(f"Fuzzy match: '{title}' -> '{matched_title}' (score: {score}%)")
            return matched_title, score, idx
        
        # If no good matches, try partial matching
        partial_matches = []
        for movie_title in self.movie_mappings['titles']:
            if title.lower() in movie_title.lower() or movie_title.lower() in title.lower():
                partial_matches.append(movie_title)
        
        if partial_matches:
            suggestions = partial_matches[:3]
            raise MovieNotFoundError(
                f"No good match found for '{title}'. "
                f"Did you mean: {', '.join(suggestions)}?"
            )
        else:
            raise MovieNotFoundError(f"No movies found similar to '{title}'")
    
    def get_movie_recommendations(
        self, 
        movie_title: str,
        n_recommendations: int = None,
        min_rating: float = None,
        min_votes: int = None,
        exclude_seen: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get content-based recommendations for a movie
        
        Args:
            movie_title: Title of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating threshold
            min_votes: Minimum vote count threshold
            exclude_seen: Exclude the input movie from results
            
        Returns:
            List of recommendation dictionaries
            
        Raises:
            MovieNotFoundError: If movie is not found
            ValueError: If similarity matrix is not computed
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Set defaults
        if n_recommendations is None:
            n_recommendations = self.config.default_n_recommendations
        if min_rating is None:
            min_rating = self.config.default_min_rating
        if min_votes is None:
            min_votes = self.config.default_min_votes
        
        # Find the movie
        try:
            matched_title, match_score, movie_idx = self.fuzzy_search_movie(movie_title)
        except MovieNotFoundError:
            raise
        
        target_movie = self.movies_df.iloc[movie_idx]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity
        if exclude_seen:
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude self
        else:
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Apply filters and collect recommendations
        recommendations = []
        
        for idx, score in sim_scores:
            if len(recommendations) >= n_recommendations:
                break
                
            movie_data = self.movies_df.iloc[idx]
            
            # Get movie rating and votes
            rating = self._get_movie_rating(movie_data)
            votes = movie_data.get('vote_count', 0)
            
            # Apply filters
            if rating < min_rating or votes < min_votes:
                continue
            
            # Get genres
            genres = self._extract_genres(movie_data)
            
            recommendations.append({
                'id': movie_data.get('id'),
                'title': movie_data.get('title'),
                'rating': rating,
                'votes': votes,
                'genres': genres,
                'similarity': score,
                'year': self._extract_year(movie_data)
            })
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for '{matched_title}'")
        return recommendations
    
    def get_genre_recommendations(
        self, 
        genre: str,
        n_recommendations: int = None,
        min_rating: float = None
    ) -> List[Dict[str, Any]]:
        """
        Get top movies by genre
        
        Args:
            genre: Genre name
            n_recommendations: Number of recommendations
            min_rating: Minimum rating threshold
            
        Returns:
            List of recommendation dictionaries
        """
        if n_recommendations is None:
            n_recommendations = self.config.default_n_recommendations
        if min_rating is None:
            min_rating = self.config.default_min_rating
        
        genre_col = f"genre_{genre.lower().replace(' ', '_')}"
        
        if genre_col not in self.movies_df.columns:
            available_genres = self.get_available_genres()
            raise ValueError(f"Genre '{genre}' not found. Available: {available_genres}")
        
        # Filter movies by genre
        genre_movies = self.movies_df[self.movies_df[genre_col] == 1].copy()
        
        # Apply rating filter
        rating_col = self._get_rating_column()
        if rating_col:
            genre_movies = genre_movies[genre_movies[rating_col] >= min_rating]
            genre_movies = genre_movies.sort_values(rating_col, ascending=False)
        
        # Prepare results
        recommendations = []
        for _, movie in genre_movies.head(n_recommendations).iterrows():
            recommendations.append({
                'id': movie.get('id'),
                'title': movie.get('title'),
                'rating': self._get_movie_rating(movie),
                'votes': movie.get('vote_count', 0),
                'genres': self._extract_genres(movie),
                'year': self._extract_year(movie)
            })
        
        return recommendations
    
    def get_similar_movies_batch(self, movie_ids: List[int], n_each: int = 5) -> Dict[int, List[Dict]]:
        """
        Get recommendations for multiple movies efficiently
        
        Args:
            movie_ids: List of movie IDs
            n_each: Number of recommendations per movie
            
        Returns:
            Dictionary mapping movie ID to recommendations
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        results = {}
        
        for movie_id in movie_ids:
            try:
                if movie_id in self.movie_mappings['id_to_idx']:
                    movie_idx = self.movie_mappings['id_to_idx'][movie_id]
                    movie_title = self.movies_df.iloc[movie_idx]['title']
                    results[movie_id] = self.get_movie_recommendations(movie_title, n_each)
                else:
                    results[movie_id] = []
            except Exception as e:
                self.logger.warning(f"Failed to get recommendations for movie ID {movie_id}: {e}")
                results[movie_id] = []
        
        return results
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres"""
        genre_cols = [col for col in self.movies_df.columns if col.startswith('genre_')]
        return [col.replace('genre_', '').replace('_', ' ').title() for col in genre_cols]
    
    def get_available_actors(self) -> List[str]:
        """Get list of available actors"""
        actor_cols = [col for col in self.movies_df.columns if col.startswith('actor_')]
        return [col.replace('actor_', '').replace('_', ' ').title() for col in actor_cols]
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self.movies_df is None:
            return {}
        
        return {
            'total_movies': len(self.movies_df),
            'total_features': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
            'genres': len(self.get_available_genres()),
            'actors': len(self.get_available_actors()),
            'rating_range': {
                'min': float(self.movies_df[self._get_rating_column()].min()) if self._get_rating_column() else None,
                'max': float(self.movies_df[self._get_rating_column()].max()) if self._get_rating_column() else None
            },
            'year_range': {
                'min': int(self._extract_years().min()) if len(self._extract_years()) > 0 else None,
                'max': int(self._extract_years().max()) if len(self._extract_years()) > 0 else None
            }
        }
    
    def _get_rating_column(self) -> Optional[str]:
        """Find the rating column in the dataset"""
        rating_cols = ['weighted_rating', 'vote_average', 'rating']
        for col in rating_cols:
            if col in self.movies_df.columns:
                return col
        return None
    
    def _get_movie_rating(self, movie_data) -> float:
        """Extract rating from movie data"""
        rating_col = self._get_rating_column()
        return movie_data.get(rating_col, 0.0) if rating_col else 0.0
    
    def _extract_genres(self, movie_data) -> str:
        """Extract genres from movie data"""
        genre_cols = [col for col in movie_data.index if col.startswith('genre_')]
        active_genres = [col.replace('genre_', '').replace('_', ' ').title() 
                        for col in genre_cols if movie_data.get(col, 0) == 1]
        return ', '.join(active_genres[:5]) if active_genres else ""
    
    def _extract_year(self, movie_data) -> Optional[int]:
        """Extract year from movie data"""
        if 'release_date' in movie_data and pd.notna(movie_data['release_date']):
            try:
                return pd.to_datetime(movie_data['release_date']).year
            except:
                pass
        
        # Try other year columns
        for col in ['year', 'release_year']:
            if col in movie_data and pd.notna(movie_data[col]):
                return int(movie_data[col])
        
        return None
    
    def _extract_years(self) -> pd.Series:
        """Extract all years from the dataset"""
        if 'year' in self.movies_df.columns:
            return self.movies_df['year'].dropna()
        elif 'release_date' in self.movies_df.columns:
            return pd.to_datetime(self.movies_df['release_date'], errors='coerce').dt.year.dropna()
        else:
            return pd.Series([])


def main():
    """Example usage of the enhanced recommender system"""
    
    # Initialize with configuration
    config = RecommendationConfig(
        similarity_method="cosine",
        default_n_recommendations=5,
        default_min_rating=6.0,
        fuzzy_search_threshold=70,
        cache_enabled=True
    )
    
    # Create recommender
    try:
        recommender = ContentBasedRecommender(config)
        recommender.load_data()
        
        print("Enhanced Content-Based Movie Recommender System")
        print("=" * 50)
        
        # Show dataset stats
        stats = recommender.get_dataset_stats()
        print(f"\nDataset Statistics:")
        print(f"Total movies: {stats['total_movies']:,}")
        print(f"Features: {stats['total_features']}")
        print(f"Genres: {stats['genres']}")
        print(f"Actors: {stats['actors']}")
        
        # Example recommendations
        print(f"\nExample: Movies similar to 'Avatar'")
        try:
            recs = recommender.get_movie_recommendations("Avatar", n_recommendations=5)
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['title']} ({rec['year']}) - {rec['rating']:.1f} ⭐")
                print(f"   Genres: {rec['genres']}")
                print(f"   Similarity: {rec['similarity']:.3f}")
                print()
        except Exception as e:
            print(f"Error: {e}")
        
        # Genre recommendations
        print(f"\nTop Action Movies:")
        try:
            action_recs = recommender.get_genre_recommendations("Action", n_recommendations=5)
            for i, rec in enumerate(action_recs, 1):
                print(f"{i}. {rec['title']} ({rec['year']}) - {rec['rating']:.1f} ⭐")
        except Exception as e:
            print(f"Error: {e}")
            
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")


if __name__ == "__main__":
    main()