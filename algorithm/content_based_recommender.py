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
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RecommendationConfig:
    """Configuration class for the recommender system"""
    dataset_path: Optional[str] = None
    cache_dir: str = "cache"
    similarity_method: str = "cosine"  # cosine, euclidean
    default_n_recommendations: int = 8
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
            # Handle raw CSV format with categorical data (like dataset/movies.csv)
            if 'genre' in self.movies_df.columns and self.movies_df['genre'].dtype == 'object':
                self._prepare_features_from_raw_csv()
            else:
                # Handle pre-processed numerical features
                self._prepare_features_from_processed_csv()
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            raise DatasetError(f"Failed to prepare features: {e}")
    
    def _prepare_features_from_raw_csv(self):
        """Prepare features from raw CSV - Multi-feature approach: title, genre, popularity, rating"""
        # Create separate feature vectors for all components
        title_features = []
        genre_features = []
        
        for i in range(len(self.movies_df)):
            # Title keywords repeated 5 times for strong weight
            title_keywords = self._extract_title_keywords(self.movies_df.iloc[i]['title'])
            title_text = ' '.join([title_keywords] * 5)
            title_features.append(title_text)
            
            # Genre repeated 3 times for moderate weight
            genre_text = ' '.join([str(self.movies_df.iloc[i]['genre']) if 'genre' in self.movies_df.columns else ''] * 3)
            genre_features.append(genre_text)
        
        # Create TF-IDF for title (most important)
        title_tfidf = TfidfVectorizer(
            max_features=2000,
            stop_words='english', 
            min_df=1,
            ngram_range=(1, 2)
        )
        title_matrix = title_tfidf.fit_transform(title_features)
        
        # Create TF-IDF for genre 
        genre_tfidf = TfidfVectorizer(
            max_features=500,
            min_df=1,
            ngram_range=(1, 2)
        )
        genre_matrix = genre_tfidf.fit_transform(genre_features)
        
        # Prepare numerical features (popularity and rating)
        numerical_features = []
        for col in ['popularity', 'vote_average']:
            if col in self.movies_df.columns:
                numerical_features.append(col)
        
        # Multi-feature weighting: title, genre, popularity, rating
        title_weight = 0.60    # Still dominant for franchise matching
        genre_weight = 0.25    # Important for content 
          
        numerical_weight = 0.15 # Popularity + rating for quality boost
        
        feature_matrices = []
        
        # Add title features (scaled)
        title_scaled = title_matrix.toarray() * title_weight
        feature_matrices.append(title_scaled)
        
        # Add genre features (scaled)
        genre_scaled = genre_matrix.toarray() * genre_weight
        feature_matrices.append(genre_scaled)
        
        # Add numerical features if available
        if numerical_features:
            numerical_data = self.movies_df[numerical_features].fillna(0)
            numerical_scaled = self.scaler.fit_transform(numerical_data) * numerical_weight
            feature_matrices.append(numerical_scaled)
            
            self.logger.info(f"Added numerical features: {numerical_features}")
        else:
            self.logger.warning("No numerical features (popularity, vote_average) found")
        
        # Combine all feature matrices
        self.feature_matrix = np.hstack(feature_matrices)
        
        self.logger.info(f"Multi-feature matrix shape: {self.feature_matrix.shape}")
        self.logger.info(f"Weights - Title: {title_weight}, Genre: {genre_weight}, Numerical: {numerical_weight}")
        
        if len(feature_matrices) >= 3:
            self.logger.info(f"Title: {title_scaled.shape[1]}, Genre: {genre_scaled.shape[1]}, Numerical: {numerical_scaled.shape[1]}")
        else:
            self.logger.info(f"Title: {title_scaled.shape[1]}, Genre: {genre_scaled.shape[1]}")
    
    def _extract_title_keywords(self, title):
        """Extract key words from movie title"""
        import re
        
        # Remove common subtitle patterns
        title = re.sub(r':\s*Part\s+\d+', '', title, flags=re.IGNORECASE)
        title = re.sub(r':\s*Chapter\s+\d+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+\d+$', '', title)  # Remove trailing numbers
        title = re.sub(r'\s+\(\d{4}\)$', '', title)  # Remove year in parentheses
        
        # Remove common words but keep franchise names
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower() for word in title.split() if word.lower() not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def _prepare_features_from_processed_csv(self):
        """Prepare features from pre-processed CSV with numerical data"""
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['id', 'title', 'overview', 'tagline', 'release_date', 'combined_features']
        feature_cols = [col for col in self.movies_df.columns if col not in exclude_cols]
        
        self.logger.info(f"Using {len(feature_cols)} features for similarity computation")
        
        # Fill missing values and scale features
        feature_data = self.movies_df[feature_cols].fillna(0)
        self.feature_matrix = self.scaler.fit_transform(feature_data)
        
        self.logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
    
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
        Enhanced fuzzy search using multiple matching strategies
        
        Args:
            title: Movie title to search for
            threshold: Minimum similarity threshold (0-100)
            
        Returns:
            Tuple of (matched_title, match_score, movie_index)
            
        Raises:
            MovieNotFoundError: If no suitable match is found
        """
        if threshold is None:
            threshold = self.config.fuzzy_search_threshold
        
        title_lower = title.lower().strip()
        
        if len(title_lower) < 2:
            raise MovieNotFoundError(f"Search query too short: '{title}'")
        
        matches = []
        
        # Strategy 1: Exact substring match (highest priority)
        for i, movie_title in enumerate(self.movie_mappings['titles']):
            movie_lower = movie_title.lower()
            
            if title_lower == movie_lower:
                # Perfect exact match
                return movie_title, 100, i
            
            if title_lower in movie_lower:
                # Substring match - calculate coverage score
                coverage = len(title_lower) / len(movie_lower) * 100
                similarity_score = min(95 + coverage, 100)
                matches.append((movie_title, similarity_score, i))
                continue
        
        # Strategy 2: Word-based matching for partial titles
        if not matches:  # Only if no exact substring matches
            search_words = title_lower.split()
            
            for i, movie_title in enumerate(self.movie_mappings['titles']):
                movie_lower = movie_title.lower()
                movie_words = movie_lower.split()
                
                word_matches = 0
                for search_word in search_words:
                    for movie_word in movie_words:
                        # Check both partial and full word matches
                        if (search_word in movie_word or movie_word in search_word) and len(search_word) >= 3:
                            word_matches += 1
                            break
                
                if word_matches > 0:
                    # Score based on percentage of words matched
                    word_score = (word_matches / len(search_words)) * 90
                    matches.append((movie_title, word_score, i))
        
        # Strategy 3: Fuzzy string matching (for typos)
        if not matches:  # Only if no word matches found
            for i, movie_title in enumerate(self.movie_mappings['titles']):
                similarity = difflib.SequenceMatcher(None, title_lower, movie_title.lower()).ratio()
                if similarity > 0.6:  # Higher threshold for fuzzy matches
                    fuzzy_score = similarity * 85  # Max 85 for fuzzy matches
                    matches.append((movie_title, fuzzy_score, i))
        
        # Strategy 4: Franchise-aware matching
        if not matches and len(search_words) > 1:
            # Try matching against franchise patterns
            franchise_patterns = {
                'austin powers': ['austin powers'],
                'spider man': ['spider-man', 'spiderman'],
                'iron man': ['iron man'],
                'fast furious': ['fast', 'furious'],
                'lord rings': ['lord', 'rings'],
                'toy story': ['toy story'],
                'matrix': ['matrix'],
                'bourne': ['bourne'],
                'godfather': ['godfather']
            }
            
            search_key = title_lower.replace('-', ' ').strip()
            
            for pattern, keywords in franchise_patterns.items():
                if any(keyword in search_key for keyword in keywords):
                    for i, movie_title in enumerate(self.movie_mappings['titles']):
                        movie_lower = movie_title.lower()
                        if any(keyword in movie_lower for keyword in keywords):
                            franchise_score = 80
                            matches.append((movie_title, franchise_score, i))
        
        if not matches:
            # Last resort: very lenient substring matching
            for i, movie_title in enumerate(self.movie_mappings['titles']):
                movie_lower = movie_title.lower()
                if any(word in movie_lower for word in search_words if len(word) >= 4):
                    matches.append((movie_title, 60, i))
        
        # Sort by similarity score and return best match
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            matched_title, score, idx = matches[0]
            
            if score >= threshold:
                self.logger.info(f"Enhanced fuzzy match: '{title}' -> '{matched_title}' (score: {score:.1f}%)")
                return matched_title, int(score), idx
        
        # If still no good matches, provide helpful suggestions
        suggestions = []
        for movie_title in self.movie_mappings['titles']:
            if any(word in movie_title.lower() for word in search_words if len(word) >= 3):
                suggestions.append(movie_title)
        
        if suggestions:
            raise MovieNotFoundError(
                f"No good match found for '{title}' (threshold: {threshold}%). "
                f"Did you mean: {', '.join(suggestions[:3])}?"
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
        
        # Apply filters and collect recommendations with similarity threshold
        recommendations = []
        min_similarity_threshold = 0.50  # Only recommend movies with >50% similarity
        
        for idx, score in sim_scores:
            movie_data = self.movies_df.iloc[idx]
            
            # Skip movies with very low similarity (quality over quantity)
            if score < min_similarity_threshold:
                continue
            
            # Get movie rating and votes
            rating = self._get_movie_rating(movie_data)
            votes = movie_data.get('vote_count', 0)
            
            # Apply basic filters
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
            
            # Stop if we have enough high-quality recommendations
            if len(recommendations) >= n_recommendations:
                break
        
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
        # Handle raw CSV format with 'genre' column containing comma-separated values
        if 'genre' in movie_data.index and pd.notna(movie_data['genre']):
            return str(movie_data['genre'])
        
        # Handle pre-processed format with individual genre columns
        genre_cols = [col for col in movie_data.index if col.startswith('genre_')]
        if genre_cols:
            active_genres = [col.replace('genre_', '').replace('_', ' ').title() 
                            for col in genre_cols if movie_data.get(col, 0) == 1]
            return ', '.join(active_genres[:5]) if active_genres else ""
        
        return ""
    
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