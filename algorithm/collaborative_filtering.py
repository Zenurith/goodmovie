import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import json
from functools import lru_cache


class CollaborativeFilter:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.user_ratings = {}
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self._users_data_cache = None
        self._similarity_cache = {}
        self._matrix_cache = {}
        self._profile_cache = {}
        self._last_cache_clear = 0
        
    def update_user_ratings(self, user_ratings: Dict[str, int]) -> None:
        """Update the user ratings dictionary."""
        self.user_ratings = user_ratings.copy()
        # Clear caches when user ratings change
        self._similarity_cache.clear()
        self._profile_cache.clear()
        
    def _clear_old_caches(self):
        """Optimized cache management for better performance"""
        import time
        current_time = time.time()
        if current_time - self._last_cache_clear > 600:  # Clear every 10 minutes (reduced for faster turnover)
            # More aggressive cache clearing for performance
            if len(self._similarity_cache) > 5:  # Reduced cache size
                self._similarity_cache.clear()
            if len(self._matrix_cache) > 3:  # Reduced cache size
                self._matrix_cache.clear()
            if len(self._profile_cache) > 10:  # Reduced cache size
                # Keep only the 5 most recent entries
                recent_items = list(self._profile_cache.items())[-5:]
                self._profile_cache.clear()
                self._profile_cache.update(recent_items)
            self._last_cache_clear = current_time
    
    def _load_users_data(self, users_file_path: str = None) -> Dict[str, Dict[str, int]]:
        """Load user rating data from users_data.json file with caching."""
        # Use cache if available
        if self._users_data_cache is not None:
            return self._users_data_cache
            
        try:
            if users_file_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                users_file_path = os.path.join(os.path.dirname(current_dir), 'users_data.json')
            
            if not os.path.exists(users_file_path):
                logging.warning(f"Users data file not found: {users_file_path}")
                return {}
            
            with open(users_file_path, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            
            # Extract ratings data from JSON structure
            real_users_data = {}
            
            if 'users' in users_data:
                for user_id, user_info in users_data['users'].items():
                    if 'ratings' in user_info and user_info['ratings']:
                        # Convert all keys to strings and ensure ratings are integers
                        user_ratings = {}
                        for movie_id, rating in user_info['ratings'].items():
                            user_ratings[str(movie_id)] = int(rating)
                        
                        # Only include users with at least one rating
                        if user_ratings:
                            real_users_data[str(user_id)] = user_ratings
            
            logging.info(f"Loaded ratings data for {len(real_users_data)} users from users_data.json")
            
            # Cache the result
            self._users_data_cache = real_users_data
            return real_users_data
            
        except Exception as e:
            logging.error(f"Error loading users ratings data: {e}")
            return {}
    
    def _extract_ratings_from_users_data(self, users_data: Dict) -> Dict[str, Dict[str, int]]:
        """
        Extract ratings from potentially nested users data structure.
        Handles both flat and nested formats.
        """
        clean_data = {}
        
        # Check if data has nested structure with 'users' key
        if 'users' in users_data and isinstance(users_data['users'], dict):
            # Nested structure: {'users': {user_id: {'ratings': {...}, ...}}}
            for user_id, user_info in users_data['users'].items():
                if isinstance(user_info, dict) and 'ratings' in user_info:
                    # Extract only the ratings part and ensure all keys are strings
                    ratings = user_info['ratings']
                    if isinstance(ratings, dict):
                        clean_data[str(user_id)] = {str(k): int(v) for k, v in ratings.items() 
                                                  if isinstance(v, (int, float)) and 1 <= int(v) <= 10}
        else:
            # Flat structure: {user_id: {movie_id: rating, ...}} or {user_id: {'ratings': {...}}}
            for user_id, user_info in users_data.items():
                # Skip metadata entries
                if user_id == 'metadata' or not isinstance(user_info, dict):
                    continue
                
                if 'ratings' in user_info:
                    # User info has nested ratings
                    ratings = user_info['ratings']
                    if isinstance(ratings, dict):
                        clean_data[str(user_id)] = {str(k): int(v) for k, v in ratings.items() 
                                                  if isinstance(v, (int, float)) and 1 <= int(v) <= 10}
                else:
                    # Assume user_info is directly the ratings dict
                    if all(isinstance(v, (int, float, str)) for v in user_info.values()):
                        try:
                            clean_data[str(user_id)] = {str(k): int(v) for k, v in user_info.items() 
                                                      if isinstance(v, (int, float)) and 1 <= int(v) <= 10}
                        except (ValueError, TypeError):
                            continue
        
        return clean_data
    
    def _build_user_item_matrix(self, all_users_data: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Build user-item rating matrix with performance optimizations and caching."""
        # Check cache first
        data_hash = hash(str(sorted(all_users_data.items())))
        if data_hash in self._matrix_cache:
            return self._matrix_cache[data_hash]
        
        # Clean and extract ratings data, handling nested structure
        clean_users_data = self._extract_ratings_from_users_data(all_users_data)
        
        if not clean_users_data:
            return pd.DataFrame()
        
        # Get all unique movie IDs that exist in our dataset
        all_movie_ids = set()
        movie_id_set = set(self.df['id'].astype(str))  # Pre-convert for faster lookup
        
        for user_ratings in clean_users_data.values():
            if isinstance(user_ratings, dict):
                # Only add movie IDs that exist in our dataset
                valid_movie_ids = {mid for mid in user_ratings.keys() if mid in movie_id_set}
                all_movie_ids.update(valid_movie_ids)
        
        all_movie_ids = sorted(list(all_movie_ids))
        
        if not all_movie_ids:
            return pd.DataFrame()
        
        # Build matrix more efficiently using numpy
        num_users = len(clean_users_data)
        num_movies = len(all_movie_ids)
        matrix_data = np.zeros((num_users, num_movies))
        user_ids = []
        
        for i, (user_id, ratings) in enumerate(clean_users_data.items()):
            if isinstance(ratings, dict) and ratings:
                user_ids.append(user_id)
                for j, movie_id in enumerate(all_movie_ids):
                    matrix_data[i, j] = ratings.get(movie_id, 0)
        
        if not user_ids:
            return pd.DataFrame()
            
        result = pd.DataFrame(matrix_data, index=user_ids, columns=all_movie_ids)
        
        # Cache the result (limit cache size)
        if len(self._matrix_cache) < 10:
            self._matrix_cache[data_hash] = result
        
        return result
    
    @lru_cache(maxsize=16)  # Further reduced cache size for speed
    def _calculate_user_similarity(self, user_item_matrix_hash: int, target_user: str) -> pd.Series:
        """
        Calculate similarity between target user and all other users with caching and optimization.
        """
        if target_user not in self.user_item_matrix.index:
            return pd.Series(dtype=float)
        
        target_ratings = self.user_item_matrix.loc[target_user].values.reshape(1, -1)
        
        # Only calculate similarity with users who have sufficient overlap
        # Pre-filter users with at least 2 movies in common for performance
        target_nonzero = target_ratings[0] != 0
        sufficient_overlap_users = []
        
        for idx, user in enumerate(self.user_item_matrix.index):
            if user != target_user:
                other_nonzero = self.user_item_matrix.iloc[idx].values != 0
                overlap = np.sum(target_nonzero & other_nonzero)
                if overlap >= 2:  # At least 2 movies in common
                    sufficient_overlap_users.append(idx)
        
        if not sufficient_overlap_users:
            return pd.Series(dtype=float)
        
        # Calculate cosine similarity only for users with sufficient overlap
        other_users_matrix = self.user_item_matrix.iloc[sufficient_overlap_users].values
        similarities = cosine_similarity(target_ratings, other_users_matrix)[0]
        
        # Create full similarity series with zeros for users without sufficient overlap
        all_similarities = np.zeros(len(self.user_item_matrix))
        all_similarities[sufficient_overlap_users] = similarities
        
        return pd.Series(all_similarities, index=self.user_item_matrix.index)
    
    @lru_cache(maxsize=1)
    def _build_item_similarity_matrix_cached(self, df_hash: int) -> np.ndarray:
        """
        Build optimized item-item similarity matrix with caching and performance improvements.
        """
        try:
            # Use simplified feature extraction for speed
            movies_df = self.df.copy()
            
            # Pre-compute feature vectors more efficiently
            feature_data = []
            
            # Vectorized operations for better performance
            genres_list = movies_df['genre'].fillna('').astype(str)
            ratings = movies_df['vote_average'].fillna(0).astype(float)
            popularity = movies_df['popularity'].fillna(0).astype(float)
            languages = movies_df['original_language'].fillna('').astype(str)
            
            # Batch process features
            for i, (genre, rating, pop, lang) in enumerate(zip(genres_list, ratings, popularity, languages)):
                # Simplified feature text for speed
                genres_clean = genre.replace(',', ' ')
                rating_bucket = f"r{int(rating)}" if rating > 0 else ""
                pop_bucket = "pop" if pop > 50 else "mod" if pop > 20 else "low"
                
                feature_text = f"{genres_clean} {rating_bucket} {pop_bucket} {lang}"
                feature_data.append(feature_text.strip())
            
            # Optimized TF-IDF with reduced features for speed
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=500,  # Reduced for speed
                ngram_range=(1, 1),  # Only unigrams for speed
                min_df=3,  # Higher threshold for speed
                max_df=0.7,
                token_pattern=r'\b\w+\b'  # Simple tokenization
            )
            
            tfidf_matrix = vectorizer.fit_transform(feature_data)
            
            # Use sparse matrix operations for memory efficiency
            # Only compute similarity for a subset if dataset is large
            if len(self.df) > 1000:
                # For large datasets, use approximate similarity
                # Compute in chunks to save memory
                chunk_size = 200
                n_movies = tfidf_matrix.shape[0]
                similarity_matrix = np.zeros((n_movies, n_movies))
                
                for i in range(0, n_movies, chunk_size):
                    end_i = min(i + chunk_size, n_movies)
                    chunk_similarities = cosine_similarity(tfidf_matrix[i:end_i], tfidf_matrix)
                    similarity_matrix[i:end_i] = chunk_similarities
                
                return similarity_matrix
            else:
                # For smaller datasets, compute full similarity
                return cosine_similarity(tfidf_matrix)
            
        except Exception as e:
            logging.error(f"Error building optimized item similarity matrix: {e}")
            return np.eye(len(self.df))
    
    def _build_item_similarity_matrix(self) -> np.ndarray:
        """
        Wrapper for cached item similarity matrix building.
        """
        # Create a hash of the dataframe for caching
        df_hash = hash(str(self.df['id'].tolist() + self.df['genre'].fillna('').tolist()))
        return self._build_item_similarity_matrix_cached(df_hash)
    
    
    def get_recommendations(self, n_recommendations: int = 5, 
                          all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> pd.DataFrame:
        """
        Generate movie recommendations using optimized collaborative filtering with caching.
        """
        try:
            self._clear_old_caches()  # Periodic cache cleanup
            
            # Create cache key
            cache_key = f"{hash(str(self.user_ratings))}_{n_recommendations}"
            if cache_key in self._profile_cache:
                return self._profile_cache[cache_key]
            
            # Validate input
            valid_ratings = {k: v for k, v in self.user_ratings.items() if v > 0 and v <= 10}
            
            if not valid_ratings or len(valid_ratings) < 1:
                result = self._get_fallback_recommendations(n_recommendations)
                self._profile_cache[cache_key] = result
                return result
            
            # Load user data if not provided
            if not all_users_data:
                all_users_data = self._load_users_data()
            
            # If we have multiple users' data, use collaborative filtering
            if all_users_data and len(all_users_data) > 1:  # Reduced threshold - only need 2+ users
                result = self._get_fast_collaborative_recommendations(n_recommendations, all_users_data)
                
                # If collaborative filtering returns empty results, fall back to content-based
                if result.empty:
                    result = self._get_content_based_recommendations(n_recommendations)
            else:
                # Fall back to content-based recommendations for speed
                result = self._get_content_based_recommendations(n_recommendations)
                
            # Final fallback if all methods fail
            if result.empty:
                result = self._get_fallback_recommendations(n_recommendations)
            
            # Cache result (limit cache size)
            if len(self._profile_cache) < 50:
                self._profile_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logging.error(f"Error in get_recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_fast_collaborative_recommendations(self, n_recommendations: int, 
                                               all_users_data: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Get recommendations using hybrid collaborative filtering (user-based + item-based).
        """
        try:
            # Add current user to the data
            current_user_id = "current_user"
            all_users_with_current = all_users_data.copy()
            all_users_with_current[current_user_id] = self.user_ratings
            
            # Build user-item matrix (cached)
            if self.user_item_matrix is None:
                self.user_item_matrix = self._build_user_item_matrix(all_users_with_current)
            
            if self.user_item_matrix.empty or current_user_id not in self.user_item_matrix.index:
                return self._get_content_based_recommendations(n_recommendations)
            
            # Get recommendations from both CF approaches (optimized allocation)
            # Prioritize user-based CF as it's typically more accurate
            user_cf_count = max(2, int(n_recommendations * 0.6))
            item_cf_count = max(2, int(n_recommendations * 0.4))
            
            user_cf_recs = self._get_user_based_recommendations(user_cf_count, all_users_data)
            item_cf_recs = self._get_item_based_recommendations(item_cf_count)
            
            # Combine and deduplicate collaborative filtering results
            combined_cf_recs = self._combine_cf_recommendations(user_cf_recs, item_cf_recs, n_recommendations)
            
            # If we don't have enough recommendations, supplement with content-based
            if len(combined_cf_recs) < n_recommendations:
                remaining_needed = n_recommendations - len(combined_cf_recs)
                content_recs = self._get_content_based_recommendations(remaining_needed)
                
                if not combined_cf_recs.empty and not content_recs.empty:
                    # Avoid duplicates when combining
                    cf_movie_ids = set(combined_cf_recs['id'].astype(str))
                    content_recs = content_recs[~content_recs['id'].astype(str).isin(cf_movie_ids)]
                    
                    # Combine results
                    import pandas as pd
                    final_recs = pd.concat([combined_cf_recs, content_recs], ignore_index=True)
                    return final_recs.head(n_recommendations)
                elif not combined_cf_recs.empty:
                    return combined_cf_recs
                else:
                    return content_recs
            
            return combined_cf_recs if not combined_cf_recs.empty else self._get_content_based_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in hybrid collaborative recommendations: {e}")
            return self._get_content_based_recommendations(n_recommendations)
    
    def _combine_cf_recommendations(self, user_recs: pd.DataFrame, item_recs: pd.DataFrame, n_recommendations: int) -> pd.DataFrame:
        """
        Combine user-based and item-based collaborative filtering recommendations with weighted scoring.
        """
        try:
            movie_scores = {}
            movie_data = {}
            
            # Process user-based recommendations (higher weight)
            if not user_recs.empty:
                for _, movie in user_recs.iterrows():
                    movie_id = str(movie['id'])
                    movie_data[movie_id] = movie.to_dict()
                    
                    # User-based CF gets weight of 0.6
                    base_score = movie.get('vote_average', 5.0) / 10.0
                    cf_score = movie.get('cf_score', 0) if 'cf_score' in movie else 0
                    combined_score = (base_score + cf_score) * 0.6
                    
                    movie_scores[movie_id] = movie_scores.get(movie_id, 0) + combined_score
            
            # Process item-based recommendations (lower weight but still significant)
            if not item_recs.empty:
                for _, movie in item_recs.iterrows():
                    movie_id = str(movie['id'])
                    
                    # Store movie data if not already stored
                    if movie_id not in movie_data:
                        movie_data[movie_id] = movie.to_dict()
                    
                    # Item-based CF gets weight of 0.4
                    base_score = movie.get('vote_average', 5.0) / 10.0
                    item_score = movie.get('item_score', 0) if 'item_score' in movie else 0
                    combined_score = (base_score + item_score * 5) * 0.4  # Scale up item scores
                    
                    movie_scores[movie_id] = movie_scores.get(movie_id, 0) + combined_score
            
            # Sort by combined score and create result
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommended_movies = []
            for movie_id, score in sorted_movies[:n_recommendations]:
                if movie_id in movie_data:
                    movie_dict = movie_data[movie_id].copy()
                    movie_dict['hybrid_cf_score'] = score
                    recommended_movies.append(movie_dict)
            
            return pd.DataFrame(recommended_movies)
            
        except Exception as e:
            logging.error(f"Error combining CF recommendations: {e}")
            return user_recs if not user_recs.empty else item_recs
    
    def _get_user_based_recommendations(self, n_recommendations: int, 
                                       all_users_data: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Get recommendations using optimized user-based collaborative filtering.
        """
        try:
            current_user_id = "current_user"
            
            # Calculate user similarities (cached)
            matrix_hash = hash(str(self.user_item_matrix.values.tobytes()))
            user_similarities = self._calculate_user_similarity(matrix_hash, current_user_id)
            
            # Find most similar users (optimized for speed)
            similar_users = user_similarities.drop(current_user_id, errors='ignore').nlargest(2)  # Only top 2 for speed
            
            if similar_users.empty or similar_users.max() < 0.01:  # Lowered threshold back
                return pd.DataFrame()
            
            # Get movie recommendations from similar users
            movie_scores = {}
            rated_movies = set(self.user_ratings.keys())
            user_avg = np.mean(list(self.user_ratings.values()))
            
            # Pre-extract clean data once
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            
            for similar_user, similarity_score in similar_users.items():
                if similarity_score < 0.01 or similar_user not in clean_all_users_data:  # Lowered threshold
                    continue
                    
                user_ratings = clean_all_users_data[similar_user]
                similar_user_avg = np.mean(list(user_ratings.values()))
                
                for movie_id, rating in user_ratings.items():
                    if movie_id not in rated_movies and rating >= 6:  # Lowered rating threshold back
                        # Simplified scoring for speed
                        normalized_rating = rating - similar_user_avg + user_avg
                        score = similarity_score * normalized_rating
                        
                        if movie_id not in movie_scores:
                            movie_scores[movie_id] = 0
                        movie_scores[movie_id] += score
            
            if not movie_scores:
                return pd.DataFrame()
            
            # Get top movies quickly
            top_movie_ids = sorted(movie_scores.keys(), key=lambda x: movie_scores[x], reverse=True)[:n_recommendations]
            
            # Batch query for movie data
            movie_id_series = pd.Series(top_movie_ids)
            result_df = self.df[self.df['id'].astype(str).isin(movie_id_series)]
            
            if not result_df.empty:
                # Add scores
                result_df = result_df.copy()
                result_df['cf_score'] = result_df['id'].astype(str).map(movie_scores)
                # Quick quality filter - more lenient
                result_df = result_df[result_df['vote_average'] >= 6.0]  # Lowered threshold
                result_df = result_df.head(n_recommendations)
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error in user-based recommendations: {e}")
            return pd.DataFrame()
    
    def _get_item_based_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Optimized item-based collaborative filtering with vectorized operations.
        """
        try:
            if not self.user_ratings:
                return pd.DataFrame()
            
            # Build item similarity matrix if not already built (now cached)
            if self.item_similarity_matrix is None:
                self.item_similarity_matrix = self._build_item_similarity_matrix()
            
            # Vectorized approach for better performance
            df_ids = self.df['id'].astype(str)
            all_movie_ids = df_ids.tolist()
            rated_movie_ids = set(self.user_ratings.keys())
            
            # Create boolean mask for unrated movies
            unrated_mask = ~df_ids.isin(rated_movie_ids)
            unrated_indices = np.where(unrated_mask)[0]
            
            if len(unrated_indices) == 0:
                return pd.DataFrame()
            
            # Get indices and ratings for rated movies
            rated_data = []
            user_avg = np.mean(list(self.user_ratings.values()))
            
            for rated_movie_id, rating in self.user_ratings.items():
                if rating > 0:
                    try:
                        rated_idx = all_movie_ids.index(rated_movie_id)
                        if rated_idx < len(self.item_similarity_matrix):
                            rated_data.append((rated_idx, rating - user_avg))
                    except ValueError:
                        continue
            
            if not rated_data:
                return pd.DataFrame()
            
            # Vectorized similarity computation
            rated_indices = np.array([idx for idx, _ in rated_data])
            rated_scores = np.array([score for _, score in rated_data])
            
            # Compute scores for all unrated movies at once
            movie_scores = []
            similarity_threshold = 0.05  # Lower threshold for more candidates
            
            # Process in batches for memory efficiency
            batch_size = 100
            for start_idx in range(0, len(unrated_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(unrated_indices))
                batch_indices = unrated_indices[start_idx:end_idx]
                
                for unrated_idx in batch_indices:
                    if unrated_idx >= len(self.item_similarity_matrix):
                        continue
                        
                    # Get similarities to all rated movies
                    similarities = self.item_similarity_matrix[unrated_idx, rated_indices]
                    
                    # Filter by threshold and compute weighted score
                    valid_mask = similarities > similarity_threshold
                    if np.any(valid_mask):
                        valid_sims = similarities[valid_mask]
                        valid_scores = rated_scores[valid_mask]
                        
                        weighted_score = np.sum(valid_sims * valid_scores)
                        weight_sum = np.sum(valid_sims)
                        
                        if weight_sum > 0:
                            normalized_score = weighted_score / weight_sum
                            if normalized_score > 0:
                                movie_scores.append((all_movie_ids[unrated_idx], normalized_score))
            
            # Sort and get top candidates
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = movie_scores[:n_recommendations * 3]  # Get more candidates
            
            if not top_candidates:
                return pd.DataFrame()
            
            # Batch query for movie data
            top_movie_ids = [mid for mid, _ in top_candidates]
            score_dict = {mid: score for mid, score in top_candidates}
            
            # Filter dataframe once
            result_df = self.df[
                (self.df['id'].astype(str).isin(top_movie_ids)) &
                (self.df['vote_average'] >= 6.0)  # Quality filter
            ].copy()
            
            if not result_df.empty:
                # Add scores
                result_df['item_score'] = result_df['id'].astype(str).map(score_dict)
                # Sort by score and limit results
                result_df = result_df.nlargest(n_recommendations, 'item_score')
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error in optimized item-based recommendations: {e}")
            return pd.DataFrame()
    
    
    def _get_content_based_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Get recommendations using fast content-based filtering.
        """
        try:
            if not self.user_ratings:
                return self._get_fallback_recommendations(n_recommendations)
                
            # Analyze user preferences (simplified)
            user_avg_rating = np.mean(list(self.user_ratings.values()))
            high_rated_movies = {k: v for k, v in self.user_ratings.items() if v >= 7}  # Fixed threshold
            
            if not high_rated_movies:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Extract preferred genres only (skip language for speed)
            genre_scores = {}
            
            for movie_id, rating in high_rated_movies.items():
                # Handle both string and numeric IDs
                try:
                    movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                    if movie_row.empty:
                        # Try numeric comparison if string comparison failed
                        movie_row = self.df[self.df['id'] == int(movie_id)]
                except (ValueError, TypeError):
                    movie_row = pd.DataFrame()
                    
                if not movie_row.empty:
                    genres = str(movie_row.iloc[0]['genre']).split(',')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            genre_scores[genre] = genre_scores.get(genre, 0) + rating
            
            if not genre_scores:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Get top 2 genres only for speed
            top_genres = sorted(genre_scores.keys(), key=lambda g: genre_scores[g], reverse=True)[:2]
            
            # Filter movies by top genres and high ratings quickly
            rated_movie_ids = set(str(k) for k in self.user_ratings.keys())
            
            # Pre-filter for quality and unrated movies
            quality_movies = self.df[
                (~self.df['id'].astype(str).isin(rated_movie_ids)) &
                (self.df['vote_average'] >= 6.5) &  # Pre-filter quality
                (self.df['vote_count'] >= 100)      # Pre-filter popularity
            ].copy()
            
            # Simple genre matching
            recommendations = []
            for _, movie in quality_movies.iterrows():
                movie_genres = str(movie['genre']).split(',')
                movie_genres = [g.strip() for g in movie_genres]
                
                # Check if any movie genre matches top genres
                if any(genre in top_genres for genre in movie_genres):
                    movie_dict = movie.to_dict()
                    # Simple score based on rating and genre match count
                    genre_matches = sum(1 for g in movie_genres if g in top_genres)
                    movie_dict['content_score'] = movie['vote_average'] + genre_matches
                    recommendations.append(movie_dict)
            
            # Sort by score and return top N
            recommendations.sort(key=lambda x: x['content_score'], reverse=True)
            
            if recommendations:
                return pd.DataFrame(recommendations[:n_recommendations])
            else:
                # If no genre matches found, return best unrated movies
                if not quality_movies.empty:
                    best_unrated = quality_movies.nlargest(n_recommendations, 'vote_average')
                    return best_unrated
                else:
                    return self._get_fallback_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in content-based recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """Get high-quality popular movies as fallback."""
        try:
            # Try high-quality movies first
            high_quality = self.df[
                (self.df['vote_average'] >= 7.5) & 
                (self.df['vote_count'] >= 1000)
            ]
            
            if not high_quality.empty:
                return high_quality.nlargest(n_recommendations, ['vote_average', 'popularity'])
            
            # Fall back to good movies
            good_quality = self.df[
                (self.df['vote_average'] >= 6.0) & 
                (self.df['vote_count'] >= 100)
            ]
            
            if not good_quality.empty:
                return good_quality.nlargest(n_recommendations, ['vote_average', 'popularity'])
            
            # Final fallback - just return top rated movies
            return self.df.nlargest(n_recommendations, 'vote_average')
            
        except Exception:
            return self.df.head(n_recommendations)
    
    def predict_rating(self, movie_id: str, all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> float:
        """
        Predict rating for a specific movie using hybrid approach.
        """
        try:
            # If movie already rated, return actual rating
            if movie_id in self.user_ratings:
                return float(self.user_ratings[movie_id])
            
            # Load user data if not provided
            if not all_users_data:
                all_users_data = self._load_users_data()
            
            # Get movie data
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            if movie_row.empty:
                return 5.0  # Default neutral rating
                
            movie = movie_row.iloc[0]
            
            predictions = []
            weights = []
            
            # Try collaborative filtering prediction
            if all_users_data and len(all_users_data) > 1:
                collab_rating = self._predict_collaborative_rating(movie_id, all_users_data)
                if collab_rating > 0:
                    confidence = self._get_prediction_confidence(movie_id, all_users_data)
                    predictions.append(collab_rating)
                    weights.append(confidence * 0.7)  # Higher weight for collaborative
            
            # Content-based prediction
            content_rating = self._predict_content_based_rating(movie)
            predictions.append(content_rating)
            weights.append(0.3)  # Lower weight for content-based
            
            # Weighted average
            if weights:
                final_rating = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
                return max(1.0, min(10.0, final_rating))
            
            return 5.0
            
        except Exception as e:
            logging.error(f"Error predicting rating for movie {movie_id}: {e}")
            return 5.0
    
    def _predict_collaborative_rating(self, movie_id: str, all_users_data: Dict[str, Dict[str, int]]) -> float:
        """
        Predict rating using collaborative filtering.
        """
        try:
            # Build user-item matrix if needed
            if self.user_item_matrix is None:
                current_user_id = "current_user"
                all_users_with_current = all_users_data.copy()
                all_users_with_current[current_user_id] = self.user_ratings
                self.user_item_matrix = self._build_user_item_matrix(all_users_with_current)
            
            if self.user_item_matrix.empty:
                return 0.0
                
            current_user_id = "current_user"
            if current_user_id not in self.user_item_matrix.index:
                return 0.0
            
            # Calculate user similarities
            matrix_hash = hash(str(self.user_item_matrix.values.tobytes()))
            user_similarities = self._calculate_user_similarity(matrix_hash, current_user_id)
            similar_users = user_similarities.drop(current_user_id, errors='ignore').nlargest(3)
            
            if similar_users.empty:
                return 0.0
            
            # Weighted average of similar users' ratings
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            
            for similar_user, similarity_score in similar_users.items():
                if (similarity_score > 0.02 and similar_user in clean_all_users_data and 
                    movie_id in clean_all_users_data[similar_user]):
                    
                    user_rating = clean_all_users_data[similar_user][movie_id]
                    weight = similarity_score
                    weighted_sum += weight * user_rating
                    similarity_sum += weight
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                # Bias adjustment
                user_avg = np.mean(list(self.user_ratings.values())) if self.user_ratings else 5.0
                global_avg = 6.0
                bias_adjusted = predicted_rating + (user_avg - global_avg) * 0.05
                
                return max(1.0, min(10.0, bias_adjusted))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_prediction_confidence(self, movie_id: str, all_users_data: Dict[str, Dict[str, int]]) -> float:
        """
        Calculate confidence score for a prediction (0-1 scale).
        """
        try:
            if self.user_item_matrix is None or self.user_item_matrix.empty:
                return 0.2
                
            current_user_id = "current_user"
            matrix_hash = hash(str(self.user_item_matrix.values.tobytes()))
            user_similarities = self._calculate_user_similarity(matrix_hash, current_user_id)
            similar_users = user_similarities.drop(current_user_id, errors='ignore').nlargest(15)
            
            confidence_factors = []
            
            # Number of similar users who rated this movie
            users_who_rated = 0
            total_similarity = 0
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            
            for user, sim_score in similar_users.items():
                if sim_score > 0.05 and user in clean_all_users_data and movie_id in clean_all_users_data[user]:
                    users_who_rated += 1
                    total_similarity += sim_score
            
            # Confidence from number of raters
            rater_confidence = min(users_who_rated / 8.0, 0.4)
            confidence_factors.append(rater_confidence)
            
            # Average similarity of users who rated it
            avg_similarity = total_similarity / users_who_rated if users_who_rated > 0 else 0
            sim_confidence = min(avg_similarity * 2, 0.3)
            confidence_factors.append(sim_confidence)
            
            # User experience (how many ratings they have)
            user_experience = min(len(self.user_ratings) / 15.0, 0.3)
            confidence_factors.append(user_experience)
            
            return sum(confidence_factors)
            
        except Exception:
            return 0.2
    
    def _predict_content_based_rating(self, movie: pd.Series) -> float:
        """
        Predict rating using content-based filtering.
        """
        try:
            if not self.user_ratings:
                return float(movie.get('vote_average', 5.0))
            
            user_avg_rating = np.mean(list(self.user_ratings.values()))
            
            # Get user's genre preferences
            genre_preferences = {}
            for rated_movie_id, rating in self.user_ratings.items():
                rated_movie_row = self.df[self.df['id'].astype(str) == str(rated_movie_id)]
                if not rated_movie_row.empty:
                    genres = str(rated_movie_row.iloc[0]['genre']).split(',')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            if genre not in genre_preferences:
                                genre_preferences[genre] = []
                            genre_preferences[genre].append(rating)
            
            # Average rating for each genre
            genre_avg_ratings = {g: np.mean(ratings) for g, ratings in genre_preferences.items()}
            
            # Calculate prediction based on movie's genres
            movie_genres = str(movie['genre']).split(',')
            movie_genres = [g.strip() for g in movie_genres if g.strip()]
            
            if not movie_genres or not genre_avg_ratings:
                # No genre overlap, use movie's quality adjusted to user's preference
                movie_rating = float(movie.get('vote_average', 5.0))
                return user_avg_rating * 0.4 + movie_rating * 0.6
            
            # Weighted average of genre preferences
            genre_prediction = 0.0
            matched_genres = 0
            
            for genre in movie_genres:
                if genre in genre_avg_ratings:
                    genre_prediction += genre_avg_ratings[genre]
                    matched_genres += 1
            
            if matched_genres > 0:
                genre_prediction /= matched_genres
                # Combine with movie's inherent quality
                movie_quality = float(movie.get('vote_average', 5.0))
                predicted_rating = genre_prediction * 0.6 + movie_quality * 0.4
                
                return max(1.0, min(10.0, predicted_rating))
            else:
                # No genre match
                movie_rating = float(movie.get('vote_average', 5.0))
                return user_avg_rating * 0.3 + movie_rating * 0.7
            
        except Exception:
            return 5.0

    def get_user_stats(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about user's rating patterns.
        """
        rated_movies = [movie_id for movie_id, rating in self.user_ratings.items() if rating > 0]
        
        if not rated_movies:
            return {
                'total_rated': 0,
                'avg_rating': 0.0,
                'favorite_genres': [],
                'rating_distribution': {},
                'diversity_score': 0.0
            }
        
        ratings = [self.user_ratings[movie_id] for movie_id in rated_movies]
        avg_rating = np.mean(ratings)
        
        # Get favorite genres with weights
        genre_counts = {}
        genre_ratings = {}
        languages = set()
        
        for movie_id in rated_movies:
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            if not movie_row.empty:
                movie = movie_row.iloc[0]
                
                # Process genres
                genres = str(movie['genre']).split(',')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        if genre not in genre_ratings:
                            genre_ratings[genre] = []
                        genre_ratings[genre].append(self.user_ratings[movie_id])
                
                # Process languages for diversity
                lang = str(movie.get('original_language', ''))
                if lang:
                    languages.add(lang)
        
        favorite_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Rating distribution
        rating_distribution = {}
        for rating in ratings:
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        # Calculate diversity score
        diversity_score = len(languages) / max(len(rated_movies), 1) + len(genre_counts) / max(len(rated_movies), 1)
        diversity_score = min(diversity_score, 1.0)
        
        return {
            'total_rated': len(rated_movies),
            'avg_rating': round(avg_rating, 1),
            'favorite_genres': [genre for genre, count in favorite_genres],
            'genre_ratings': {genre: round(np.mean(ratings), 1) for genre, ratings in genre_ratings.items()},
            'rating_distribution': rating_distribution,
            'diversity_score': round(diversity_score, 2),
            'languages_explored': len(languages)
        }


def create_recommender(df: pd.DataFrame) -> CollaborativeFilter:
    """Factory function to create a collaborative filter instance."""
    return CollaborativeFilter(df)


def load_users_data(users_file_path: str = None) -> Dict[str, Dict[str, int]]:
    """Load user rating data from users_data.json file."""
    try:
        if users_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            users_file_path = os.path.join(os.path.dirname(current_dir), 'users_data.json')
        
        if not os.path.exists(users_file_path):
            logging.warning(f"Users data file not found at {users_file_path}")
            return {}
        
        with open(users_file_path, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # Extract ratings data from JSON structure
        real_users_data = {}
        
        if 'users' in users_data:
            for user_id, user_info in users_data['users'].items():
                if 'ratings' in user_info and user_info['ratings']:
                    # Convert all keys to strings and ensure ratings are integers
                    user_ratings = {}
                    for movie_id, rating in user_info['ratings'].items():
                        user_ratings[str(movie_id)] = int(rating)
                    
                    # Only include users with at least one rating
                    if user_ratings:
                        real_users_data[str(user_id)] = user_ratings
        
        logging.info(f"Loaded ratings data for {len(real_users_data)} users from users_data.json")
        return real_users_data
        
    except Exception as e:
        logging.error(f"Error loading users ratings data: {e}")
        return {}


def get_recommendations(df: pd.DataFrame, user_ratings: Dict[str, int], 
                       n_recommendations: int = 5, 
                       all_users_data: Optional[Dict[str, Dict[str, int]]] = None,
                       implicit_signals: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Get movie recommendations using hybrid collaborative filtering (user-based + item-based) with confidence scoring."""
    recommender = create_recommender(df)
    recommender.update_user_ratings(user_ratings)
    
    # Enhanced recommendations with confidence scoring
    result = recommender.get_recommendations(n_recommendations, all_users_data)
    
    # Add confidence score to the results
    if not result.empty:
        confidence = get_collaborative_confidence(len([r for r in user_ratings.values() if r > 0]), implicit_signals)
        result['collaborative_confidence'] = confidence
        result['recommendation_method'] = 'hybrid' if confidence > 0.4 else 'content_fallback'
    
    return result

def get_collaborative_confidence(num_ratings: int, implicit_signals: Optional[Dict[str, float]] = None) -> float:
    """Calculate confidence score for collaborative filtering recommendations."""
    if num_ratings == 0:
        return 0.0
    elif num_ratings == 1:
        base_confidence = 0.2
    elif num_ratings == 2:
        base_confidence = 0.4
    else:
        base_confidence = min(0.8 + (num_ratings - 3) * 0.05, 1.0)
    
    # Boost confidence with implicit signals
    if implicit_signals:
        implicit_boost = min(len(implicit_signals) * 0.1, 0.3)
        base_confidence = min(base_confidence + implicit_boost, 1.0)
    
    return base_confidence


def get_recommendations_with_users_data(df: pd.DataFrame, user_ratings: Dict[str, int], 
                                      n_recommendations: int = 5) -> pd.DataFrame:
    """Get recommendations explicitly using actual user data for collaborative filtering."""
    users_data = load_users_data()
    return get_recommendations(df, user_ratings, n_recommendations, users_data)