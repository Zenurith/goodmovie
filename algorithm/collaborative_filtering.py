import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.decomposition import TruncatedSVD
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
        self.svd_model = None
        self._users_data_cache = None
        self._similarity_cache = {}
        
    def update_user_ratings(self, user_ratings: Dict[str, int]) -> None:
        """Update the user ratings dictionary."""
        self.user_ratings = user_ratings.copy()
        # Clear caches when user ratings change
        self._similarity_cache.clear()
    
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
        """Build user-item rating matrix with performance optimizations."""
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
        
        # Build matrix more efficiently
        matrix_data = []
        user_ids = []
        
        for user_id, ratings in clean_users_data.items():
            if isinstance(ratings, dict) and ratings:  # Only include users with ratings
                user_row = []
                for movie_id in all_movie_ids:
                    user_row.append(ratings.get(movie_id, 0))  # 0 for unrated movies
                matrix_data.append(user_row)
                user_ids.append(user_id)
        
        if not matrix_data:
            return pd.DataFrame()
            
        return pd.DataFrame(matrix_data, index=user_ids, columns=all_movie_ids)
    
    @lru_cache(maxsize=128)
    def _calculate_user_similarity(self, user_item_matrix_hash: int, target_user: str) -> pd.Series:
        """
        Calculate similarity between target user and all other users with caching.
        """
        if target_user not in self.user_item_matrix.index:
            return pd.Series(dtype=float)
        
        target_ratings = self.user_item_matrix.loc[target_user].values.reshape(1, -1)
        
        # Calculate cosine similarity with all users
        similarities = cosine_similarity(target_ratings, self.user_item_matrix.values)[0]
        
        return pd.Series(similarities, index=self.user_item_matrix.index)
    
    def _build_item_similarity_matrix(self) -> np.ndarray:
        """
        Build item-item similarity matrix based on content features with improvements.
        """
        try:
            # Create content features for movies
            movies_with_features = self.df.copy()
            
            # Combine genre and other features into text
            feature_texts = []
            for _, movie in movies_with_features.iterrows():
                # Combine genre, normalized rating, and popularity into feature text
                genres = str(movie.get('genre', '')).replace(',', ' ')
                
                # Create rating and popularity buckets
                rating = float(movie.get('vote_average', 0))
                rating_bucket = f"rating_{int(rating)}" if rating > 0 else ""
                
                popularity = float(movie.get('popularity', 0))
                if popularity > 80:
                    pop_bucket = "very_popular"
                elif popularity > 40:
                    pop_bucket = "popular"
                elif popularity > 10:
                    pop_bucket = "moderate"
                else:
                    pop_bucket = "niche"
                
                # Add language and year if available
                language = str(movie.get('original_language', ''))
                
                feature_text = f"{genres} {rating_bucket} {pop_bucket} {language}"
                feature_texts.append(feature_text.strip())
            
            # Create TF-IDF vectors with improved parameters
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=2000,  # Increased for better feature representation
                ngram_range=(1, 2),
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8  # Ignore terms that appear in more than 80% of documents
            )
            
            tfidf_matrix = vectorizer.fit_transform(feature_texts)
            
            # Calculate cosine similarity
            item_similarity = cosine_similarity(tfidf_matrix)
            
            return item_similarity
            
        except Exception as e:
            logging.error(f"Error building item similarity matrix: {e}")
            return np.eye(len(self.df))  # Return identity matrix as fallback
    
    def _build_svd_model(self, user_item_matrix: pd.DataFrame, n_components: int = 50):
        """Build SVD model with improved handling."""
        try:
            if user_item_matrix.empty:
                return False
                
            # Replace 0s with NaN for missing values
            matrix_filled = user_item_matrix.copy()
            matrix_filled = matrix_filled.replace(0, np.nan)
            
            # Fill NaN with user means, then global mean
            for user_id in matrix_filled.index:
                user_mean = matrix_filled.loc[user_id].mean()
                if np.isnan(user_mean):
                    user_mean = matrix_filled.stack().mean()  # Global mean
                matrix_filled.loc[user_id] = matrix_filled.loc[user_id].fillna(user_mean)
            
            # Determine appropriate number of components
            min_dim = min(matrix_filled.shape[0], matrix_filled.shape[1])
            n_components = min(n_components, min_dim - 1, 20)
            
            if n_components < 2:
                return False
                
            # Apply SVD
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_model.fit(matrix_filled.fillna(0))
            
            return True
            
        except Exception as e:
            logging.error(f"Error building SVD model: {e}")
            return False
    
    def get_recommendations(self, n_recommendations: int = 5, 
                          all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> pd.DataFrame:
        """
        Generate movie recommendations using optimized collaborative filtering.
        """
        try:
            # Validate input
            valid_ratings = {k: v for k, v in self.user_ratings.items() if v > 0 and v <= 10}
            
            if not valid_ratings or len(valid_ratings) < 1:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Load user data if not provided
            if not all_users_data:
                all_users_data = self._load_users_data()
            
            # If we have multiple users' data, use collaborative filtering
            if all_users_data and len(all_users_data) > 3:  # Need at least 4 users for good CF
                return self._get_fast_collaborative_recommendations(n_recommendations, all_users_data)
            else:
                # Fall back to content-based recommendations for speed
                return self._get_content_based_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in get_recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_fast_collaborative_recommendations(self, n_recommendations: int, 
                                               all_users_data: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Get recommendations using fast user-based collaborative filtering only.
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
            
            # Use only user-based collaborative filtering for speed
            user_cf_recs = self._get_user_based_recommendations(n_recommendations, all_users_data)
            
            # If we don't have enough recommendations, supplement with content-based
            if len(user_cf_recs) < n_recommendations:
                content_recs = self._get_content_based_recommendations(n_recommendations - len(user_cf_recs))
                
                if not user_cf_recs.empty and not content_recs.empty:
                    # Avoid duplicates when combining
                    cf_movie_ids = set(user_cf_recs['id'].astype(str))
                    content_recs = content_recs[~content_recs['id'].astype(str).isin(cf_movie_ids)]
                    
                    # Combine results
                    import pandas as pd
                    combined_recs = pd.concat([user_cf_recs, content_recs], ignore_index=True)
                    return combined_recs.head(n_recommendations)
                elif not user_cf_recs.empty:
                    return user_cf_recs
                else:
                    return content_recs
            
            return user_cf_recs if not user_cf_recs.empty else self._get_content_based_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in fast collaborative recommendations: {e}")
            return self._get_content_based_recommendations(n_recommendations)
    
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
            
            # Find most similar users (reduced number for speed)
            similar_users = user_similarities.drop(current_user_id, errors='ignore').nlargest(3)  # Only top 3 most similar
            
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
        Get recommendations using item-based collaborative filtering.
        """
        try:
            if not self.user_ratings:
                return pd.DataFrame()
            
            # Build item similarity matrix if not already built
            if self.item_similarity_matrix is None:
                self.item_similarity_matrix = self._build_item_similarity_matrix()
            
            # Get all movie IDs from the dataset
            all_movie_ids = self.df['id'].astype(str).tolist()
            rated_movie_ids = set(self.user_ratings.keys())
            unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
            
            if not unrated_movie_ids:
                return pd.DataFrame()
            
            # Calculate scores for unrated movies
            movie_scores = []
            
            for unrated_movie_id in unrated_movie_ids:
                score = 0.0
                weight_sum = 0.0
                
                # Find this movie's index in the dataset
                try:
                    movie_idx = all_movie_ids.index(unrated_movie_id)
                except ValueError:
                    continue
                
                # Calculate score based on similarity to rated movies
                for rated_movie_id, rating in self.user_ratings.items():
                    if rating <= 0:
                        continue
                    
                    try:
                        rated_idx = all_movie_ids.index(rated_movie_id)
                    except ValueError:
                        continue
                    
                    if rated_idx < len(self.item_similarity_matrix) and movie_idx < len(self.item_similarity_matrix[rated_idx]):
                        similarity = self.item_similarity_matrix[rated_idx][movie_idx]
                        if similarity > 0.1:  # Only consider reasonably similar items
                            user_avg = np.mean(list(self.user_ratings.values()))
                            score += similarity * (rating - user_avg)
                            weight_sum += similarity
                
                if weight_sum > 0:
                    normalized_score = score / weight_sum
                    movie_scores.append((unrated_movie_id, normalized_score))
            
            # Sort by score and get top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get movie data for top recommendations
            recommended_movies = []
            for movie_id, score in movie_scores[:n_recommendations * 2]:
                if score > 0:
                    movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                    if not movie_row.empty:
                        movie_dict = movie_row.iloc[0].to_dict()
                        movie_dict['item_score'] = score
                        recommended_movies.append(movie_dict)
            
            result_df = pd.DataFrame(recommended_movies)
            
            # Filter by quality
            if not result_df.empty:
                result_df = result_df[result_df['vote_average'] >= 6.0]
                result_df = result_df.head(n_recommendations)
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error in item-based recommendations: {e}")
            return pd.DataFrame()
    
    def _get_svd_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Get recommendations using SVD matrix factorization.
        """
        try:
            if self.svd_model is None or self.user_item_matrix.empty:
                return pd.DataFrame()
                
            current_user_id = "current_user"
            if current_user_id not in self.user_item_matrix.index:
                return pd.DataFrame()
            
            # Get user vector and predict ratings
            user_vector = self.user_item_matrix.loc[current_user_id].values.reshape(1, -1)
            user_latent = self.svd_model.transform(user_vector)
            reconstructed = self.svd_model.inverse_transform(user_latent)[0]
            
            # Get unrated movies and their predicted ratings
            movie_predictions = []
            rated_movie_ids = set(self.user_ratings.keys())
            
            for i, movie_id in enumerate(self.user_item_matrix.columns):
                if movie_id not in rated_movie_ids and i < len(reconstructed):
                    predicted_rating = reconstructed[i]
                    if predicted_rating > 6.0:
                        movie_predictions.append((movie_id, predicted_rating))
            
            # Sort by predicted rating
            movie_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get movie data for recommendations
            recommended_movies = []
            for movie_id, pred_rating in movie_predictions[:n_recommendations]:
                movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                if not movie_row.empty:
                    movie_dict = movie_row.iloc[0].to_dict()
                    movie_dict['svd_score'] = pred_rating
                    recommended_movies.append(movie_dict)
            
            return pd.DataFrame(recommended_movies)
            
        except Exception as e:
            logging.error(f"Error in SVD recommendations: {e}")
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
                movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                if not movie_row.empty:
                    genres = str(movie_row.iloc[0]['genre']).split(',')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            genre_scores[genre] = genre_scores.get(genre, 0) + rating
            
            if not genre_scores:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Get top 3 genres only
            top_genres = sorted(genre_scores.keys(), key=lambda g: genre_scores[g], reverse=True)[:3]
            
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
            
            return pd.DataFrame(recommendations[:n_recommendations])
            
        except Exception as e:
            logging.error(f"Error in content-based recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _calculate_content_score(self, movie: pd.Series, top_genres: List[str], top_languages: List[str],
                               genre_scores: Dict[str, float], language_scores: Dict[str, float], user_avg_rating: float) -> float:
        """
        Calculate improved content-based score.
        """
        try:
            # Get movie genres and language
            movie_genres = str(movie['genre']).split(',')
            movie_genres = [g.strip() for g in movie_genres if g.strip()]
            movie_language = str(movie.get('original_language', ''))
            
            if not movie_genres:
                return 0.0
            
            # Genre matching score
            genre_match_score = 0.0
            for genre in movie_genres:
                if genre in genre_scores:
                    genre_match_score += genre_scores[genre] / 10.0  # Normalize
            
            if genre_match_score == 0:
                return 0.0  # No genre match
            
            # Language bonus
            language_bonus = 0.0
            if movie_language in language_scores:
                language_bonus = language_scores[movie_language] / 20.0  # Smaller weight
            
            # Quality scores
            vote_avg = float(movie.get('vote_average', 0))
            quality_score = vote_avg / 10.0
            
            # Popularity score (capped and normalized)
            popularity = float(movie.get('popularity', 0))
            popularity_score = min(popularity / 100.0, 1.0) * 0.3
            
            # Vote count bonus (prefer movies with more votes)
            vote_count = float(movie.get('vote_count', 0))
            vote_count_bonus = min(vote_count / 1000.0, 1.0) * 0.2
            
            # Rating alignment bonus
            rating_bonus = 0.5 if vote_avg >= user_avg_rating else 0.0
            
            # Final score
            total_score = (genre_match_score + language_bonus + quality_score + 
                          popularity_score + vote_count_bonus + rating_bonus)
            
            return total_score
            
        except Exception:
            return 0.0
    
    def _combine_recommendations(self, rec_sources: List[tuple], n_recommendations: int) -> pd.DataFrame:
        """
        Combine recommendations from multiple sources with weighted scoring.
        """
        try:
            movie_scores = {}
            movie_data = {}
            
            for recs_df, weight in rec_sources:
                if recs_df.empty:
                    continue
                    
                for _, movie in recs_df.iterrows():
                    movie_id = str(movie['id'])
                    
                    # Store movie data
                    if movie_id not in movie_data:
                        movie_data[movie_id] = movie.to_dict()
                    
                    # Calculate weighted score
                    base_score = movie.get('vote_average', 5.0) / 10.0
                    
                    # Add method-specific scores if available
                    method_score = 0
                    if 'cf_score' in movie:
                        method_score = movie['cf_score']
                    elif 'item_score' in movie:
                        method_score = movie['item_score'] * 5  # Scale up item scores
                    elif 'svd_score' in movie:
                        method_score = movie['svd_score']
                    elif 'content_score' in movie:
                        method_score = movie['content_score'] * 2  # Scale up content scores
                    
                    combined_score = (base_score + method_score) * weight
                    
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = 0
                    movie_scores[movie_id] += combined_score
            
            # Sort by combined score
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create result DataFrame
            recommended_movies = []
            for movie_id, score in sorted_movies[:n_recommendations]:
                if movie_id in movie_data:
                    movie_dict = movie_data[movie_id].copy()
                    movie_dict['combined_score'] = score
                    recommended_movies.append(movie_dict)
            
            return pd.DataFrame(recommended_movies)
            
        except Exception as e:
            logging.error(f"Error combining recommendations: {e}")
            return pd.DataFrame()
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """Get high-quality popular movies as fallback."""
        try:
            return self.df[
                (self.df['vote_average'] >= 7.5) & 
                (self.df['vote_count'] >= 1000)
            ].nlargest(n_recommendations, ['vote_average', 'popularity'])
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
                       all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> pd.DataFrame:
    """Get movie recommendations using improved hybrid collaborative filtering."""
    recommender = create_recommender(df)
    recommender.update_user_ratings(user_ratings)
    
    # Automatically load user data if not provided
    return recommender.get_recommendations(n_recommendations, all_users_data)


def get_recommendations_with_users_data(df: pd.DataFrame, user_ratings: Dict[str, int], 
                                      n_recommendations: int = 5) -> pd.DataFrame:
    """Get recommendations explicitly using actual user data for collaborative filtering."""
    users_data = load_users_data()
    return get_recommendations(df, user_ratings, n_recommendations, users_data)