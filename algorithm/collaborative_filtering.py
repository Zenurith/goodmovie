

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


class CollaborativeFilter:

    
    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()
        self.user_ratings = {}
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        
    def update_user_ratings(self, user_ratings: Dict[str, int]) -> None:
        """Update the user ratings dictionary."""
        self.user_ratings = user_ratings.copy()
    
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

        # Clean and extract ratings data, handling nested structure
        clean_users_data = self._extract_ratings_from_users_data(all_users_data)
        
        # Get all unique movie IDs
        all_movie_ids = set()
        for user_ratings in clean_users_data.values():
            if isinstance(user_ratings, dict):
                all_movie_ids.update(user_ratings.keys())
        
        all_movie_ids = sorted(list(all_movie_ids))
        
        # Build matrix
        matrix_data = []
        user_ids = []
        
        for user_id, ratings in clean_users_data.items():
            if isinstance(ratings, dict):
                user_row = []
                for movie_id in all_movie_ids:
                    user_row.append(ratings.get(movie_id, 0))  # 0 for unrated movies
                matrix_data.append(user_row)
                user_ids.append(user_id)
        
        return pd.DataFrame(matrix_data, index=user_ids, columns=all_movie_ids)
    
    def _calculate_user_similarity(self, user_item_matrix: pd.DataFrame, target_user: str) -> pd.Series:
        """
        Calculate similarity between target user and all other users.
        
        Args:
            user_item_matrix: User-item rating matrix
            target_user: ID of target user
            
        Returns:
            Series of similarity scores
        """
        if target_user not in user_item_matrix.index:
            return pd.Series(dtype=float)
        
        target_ratings = user_item_matrix.loc[target_user].values.reshape(1, -1)
        
        # Calculate cosine similarity with all users
        similarities = cosine_similarity(target_ratings, user_item_matrix.values)[0]
        
        return pd.Series(similarities, index=user_item_matrix.index)
    
    def _build_item_similarity_matrix(self) -> np.ndarray:
        """
        Build item-item similarity matrix based on content features.
        
        Returns:
            Item similarity matrix
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
                pop_bucket = ""
                if popularity > 80:
                    pop_bucket = "very_popular"
                elif popularity > 40:
                    pop_bucket = "popular"
                elif popularity > 10:
                    pop_bucket = "moderate"
                else:
                    pop_bucket = "niche"
                
                feature_text = f"{genres} {rating_bucket} {pop_bucket}"
                feature_texts.append(feature_text.strip())
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(feature_texts)
            
            # Calculate cosine similarity
            item_similarity = cosine_similarity(tfidf_matrix)
            
            return item_similarity
            
        except Exception as e:
            logging.error(f"Error building item similarity matrix: {e}")
            return np.eye(len(self.df))  # Return identity matrix as fallback
    
    def get_recommendations(self, n_recommendations: int = 5, 
                          all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> pd.DataFrame:
        """
        Generate movie recommendations using collaborative filtering.
        
        Args:
            n_recommendations: Number of recommendations to return
            all_users_data: Dictionary of all users' ratings for collaborative filtering
            
        Returns:
            DataFrame containing recommended movies
        """
        try:
            # Validate input
            valid_ratings = {k: v for k, v in self.user_ratings.items() if v > 0 and v <= 10}
            
            if not valid_ratings or len(valid_ratings) < 2:
                return self._get_content_based_recommendations(n_recommendations)
            
            # If we have multiple users' data, use collaborative filtering
            if all_users_data and len(all_users_data) > 1:
                return self._get_collaborative_recommendations(n_recommendations, all_users_data)
            else:
                # Fall back to improved content-based recommendations
                return self._get_content_based_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in get_recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_collaborative_recommendations(self, n_recommendations: int, 
                                         all_users_data: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Get recommendations using user-based collaborative filtering.
        """
        try:
            # Add current user to the data
            current_user_id = "current_user"
            all_users_with_current = all_users_data.copy()
            all_users_with_current[current_user_id] = self.user_ratings
            
            # Build user-item matrix
            user_item_matrix = self._build_user_item_matrix(all_users_with_current)
            
            # Calculate user similarities
            user_similarities = self._calculate_user_similarity(user_item_matrix, current_user_id)
            
            # Find most similar users (excluding current user) - increased count and lowered threshold
            similar_users = user_similarities.drop(current_user_id).nlargest(10)
            
            if similar_users.empty or similar_users.max() < 0.05:  # Lowered threshold from 0.1 to 0.05
                # No similar users found, use item-based collaborative filtering
                return self._get_item_based_recommendations(n_recommendations)
            
            # Get movie recommendations from similar users using reference approach
            movie_scores = {}
            rated_movies = set(self.user_ratings.keys())
            
            # Get clean user ratings data
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            
            for similar_user, similarity_score in similar_users.items():
                if similarity_score < 0.05:  # Lowered threshold
                    continue
                
                if similar_user not in clean_all_users_data:
                    continue
                user_ratings = clean_all_users_data[similar_user]
                
                for movie_id, rating in user_ratings.items():
                    if movie_id not in rated_movies and rating >= 6:  # Lowered from 7 to 6 to include more movies
                        if movie_id not in movie_scores:
                            movie_scores[movie_id] = 0
                        # Use reference-style scoring: similarity * (rating - average_rating)
                        user_avg = np.mean(list(self.user_ratings.values())) if self.user_ratings else 5.0
                        score_contribution = similarity_score * (rating - user_avg)
                        movie_scores[movie_id] += score_contribution
            
            # Sort and get top recommendations
            if not movie_scores:
                return self._get_item_based_recommendations(n_recommendations)
            
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get movie data for recommendations
            recommended_movies = []
            for movie_id, score in sorted_movies[:n_recommendations * 2]:  # Get more candidates
                movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                if not movie_row.empty:
                    recommended_movies.append(movie_row.iloc[0].to_dict())
            
            result_df = pd.DataFrame(recommended_movies)
            
            # Filter by quality to ensure good recommendations
            if not result_df.empty:
                result_df = result_df[result_df['vote_average'] >= 6.0]  # Ensure minimum quality
                result_df = result_df.head(n_recommendations)
            
            # If we don't have enough, supplement with item-based
            if len(result_df) < n_recommendations:
                remaining = n_recommendations - len(result_df)
                item_recs = self._get_item_based_recommendations(remaining)
                
                # Avoid duplicates
                existing_ids = set(result_df['id'].astype(str)) if not result_df.empty else set()
                new_recs = item_recs[~item_recs['id'].astype(str).isin(existing_ids)]
                
                if not new_recs.empty:
                    if result_df.empty:
                        result_df = new_recs.head(remaining)
                    else:
                        result_df = pd.concat([result_df, new_recs.head(remaining)], ignore_index=True)
            
            return result_df.head(n_recommendations) if not result_df.empty else self._get_fallback_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in collaborative recommendations: {e}")
            return self._get_item_based_recommendations(n_recommendations)
    
    def _get_item_based_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Get recommendations using item-based collaborative filtering (like the reference).
        """
        try:
            if not self.user_ratings:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Build item similarity matrix if not already built
            if self.item_similarity_matrix is None:
                self.item_similarity_matrix = self._build_item_similarity_matrix()
            
            # Get all movie IDs from the dataset
            all_movie_ids = self.df['id'].astype(str).tolist()
            rated_movie_ids = set(self.user_ratings.keys())
            unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
            
            if not unrated_movie_ids:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Calculate scores for unrated movies using reference approach
            movie_scores = []
            
            for unrated_movie_id in unrated_movie_ids:
                score = 0.0
                
                # Find this movie's index in the dataset
                movie_idx = None
                for idx, movie_id in enumerate(all_movie_ids):
                    if movie_id == unrated_movie_id:
                        movie_idx = idx
                        break
                
                if movie_idx is None:
                    continue
                
                # Calculate score based on similarity to rated movies
                for rated_movie_id, rating in self.user_ratings.items():
                    if rating <= 0:
                        continue
                    
                    # Find rated movie index
                    rated_idx = None
                    for idx, movie_id in enumerate(all_movie_ids):
                        if movie_id == rated_movie_id:
                            rated_idx = idx
                            break
                    
                    if rated_idx is None or rated_idx >= len(self.item_similarity_matrix):
                        continue
                    
                    # Get similarity score
                    if movie_idx < len(self.item_similarity_matrix[rated_idx]):
                        similarity = self.item_similarity_matrix[rated_idx][movie_idx]
                        # Use reference-style scoring: similarity * (rating - neutral_rating)
                        user_avg = np.mean(list(self.user_ratings.values()))
                        score += similarity * (rating - user_avg)
                
                movie_scores.append((unrated_movie_id, score))
            
            # Sort by score and get top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get movie data for top recommendations
            recommended_movies = []
            for movie_id, score in movie_scores[:n_recommendations * 2]:  # Get more candidates
                if score > 0:  # Only positive scores
                    movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                    if not movie_row.empty:
                        recommended_movies.append(movie_row.iloc[0].to_dict())
            
            result_df = pd.DataFrame(recommended_movies)
            
            # Filter by quality
            if not result_df.empty:
                result_df = result_df[result_df['vote_average'] >= 6.0]
                result_df = result_df.head(n_recommendations)
            
            if result_df.empty:
                return self._get_content_based_recommendations(n_recommendations)
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error in item-based recommendations: {e}")
            return self._get_content_based_recommendations(n_recommendations)
    
    def _get_content_based_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Get recommendations using improved content-based filtering.
        """
        try:
            # Analyze user preferences
            user_avg_rating = np.mean(list(self.user_ratings.values()))
            high_rated_movies = {k: v for k, v in self.user_ratings.items() if v >= max(7, user_avg_rating)}
            
            if not high_rated_movies:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Extract preferred genres
            genre_scores = {}
            for movie_id, rating in high_rated_movies.items():
                movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                if not movie_row.empty:
                    genres = str(movie_row.iloc[0]['genre']).split(',')
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            genre_scores[genre] = genre_scores.get(genre, 0) + rating
            
            # Get top genres
            top_genres = sorted(genre_scores.keys(), key=lambda g: genre_scores[g], reverse=True)[:3]
            
            if not top_genres:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Find unrated movies
            rated_movie_ids = set(str(k) for k in self.user_ratings.keys())
            unrated_movies = self.df[~self.df['id'].astype(str).isin(rated_movie_ids)].copy()
            
            # Score unrated movies
            movie_scores = []
            for _, movie in unrated_movies.iterrows():
                score = self._calculate_simple_content_score(movie, top_genres, genre_scores, user_avg_rating)
                if score > 0:
                    movie_scores.append((movie.to_dict(), score))
            
            # Sort and return top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            recommended_movies = [movie for movie, score in movie_scores[:n_recommendations]]
            
            return pd.DataFrame(recommended_movies)
            
        except Exception as e:
            logging.error(f"Error in content-based recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _calculate_simple_content_score(self, movie: pd.Series, top_genres: List[str], 
                                      genre_scores: Dict[str, float], user_avg_rating: float) -> float:
        """
        Calculate a simpler, more effective content-based score.
        """
        try:
            # Get movie genres
            movie_genres = str(movie['genre']).split(',')
            movie_genres = [g.strip() for g in movie_genres if g.strip()]
            
            if not movie_genres:
                return 0.0
            
            # Genre matching score
            genre_match_score = 0.0
            for genre in movie_genres:
                if genre in genre_scores:
                    genre_match_score += genre_scores[genre] / 10.0  # Normalize
            
            if genre_match_score == 0:
                return 0.0  # No genre match
            
            # Quality score
            vote_avg = float(movie.get('vote_average', 0))
            quality_score = vote_avg / 10.0
            
            # Popularity score (capped)
            popularity = min(float(movie.get('popularity', 0)) / 50.0, 1.0)
            
            # Rating alignment bonus
            rating_bonus = 0.5 if vote_avg >= user_avg_rating else 0.0
            
            # Final score
            total_score = genre_match_score + quality_score + popularity * 0.3 + rating_bonus
            
            return total_score
            
        except Exception:
            return 0.0
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """Get fallback recommendations when main algorithm fails."""
        try:
            return self.df.nlargest(n_recommendations, ['vote_average', 'popularity'])
        except Exception:
            return self.df.head(n_recommendations)
    
    def predict_rating(self, movie_id: str, all_users_data: Optional[Dict[str, Dict[str, int]]] = None) -> float:
        """
        Predict rating for a specific movie.
        
        Args:
            movie_id: ID of movie to predict rating for
            all_users_data: Dictionary of all users' ratings for collaborative filtering
            
        Returns:
            Predicted rating (1-10 scale)
        """
        try:
            # If movie already rated, return actual rating
            if movie_id in self.user_ratings:
                return float(self.user_ratings[movie_id])
            
            # Get movie data
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            if movie_row.empty:
                return 5.0  # Default neutral rating
                
            movie = movie_row.iloc[0]
            
            # Try collaborative filtering first
            if all_users_data and len(all_users_data) > 1:
                collab_rating = self._predict_collaborative_rating(movie_id, all_users_data)
                if collab_rating > 0:
                    # Add confidence-based adjustment
                    confidence = self._get_prediction_confidence(movie_id, all_users_data)
                    if confidence < 0.3:  # Low confidence, be more conservative
                        if collab_rating >= 7.0:
                            collab_rating = collab_rating * 0.9
                    return collab_rating
            
            # Fall back to content-based prediction
            content_rating = self._predict_content_based_rating(movie)
            # Content-based is less reliable, be more conservative
            if content_rating >= 7.0:
                content_rating = content_rating * 0.95
            return content_rating
            
        except Exception as e:
            logging.error(f"Error predicting rating for movie {movie_id}: {e}")
            return 5.0  # Default neutral rating
    
    def _predict_collaborative_rating(self, movie_id: str, all_users_data: Dict[str, Dict[str, int]]) -> float:
        """
        Predict rating using collaborative filtering.
        """
        try:
            # Build user-item matrix
            current_user_id = "current_user"
            all_users_with_current = all_users_data.copy()
            all_users_with_current[current_user_id] = self.user_ratings
            
            user_item_matrix = self._build_user_item_matrix(all_users_with_current)
            
            # Calculate user similarities
            user_similarities = self._calculate_user_similarity(user_item_matrix, current_user_id)
            similar_users = user_similarities.drop(current_user_id).nlargest(15)  # Consider more users
            
            if similar_users.empty:
                return 0.0
            
            # Weighted average of similar users' ratings
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            # Get clean user ratings data
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            
            for similar_user, similarity_score in similar_users.items():
                if similarity_score > 0.05 and similar_user in clean_all_users_data and movie_id in clean_all_users_data[similar_user]:  # Lowered threshold
                    user_rating = clean_all_users_data[similar_user][movie_id]
                    # Use linear weighting instead of exponential to be less aggressive
                    weight = similarity_score
                    weighted_sum += weight * user_rating
                    similarity_sum += weight
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                # Minimal bias adjustment - trust the collaborative prediction more
                user_avg = np.mean(list(self.user_ratings.values())) if self.user_ratings else 5.0
                global_avg = 6.0  # Assume global average
                bias_adjusted = predicted_rating + (user_avg - global_avg) * 0.1  # Reduced bias influence
                
                # Remove overly conservative adjustment - trust the algorithm
                return max(1.0, min(10.0, bias_adjusted))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_prediction_confidence(self, movie_id: str, all_users_data: Dict[str, Dict[str, int]]) -> float:
        """
        Calculate confidence score for a prediction (0-1 scale).
        Higher confidence = more reliable prediction.
        """
        try:
            # Build user-item matrix
            current_user_id = "current_user"
            all_users_with_current = all_users_data.copy()
            all_users_with_current[current_user_id] = self.user_ratings
            
            user_item_matrix = self._build_user_item_matrix(all_users_with_current)
            user_similarities = self._calculate_user_similarity(user_item_matrix, current_user_id)
            similar_users = user_similarities.drop(current_user_id).nlargest(10)
            
            # Factors that increase confidence:
            confidence_factors = []
            
            # 1. Number of similar users who rated this movie
            users_who_rated = 0
            total_similarity = 0
            clean_all_users_data = self._extract_ratings_from_users_data(all_users_data)
            for user, sim_score in similar_users.items():
                if sim_score > 0.1 and user in clean_all_users_data and movie_id in clean_all_users_data[user]:
                    users_who_rated += 1
                    total_similarity += sim_score
            
            # Confidence from number of raters (0-0.4)
            rater_confidence = min(users_who_rated / 5.0, 0.4)
            confidence_factors.append(rater_confidence)
            
            # 2. Average similarity of users who rated it (0-0.3)
            avg_similarity = total_similarity / users_who_rated if users_who_rated > 0 else 0
            sim_confidence = min(avg_similarity, 0.3)
            confidence_factors.append(sim_confidence)
            
            # 3. How many ratings the current user has (0-0.3)
            user_experience = min(len(self.user_ratings) / 20.0, 0.3)
            confidence_factors.append(user_experience)
            
            return sum(confidence_factors)
            
        except Exception:
            return 0.2  # Low default confidence
    
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
                # No genre overlap, use movie's inherent quality adjusted to user's preference
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
                # Combine with movie's inherent quality - balanced approach
                movie_quality = float(movie.get('vote_average', 5.0))
                predicted_rating = genre_prediction * 0.7 + movie_quality * 0.3  # More weight to user preference
                
                # Remove overly conservative penalty
                return max(1.0, min(10.0, predicted_rating))
            else:
                # No genre match, still give reasonable prediction
                movie_rating = float(movie.get('vote_average', 5.0))
                conservative_pred = user_avg_rating * 0.4 + movie_rating * 0.6
                
                # Remove harsh penalty for no genre match
                return max(1.0, min(10.0, conservative_pred))
            
        except Exception:
            return 5.0

    def get_user_stats(self) -> Dict[str, any]:
        """
        Get statistics about user's rating patterns.
        """
        rated_movies = [movie_id for movie_id, rating in self.user_ratings.items() if rating > 0]
        
        if not rated_movies:
            return {
                'total_rated': 0,
                'avg_rating': 0.0,
                'favorite_genres': [],
                'total_watch_time': 0
            }
        
        ratings = [self.user_ratings[movie_id] for movie_id in rated_movies]
        avg_rating = np.mean(ratings)
        
        # Get favorite genres
        genre_counts = {}
        for movie_id in rated_movies:
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            if not movie_row.empty:
                genres = str(movie_row.iloc[0]['genre']).split(',')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        favorite_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_rated': len(rated_movies),
            'avg_rating': round(avg_rating, 1),
            'favorite_genres': [genre for genre, count in favorite_genres],
            'genre_distribution': dict(favorite_genres)
        }


def create_recommender(df: pd.DataFrame) -> CollaborativeFilter:
    """
    Factory function to create a collaborative filter instance.
    
    Args:
        df: Movie dataframe
        
    Returns:
        CollaborativeFilter instance
    """
    return CollaborativeFilter(df)


def get_recommendations(df: pd.DataFrame, user_ratings: Dict[str, int], 
                       n_recommendations: int = 5) -> pd.DataFrame:
    """
    Convenience function to get movie recommendations.
    
    Args:
        df: Movie dataframe
        user_ratings: Dictionary of user ratings {movie_id: rating}
        n_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame containing recommended movies
    """
    recommender = create_recommender(df)
    recommender.update_user_ratings(user_ratings)
    return recommender.get_recommendations(n_recommendations)