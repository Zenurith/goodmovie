"""
Collaborative Filtering Algorithm for Movie Recommendations

This module implements a content-based collaborative filtering system
that recommends movies based on user ratings and genre preferences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging


class CollaborativeFilter:
    """
    A collaborative filtering system that recommends movies based on:
    1. User's rating patterns
    2. Genre preferences derived from highly-rated movies
    3. Content similarity between movies
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the collaborative filtering system.
        
        Args:
            df: DataFrame containing movie data with columns:
                - id: movie ID
                - title: movie title
                - genre: comma-separated genres
                - vote_average: average rating
                - popularity: movie popularity score
        """
        self.df = df.copy()
        self.user_ratings = {}
        
    def update_user_ratings(self, user_ratings: Dict[str, int]) -> None:
        """Update the user ratings dictionary."""
        self.user_ratings = user_ratings.copy()
    
    def _get_user_preferences(self) -> Tuple[Dict[str, int], float, List[str], Dict[str, float]]:
        """
        Analyze user preferences based on their ratings.
        
        Returns:
            Tuple of (rated_movies, user_avg_rating, top_genres_ordered, genre_scores)
        """
        try:
            # Get user's rated movies and their average rating
            rated_movies = {k: v for k, v in self.user_ratings.items() if v > 0 and v <= 10}
            
            if not rated_movies:
                return {}, 0.0, [], {}
                
            user_avg_rating = np.mean(list(rated_movies.values()))
            
            # Get genres of highly rated movies (>= 7 or above average)
            highly_rated_threshold = max(6, user_avg_rating)  # Lowered threshold for better matching
            highly_rated_movies = {k: v for k, v in rated_movies.items() 
                                 if v >= highly_rated_threshold}
            
            # If no highly rated movies, use above average movies
            if not highly_rated_movies:
                highly_rated_movies = {k: v for k, v in rated_movies.items() 
                                     if v >= user_avg_rating}
            
            # Extract and score genres from highly rated movies
            genre_scores = {}
            for movie_id, rating in highly_rated_movies.items():
                try:
                    # Handle both string and numeric IDs
                    movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
                    if movie_row.empty:
                        # Try by title if ID doesn't match
                        movie_row = self.df[self.df['title'].str.lower() == str(movie_id).lower()]
                    
                    if not movie_row.empty:
                        genres_str = str(movie_row.iloc[0]['genre'])
                        if genres_str and genres_str != 'nan':
                            genres = genres_str.split(',')
                            for genre in genres:
                                genre = genre.strip()
                                if genre and len(genre) > 0:
                                    # Score genres based on user rating and frequency
                                    weight_factor = min(rating / 10.0 * 2.0, 2.0)  # Normalize rating influence
                                    if genre in genre_scores:
                                        genre_scores[genre] += weight_factor
                                    else:
                                        genre_scores[genre] = weight_factor
                except Exception as e:
                    logging.warning(f"Error processing movie {movie_id}: {e}")
                    continue
            
            # Sort genres by score (highest preference first)
            top_genres_ordered = sorted(genre_scores.keys(), 
                                      key=lambda g: genre_scores[g], reverse=True)
            
            return rated_movies, user_avg_rating, top_genres_ordered, genre_scores
            
        except Exception as e:
            logging.error(f"Error in _get_user_preferences: {e}")
            return {}, 0.0, [], {}
    
    def _calculate_movie_score(self, movie: pd.Series, top_genres_ordered: List[str], 
                              genre_scores: Dict[str, float], user_avg_rating: float) -> float:
        """
        Calculate recommendation score for a movie with enhanced genre matching.
        
        Args:
            movie: Movie data series
            top_genres_ordered: List of user's preferred genres in order of preference
            genre_scores: Dictionary of genre preference scores
            user_avg_rating: User's average rating
            
        Returns:
            Recommendation score (higher is better)
        """
        try:
            # Validate movie data
            if pd.isna(movie['genre']) or str(movie['genre']) == 'nan':
                return 0.0
            
            # Get movie genres with better error handling
            genre_str = str(movie['genre'])
            movie_genres = [g.strip() for g in genre_str.split(',') if g.strip() and len(g.strip()) > 0]
            movie_genre_set = set(movie_genres)
            
            if not movie_genres:
                return 0.0
            
            # Calculate enhanced genre matching
            genre_match_score = 0.0
            matched_genres = movie_genre_set.intersection(set(top_genres_ordered))
            
            if not matched_genres:
                return 0.0  # No genre match, not recommended
            
            # Priority scoring for top genres with safer calculation
            for i, preferred_genre in enumerate(top_genres_ordered[:5]):  # Focus on top 5 genres
                if preferred_genre in movie_genres:
                    base_score = genre_scores.get(preferred_genre, 1.0)
                    # Higher weight for top genres (first 2 get highest priority)
                    if i == 0:  # Top genre gets highest weight
                        genre_match_score += base_score * 2.0
                    elif i == 1:  # Second genre gets high weight
                        genre_match_score += base_score * 1.5
                    elif i == 2:  # Third genre gets medium weight
                        genre_match_score += base_score * 1.2
                    else:  # Other genres get base weight
                        genre_match_score += base_score * 0.8
            
            # Multi-genre bonus: Extra points for movies matching multiple top genres
            top_2_genres = set(top_genres_ordered[:2])
            top_3_genres = set(top_genres_ordered[:3])
            
            multi_genre_bonus = 0.0
            matched_top_2 = len(movie_genre_set.intersection(top_2_genres))
            matched_top_3 = len(movie_genre_set.intersection(top_3_genres))
            
            if matched_top_2 >= 2:
                # Perfect match: movie has both top 2 preferred genres
                multi_genre_bonus += 4.0
            elif matched_top_3 >= 2:
                # Good match: movie has 2 of top 3 preferred genres
                multi_genre_bonus += 2.5
            elif len(matched_genres) >= 2:
                # Decent match: movie has 2 or more preferred genres
                multi_genre_bonus += 1.5
            
            # Base scoring components with validation
            vote_avg = float(movie.get('vote_average', 0))
            popularity = float(movie.get('popularity', 0))
            
            quality_score = max(0, min(vote_avg / 10.0 * 2.5, 2.5))  # Scale 0-2.5
            popularity_score = max(0, min(popularity / 100.0, 1.5))    # Cap at 1.5
            
            # Rating alignment bonus
            rating_bonus = 0.0
            if vote_avg >= user_avg_rating:
                rating_bonus += 0.8
            if vote_avg >= 8.0:  # High quality movies get extra boost
                rating_bonus += 0.7
            if vote_avg >= 9.0:  # Exceptional movies get maximum boost
                rating_bonus += 0.5
            
            # Final score calculation
            total_score = (
                genre_match_score * 1.2 +   # User's genre preferences (boosted)
                multi_genre_bonus +          # Multi-genre matching bonus
                quality_score +              # Movie quality
                popularity_score +           # Movie popularity (capped)
                rating_bonus                # Rating alignment bonus
            )
            
            return max(0.0, total_score)  # Ensure non-negative score
            
        except Exception as e:
            logging.error(f"Error calculating movie score: {e}")
            return 0.0
    
    def get_recommendations(self, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Generate movie recommendations using collaborative filtering.
        
        Args:
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame containing recommended movies
        """
        try:
            # Check if user has rated enough movies
            valid_ratings = {k: v for k, v in self.user_ratings.items() if v > 0 and v <= 10}
            
            if not valid_ratings or len(valid_ratings) < 1:
                # Fallback to popular movies if insufficient ratings
                return self._get_fallback_recommendations(n_recommendations)
            
            # Analyze user preferences with enhanced genre analysis
            rated_movies, user_avg_rating, top_genres_ordered, genre_scores = self._get_user_preferences()
            
            if not top_genres_ordered or not genre_scores:
                # Fallback if no genre preferences found
                return self._get_fallback_recommendations(n_recommendations)
            
            # Find unrated movies with better filtering
            rated_movie_ids = set(str(k) for k in self.user_ratings.keys())
            unrated_movies = self.df[~self.df['id'].astype(str).isin(rated_movie_ids)].copy()
            
            # Filter out movies with missing essential data
            unrated_movies = unrated_movies.dropna(subset=['genre', 'vote_average', 'popularity'])
            
            if unrated_movies.empty:
                return pd.DataFrame()  # All movies have been rated
            
            # Score and rank unrated movies with enhanced algorithm
            movie_scores = []
            for idx, movie in unrated_movies.iterrows():
                try:
                    score = self._calculate_movie_score(movie, top_genres_ordered, genre_scores, user_avg_rating)
                    if score > 0:  # Only consider movies with positive scores
                        movie_scores.append((movie.to_dict(), score))
                except Exception as e:
                    logging.warning(f"Error scoring movie {movie.get('title', 'Unknown')}: {e}")
                    continue
            
            if not movie_scores:
                # Try with relaxed criteria
                return self._get_relaxed_recommendations(unrated_movies, top_genres_ordered, n_recommendations)
            
            # Sort by score and get top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select diverse recommendations (avoid too many from same genre)
            recommended_movies = self._diversify_recommendations(
                movie_scores, n_recommendations, top_genres_ordered[:3]
            )
            
            # If we need more recommendations, fill with popular unrated movies
            if len(recommended_movies) < n_recommendations:
                remaining = n_recommendations - len(recommended_movies)
                recommended_ids = {str(movie['id']) for movie in recommended_movies}
                remaining_unrated = unrated_movies[~unrated_movies['id'].astype(str).isin(recommended_ids)]
                popular_unrated = remaining_unrated.nlargest(remaining, ['vote_average', 'popularity'])
                recommended_movies.extend(popular_unrated.to_dict('records'))
            
            result_df = pd.DataFrame(recommended_movies[:n_recommendations])
            return result_df if not result_df.empty else self._get_fallback_recommendations(n_recommendations)
            
        except Exception as e:
            logging.error(f"Error in get_recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """Get fallback recommendations when main algorithm fails."""
        try:
            return self.df.nlargest(n_recommendations, ['vote_average', 'popularity'])
        except Exception:
            return self.df.head(n_recommendations)
    
    def _get_relaxed_recommendations(self, unrated_movies: pd.DataFrame, 
                                   top_genres: List[str], n_recommendations: int) -> pd.DataFrame:
        """Get recommendations with relaxed genre matching."""
        try:
            relaxed_scores = []
            top_5_genres = set(top_genres[:5])
            
            for _, movie in unrated_movies.iterrows():
                try:
                    genre_str = str(movie['genre'])
                    if genre_str and genre_str != 'nan':
                        movie_genres = set([g.strip() for g in genre_str.split(',') if g.strip()])
                        if movie_genres.intersection(top_5_genres):
                            # Simple scoring for relaxed matching
                            quality = float(movie.get('vote_average', 0)) / 10.0
                            popularity = min(float(movie.get('popularity', 0)) / 100.0, 1.0)
                            basic_score = quality * 2.0 + popularity
                            relaxed_scores.append((movie.to_dict(), basic_score))
                except Exception:
                    continue
            
            if relaxed_scores:
                relaxed_scores.sort(key=lambda x: x[1], reverse=True)
                recommended_movies = [movie for movie, score in relaxed_scores[:n_recommendations]]
                return pd.DataFrame(recommended_movies)
            else:
                return self._get_fallback_recommendations(n_recommendations)
                
        except Exception:
            return self._get_fallback_recommendations(n_recommendations)
    
    def _diversify_recommendations(self, movie_scores: List, n_recommendations: int, 
                                 top_genres: List[str]) -> List[Dict]:
        """Diversify recommendations to avoid too many movies from the same genre."""
        try:
            recommendations = []
            genre_counts = {genre: 0 for genre in top_genres}
            max_per_genre = max(2, n_recommendations // len(top_genres)) if top_genres else n_recommendations
            
            for movie_dict, score in movie_scores:
                if len(recommendations) >= n_recommendations:
                    break
                    
                # Check genre diversity
                movie_genres = [g.strip() for g in str(movie_dict.get('genre', '')).split(',')]
                main_genre = None
                
                for genre in top_genres:
                    if genre in movie_genres:
                        main_genre = genre
                        break
                
                # Add movie if it doesn't exceed genre limit or no main genre found
                if not main_genre or genre_counts.get(main_genre, 0) < max_per_genre:
                    recommendations.append(movie_dict)
                    if main_genre:
                        genre_counts[main_genre] += 1
            
            # Fill remaining slots with any remaining movies
            remaining_slots = n_recommendations - len(recommendations)
            if remaining_slots > 0:
                added_ids = {str(movie['id']) for movie in recommendations}
                for movie_dict, score in movie_scores:
                    if len(recommendations) >= n_recommendations:
                        break
                    if str(movie_dict['id']) not in added_ids:
                        recommendations.append(movie_dict)
                        added_ids.add(str(movie_dict['id']))
            
            return recommendations
            
        except Exception:
            # Fallback to simple selection
            return [movie for movie, score in movie_scores[:n_recommendations]]
    
    def get_user_stats(self) -> Dict[str, any]:
        """
        Get statistics about user's rating patterns.
        
        Returns:
            Dictionary containing user statistics
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
            movie_row = self.df[self.df['id'].astype(str) == movie_id]
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