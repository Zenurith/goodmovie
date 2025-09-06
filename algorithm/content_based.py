import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging
import re
import json
from collections import Counter, defaultdict


class ContentBasedRecommender:
    """
    Content-based recommendation system optimized for cold start scenarios.
    
    This system focuses on movie content features (genres, ratings, popularity, language)
    to provide recommendations even when users have very few ratings or are completely new.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the content-based recommender.
        
        Args:
            df: DataFrame containing movie data with columns:
                - id: movie ID
                - title: movie title
                - genre: comma-separated genres
                - vote_average: average rating
                - vote_count: number of votes
                - popularity: popularity score
                - original_language: language code
                - overview: movie description
        """
        self.df = df.copy()
        self.user_ratings = {}
        self.movie_features = None
        self.tfidf_matrix = None
        self.content_similarity_matrix = None
        self.scaler = StandardScaler()
        self.genre_list = self._extract_all_genres()
        self.user_profile = None
        
        # Precompute movie features for better performance
        self._build_movie_features()
        
    def update_user_ratings(self, user_ratings: Dict[str, int]) -> None:
        """Update user ratings and rebuild user profile."""
        self.user_ratings = user_ratings.copy()
        self.user_profile = None  # Reset profile to rebuild
        
    def _extract_all_genres(self) -> List[str]:
        """Extract all unique genres from the dataset."""
        all_genres = set()
        for genres_str in self.df['genre'].fillna(''):
            genres = [g.strip() for g in str(genres_str).split(',') if g.strip()]
            all_genres.update(genres)
        return sorted(list(all_genres))
        
    def _build_movie_features(self) -> None:
        """
        Build comprehensive feature matrix for all movies.
        Features include: genres, rating buckets, popularity buckets, language, text features.
        """
        features = []
        
        for _, movie in self.df.iterrows():
            movie_features = self._extract_movie_features(movie)
            features.append(movie_features)
            
        # Convert to DataFrame for easier manipulation
        self.movie_features = pd.DataFrame(features, index=self.df['id'])
        
        # Build TF-IDF matrix for text-based features
        self._build_tfidf_matrix()
        
        # Build content similarity matrix
        self._build_content_similarity_matrix()
        
    def _extract_movie_features(self, movie: pd.Series) -> Dict[str, float]:
        """
        Extract comprehensive features from a single movie.
        
        Returns:
            Dictionary of feature_name -> feature_value
        """
        features = {}
        
        # 1. Genre features (binary encoding)
        movie_genres = self._parse_genres(str(movie.get('genre', '')))
        for genre in self.genre_list:
            features[f'genre_{genre}'] = 1.0 if genre in movie_genres else 0.0
            
        # 2. Rating features
        vote_avg = float(movie.get('vote_average', 0))
        features['rating_high'] = 1.0 if vote_avg >= 8.0 else 0.0
        features['rating_good'] = 1.0 if 7.0 <= vote_avg < 8.0 else 0.0
        features['rating_average'] = 1.0 if 6.0 <= vote_avg < 7.0 else 0.0
        features['rating_below_average'] = 1.0 if vote_avg < 6.0 else 0.0
        features['vote_average_normalized'] = vote_avg / 10.0
        
        # 3. Popularity features
        popularity = float(movie.get('popularity', 0))
        features['popularity_very_high'] = 1.0 if popularity >= 80 else 0.0
        features['popularity_high'] = 1.0 if 40 <= popularity < 80 else 0.0
        features['popularity_medium'] = 1.0 if 10 <= popularity < 40 else 0.0
        features['popularity_low'] = 1.0 if popularity < 10 else 0.0
        features['popularity_normalized'] = min(popularity / 100.0, 1.0)
        
        # 4. Vote count features (reliability indicator)
        vote_count = float(movie.get('vote_count', 0))
        features['reliable_rating'] = 1.0 if vote_count >= 1000 else 0.0
        features['some_votes'] = 1.0 if 100 <= vote_count < 1000 else 0.0
        features['few_votes'] = 1.0 if vote_count < 100 else 0.0
        features['vote_count_log'] = np.log1p(vote_count) / 10.0  # Normalized log
        
        # 5. Language features
        language = str(movie.get('original_language', ''))
        # Focus on major languages for cold start
        major_languages = ['en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'hi', 'ru']
        for lang in major_languages:
            features[f'lang_{lang}'] = 1.0 if language == lang else 0.0
        features['lang_english'] = 1.0 if language == 'en' else 0.0
        features['lang_international'] = 1.0 if language != 'en' else 0.0
        
        # 6. Content quality score (composite feature)
        quality_score = (vote_avg / 10.0) * 0.6 + min(popularity / 50.0, 1.0) * 0.3 + min(vote_count / 2000.0, 1.0) * 0.1
        features['quality_score'] = quality_score
        
        return features
        
    def _parse_genres(self, genre_str: str) -> Set[str]:
        """Parse comma-separated genre string into set of genres."""
        if not genre_str or genre_str == 'nan':
            return set()
        return {g.strip() for g in genre_str.split(',') if g.strip()}
        
    def _build_tfidf_matrix(self) -> None:
        """Build TF-IDF matrix from movie text features (genres + overview)."""
        text_features = []
        
        for _, movie in self.df.iterrows():
            # Combine genre and overview for text analysis
            genres = str(movie.get('genre', '')).replace(',', ' ')
            overview = str(movie.get('overview', ''))
            
            # Clean and combine text
            text_content = f"{genres} {overview}"
            text_content = re.sub(r'[^a-zA-Z\s]', ' ', text_content)
            text_content = ' '.join(text_content.split())  # Normalize whitespace
            
            text_features.append(text_content)
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = vectorizer.fit_transform(text_features)
        
    def _build_content_similarity_matrix(self) -> None:
        """Build similarity matrix combining numerical and text features."""
        # Normalize numerical features
        numerical_features = self.movie_features.values
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine numerical and text features
        # Weight: 70% numerical features, 30% text features
        text_features_dense = self.tfidf_matrix.toarray()
        
        # Ensure same number of samples
        min_samples = min(numerical_features_scaled.shape[0], text_features_dense.shape[0])
        numerical_features_scaled = numerical_features_scaled[:min_samples]
        text_features_dense = text_features_dense[:min_samples]
        
        # Combine features
        combined_features = np.hstack([
            numerical_features_scaled * 0.7,
            text_features_dense * 0.3
        ])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(combined_features)
        
    def build_user_profile(self) -> Dict[str, float]:
        """
        Build user preference profile from their ratings.
        
        For cold start (few ratings), this creates a robust profile that can
        generalize well to unseen movies.
        """
        if not self.user_ratings:
            return self._get_default_profile()
            
        if self.user_profile is not None:
            return self.user_profile
            
        profile = defaultdict(float)
        total_weight = 0
        
        # Analyze rated movies
        for movie_id, rating in self.user_ratings.items():
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            
            if movie_row.empty:
                continue
                
            movie = movie_row.iloc[0]
            weight = self._calculate_rating_weight(rating)
            
            if weight <= 0:
                continue
                
            # Extract features for this movie
            movie_features = self._extract_movie_features(movie)
            
            # Update profile with weighted features
            for feature_name, feature_value in movie_features.items():
                profile[feature_name] += weight * feature_value
                
            total_weight += weight
            
        # Normalize profile
        if total_weight > 0:
            for feature_name in profile:
                profile[feature_name] /= total_weight
                
        # Convert to regular dict and cache
        self.user_profile = dict(profile)
        
        # Add derived preferences for cold start robustness
        self._enhance_user_profile()
        
        return self.user_profile
        
    def _calculate_rating_weight(self, rating: int) -> float:
        """
        Calculate weight for a rating in profile building.
        
        Higher ratings get more weight, but even medium ratings contribute.
        This helps in cold start scenarios where users might have few high ratings.
        """
        if rating >= 8:
            return 1.0
        elif rating >= 7:
            return 0.8
        elif rating >= 6:
            return 0.5
        elif rating >= 5:
            return 0.2
        else:
            return 0.0  # Negative ratings don't contribute positively
            
    def _enhance_user_profile(self) -> None:
        """Enhance user profile with derived features for better cold start performance."""
        if not self.user_profile:
            return
            
        # Calculate genre diversity preference
        genre_features = {k: v for k, v in self.user_profile.items() if k.startswith('genre_')}
        if genre_features:
            genre_count = sum(1 for v in genre_features.values() if v > 0.1)
            self.user_profile['genre_diversity'] = min(genre_count / 5.0, 1.0)
            
        # Calculate quality preference
        quality_indicators = ['rating_high', 'rating_good', 'reliable_rating']
        quality_score = sum(self.user_profile.get(indicator, 0) for indicator in quality_indicators)
        self.user_profile['quality_preference'] = quality_score / len(quality_indicators)
        
        # Calculate mainstream vs niche preference
        mainstream_score = self.user_profile.get('popularity_high', 0) + self.user_profile.get('lang_english', 0)
        niche_score = self.user_profile.get('popularity_low', 0) + self.user_profile.get('lang_international', 0)
        self.user_profile['mainstream_preference'] = mainstream_score / 2.0
        self.user_profile['niche_preference'] = niche_score / 2.0
        
    def _get_default_profile(self) -> Dict[str, float]:
        """Get default profile for completely new users (cold start)."""
        return {
            'quality_preference': 0.8,  # Prefer good movies
            'mainstream_preference': 0.6,  # Slightly prefer mainstream
            'genre_diversity': 0.5,  # Moderate diversity
            'rating_high': 0.3,
            'rating_good': 0.4,
            'popularity_high': 0.3,
            'popularity_medium': 0.4,
            'lang_english': 0.7,  # Default to English for cold start
            'reliable_rating': 0.6,  # Prefer movies with many votes
        }
        
    def get_recommendations(self, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get content-based recommendations optimized for cold start scenarios.
        
        Args:
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies and their content scores
        """
        try:
            # Build user profile
            user_profile = self.build_user_profile()
            
            if not user_profile:
                return self._get_popular_diverse_recommendations(n_recommendations)
                
            # Get candidate movies (exclude already rated)
            rated_movie_ids = {str(movie_id) for movie_id in self.user_ratings.keys()}
            candidate_movies = self.df[~self.df['id'].astype(str).isin(rated_movie_ids)].copy()
            
            if candidate_movies.empty:
                return pd.DataFrame()
                
            # Score all candidate movies
            movie_scores = []
            
            for _, movie in candidate_movies.iterrows():
                score = self._calculate_movie_score(movie, user_profile)
                movie_scores.append({
                    'movie_id': movie['id'],
                    'content_score': score,
                    'movie_data': movie
                })
                
            # Sort by score and apply quality filters
            movie_scores.sort(key=lambda x: x['content_score'], reverse=True)
            
            # Filter and diversify results
            recommendations = self._filter_and_diversify_recommendations(
                movie_scores, n_recommendations
            )
            
            # Convert to DataFrame
            result_movies = []
            for rec in recommendations:
                movie_dict = rec['movie_data'].to_dict()
                movie_dict['content_score'] = rec['content_score']
                result_movies.append(movie_dict)
                
            return pd.DataFrame(result_movies)
            
        except Exception as e:
            logging.error(f"Error in content-based recommendations: {e}")
            return self._get_popular_diverse_recommendations(n_recommendations)
            
    def _calculate_movie_score(self, movie: pd.Series, user_profile: Dict[str, float]) -> float:
        """Calculate content-based score for a movie given user profile."""
        try:
            movie_features = self._extract_movie_features(movie)
            
            # Calculate similarity score
            similarity_score = 0.0
            
            for feature_name, user_pref in user_profile.items():
                if feature_name in movie_features:
                    movie_feature_value = movie_features[feature_name]
                    similarity_score += user_pref * movie_feature_value
                    
            # Apply cold start bonuses
            score = similarity_score
            
            # Quality bonus for cold start robustness
            vote_avg = float(movie.get('vote_average', 0))
            vote_count = float(movie.get('vote_count', 0))
            
            if vote_avg >= 7.5 and vote_count >= 500:
                score += 0.3  # High quality bonus
            elif vote_avg >= 7.0 and vote_count >= 200:
                score += 0.2  # Good quality bonus
            elif vote_avg >= 6.5 and vote_count >= 100:
                score += 0.1  # Decent quality bonus
                
            # Popularity balance for cold start
            popularity = float(movie.get('popularity', 0))
            if 20 <= popularity <= 60:  # Sweet spot for cold start
                score += 0.1
                
            # Recency bonus (newer movies for cold start)
            release_date = str(movie.get('release_date', ''))
            if release_date and len(release_date) >= 4:
                try:
                    release_year = int(release_date[:4])
                    current_year = 2025
                    if current_year - release_year <= 5:  # Recent movies
                        score += 0.05
                except ValueError:
                    pass
                    
            return max(0, score)
            
        except Exception as e:
            logging.error(f"Error calculating movie score: {e}")
            return 0.0
            
    def _filter_and_diversify_recommendations(self, movie_scores: List[Dict], 
                                            n_recommendations: int) -> List[Dict]:
        """
        Filter recommendations and ensure diversity for cold start scenarios.
        """
        filtered_recommendations = []
        seen_genres = set()
        seen_languages = set()
        
        # Quality threshold - lower for cold start
        min_score_threshold = 0.1 if len(self.user_ratings) <= 3 else 0.2
        
        for movie_data in movie_scores:
            if len(filtered_recommendations) >= n_recommendations:
                break
                
            movie = movie_data['movie_data']
            score = movie_data['content_score']
            
            if score < min_score_threshold:
                continue
                
            # Basic quality filter
            vote_avg = float(movie.get('vote_average', 0))
            vote_count = float(movie.get('vote_count', 0))
            
            if vote_avg < 5.5:  # Very low threshold for cold start
                continue
                
            # Diversity considerations for cold start
            movie_genres = self._parse_genres(str(movie.get('genre', '')))
            movie_language = str(movie.get('original_language', ''))
            
            # For cold start, ensure some diversity but don't be too restrictive
            genre_overlap = len(movie_genres.intersection(seen_genres))
            
            # Allow some genre repetition but prefer diversity
            if len(filtered_recommendations) < n_recommendations // 2:
                # First half: accept any good movie
                filtered_recommendations.append(movie_data)
                seen_genres.update(movie_genres)
                seen_languages.add(movie_language)
            else:
                # Second half: prefer diverse options
                if genre_overlap <= 1 or len(seen_genres) < 3:
                    filtered_recommendations.append(movie_data)
                    seen_genres.update(movie_genres)
                    seen_languages.add(movie_language)
                    
        return filtered_recommendations
        
    def _get_popular_diverse_recommendations(self, n_recommendations: int) -> pd.DataFrame:
        """
        Fallback recommendations for completely cold start scenarios.
        Returns popular, high-quality, diverse movies.
        """
        try:
            # Get high-quality movies
            quality_movies = self.df[
                (self.df['vote_average'] >= 7.0) &
                (self.df['vote_count'] >= 500)
            ].copy()
            
            if quality_movies.empty:
                # Fallback to any decent movies
                quality_movies = self.df[
                    (self.df['vote_average'] >= 6.0) &
                    (self.df['vote_count'] >= 100)
                ].copy()
                
            if quality_movies.empty:
                return self.df.head(n_recommendations)
                
            # Sort by combined score of rating and popularity
            quality_movies['fallback_score'] = (
                quality_movies['vote_average'] * 0.6 +
                np.log1p(quality_movies['popularity']) * 0.4
            )
            
            # Get diverse selection
            recommendations = []
            seen_genres = set()
            
            for _, movie in quality_movies.nlargest(n_recommendations * 3, 'fallback_score').iterrows():
                if len(recommendations) >= n_recommendations:
                    break
                    
                movie_genres = self._parse_genres(str(movie.get('genre', '')))
                
                # Ensure genre diversity
                if not movie_genres.intersection(seen_genres) or len(seen_genres) < 3:
                    recommendations.append(movie)
                    seen_genres.update(movie_genres)
                elif len(recommendations) < n_recommendations // 2:
                    # Fill first half even with some repetition
                    recommendations.append(movie)
                    seen_genres.update(movie_genres)
                    
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            logging.error(f"Error in fallback recommendations: {e}")
            return self.df.head(n_recommendations)
            
    def get_similar_movies(self, movie_id: str, n_similar: int = 5) -> pd.DataFrame:
        """
        Get movies similar to a given movie using content features.
        Useful for cold start scenarios when user likes a specific movie.
        """
        try:
            # Find movie index
            movie_indices = self.df[self.df['id'].astype(str) == str(movie_id)].index
            
            if movie_indices.empty:
                return pd.DataFrame()
                
            movie_idx = movie_indices[0]
            
            # Get similarities from precomputed matrix
            if (self.content_similarity_matrix is not None and 
                movie_idx < len(self.content_similarity_matrix)):
                
                similarities = self.content_similarity_matrix[movie_idx]
                
                # Get most similar movies (excluding the movie itself)
                similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
                
                similar_movies = self.df.iloc[similar_indices].copy()
                similar_movies['similarity_score'] = similarities[similar_indices]
                
                return similar_movies
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Error finding similar movies: {e}")
            return pd.DataFrame()
            
    def explain_recommendation(self, movie_id: str) -> Dict[str, any]:
        """
        Provide explanation for why a movie was recommended.
        Helpful for cold start scenarios to build user trust.
        """
        try:
            movie_row = self.df[self.df['id'].astype(str) == str(movie_id)]
            
            if movie_row.empty:
                return {"error": "Movie not found"}
                
            movie = movie_row.iloc[0]
            user_profile = self.build_user_profile()
            
            explanation = {
                "movie_title": movie.get('title', 'Unknown'),
                "reasons": [],
                "movie_features": {},
                "match_score": 0.0
            }
            
            if not user_profile:
                explanation["reasons"] = [
                    "High-quality movie recommended for new users",
                    f"Rating: {movie.get('vote_average', 0)}/10",
                    f"Popular movie with {movie.get('vote_count', 0)} votes"
                ]
                return explanation
                
            # Analyze feature matches
            movie_features = self._extract_movie_features(movie)
            match_score = self._calculate_movie_score(movie, user_profile)
            
            explanation["match_score"] = round(match_score, 2)
            
            # Genre matches
            movie_genres = self._parse_genres(str(movie.get('genre', '')))
            preferred_genres = [
                genre.replace('genre_', '') for genre, score in user_profile.items()
                if genre.startswith('genre_') and score > 0.3
            ]
            
            genre_matches = movie_genres.intersection(set(preferred_genres))
            if genre_matches:
                explanation["reasons"].append(
                    f"Matches your preferred genres: {', '.join(genre_matches)}"
                )
                
            # Rating preference
            vote_avg = float(movie.get('vote_average', 0))
            if user_profile.get('quality_preference', 0) > 0.6 and vote_avg >= 7.5:
                explanation["reasons"].append(
                    f"High-quality movie (rating: {vote_avg}/10) matching your preference for good films"
                )
                
            # Popularity preference
            popularity = float(movie.get('popularity', 0))
            if user_profile.get('mainstream_preference', 0) > 0.6 and popularity > 50:
                explanation["reasons"].append("Popular movie matching your mainstream preferences")
            elif user_profile.get('niche_preference', 0) > 0.6 and popularity < 20:
                explanation["reasons"].append("Hidden gem matching your taste for unique films")
                
            # Language preference
            movie_language = str(movie.get('original_language', ''))
            if user_profile.get('lang_international', 0) > 0.5 and movie_language != 'en':
                explanation["reasons"].append(f"International film ({movie_language}) matching your diverse taste")
                
            # Store key movie features
            explanation["movie_features"] = {
                "genres": list(movie_genres),
                "rating": vote_avg,
                "popularity": popularity,
                "language": movie_language,
                "vote_count": int(movie.get('vote_count', 0))
            }
            
            if not explanation["reasons"]:
                explanation["reasons"] = ["Recommended based on overall content similarity to your preferences"]
                
            return explanation
            
        except Exception as e:
            logging.error(f"Error explaining recommendation: {e}")
            return {"error": "Could not generate explanation"}
            
    def get_user_preferences_summary(self) -> Dict[str, any]:
        """
        Get summary of user preferences learned from their ratings.
        Useful for understanding cold start user profiling.
        """
        try:
            if not self.user_ratings:
                return {
                    "status": "No ratings yet",
                    "recommendation": "Rate a few movies to get personalized recommendations"
                }
                
            user_profile = self.build_user_profile()
            
            # Extract top preferences
            genre_prefs = {
                k.replace('genre_', ''): v for k, v in user_profile.items() 
                if k.startswith('genre_') and v > 0.1
            }
            
            top_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            summary = {
                "total_ratings": len(self.user_ratings),
                "avg_rating": round(np.mean(list(self.user_ratings.values())), 1),
                "preferred_genres": [genre for genre, _ in top_genres],
                "quality_preference": round(user_profile.get('quality_preference', 0), 2),
                "diversity_preference": round(user_profile.get('genre_diversity', 0), 2),
                "mainstream_vs_niche": {
                    "mainstream": round(user_profile.get('mainstream_preference', 0), 2),
                    "niche": round(user_profile.get('niche_preference', 0), 2)
                },
                "confidence_level": "High" if len(self.user_ratings) >= 10 else 
                                 "Medium" if len(self.user_ratings) >= 5 else "Low (Cold Start)"
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating user preferences summary: {e}")
            return {"error": "Could not generate summary"}


def create_content_based_recommender(df: pd.DataFrame) -> ContentBasedRecommender:
    """Factory function to create a content-based recommender instance."""
    return ContentBasedRecommender(df)


def get_content_based_recommendations(df: pd.DataFrame, user_ratings: Dict[str, int], 
                                    n_recommendations: int = 10) -> pd.DataFrame:
    """
    Get content-based recommendations optimized for cold start scenarios.
    
    Args:
        df: Movie dataset DataFrame
        user_ratings: Dictionary of movie_id -> rating
        n_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame with recommended movies and content scores
    """
    recommender = create_content_based_recommender(df)
    recommender.update_user_ratings(user_ratings)
    return recommender.get_recommendations(n_recommendations)