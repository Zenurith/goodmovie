import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st

class SimpleMovieRecommender:
    def __init__(self, dataset_path):
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.load_data(dataset_path)
    
    def load_data(self, dataset_path):
        """Load and prepare the movie dataset"""
        try:
            self.movies_df = pd.read_csv(dataset_path)
            self.prepare_features()
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare features for similarity calculation"""
        # Combine text features
        self.movies_df['combined_features'] = (
            self.movies_df['genre'].fillna('') + ' ' + 
            self.movies_df['overview'].fillna('') + ' ' + 
            self.movies_df['original_language'].fillna('')
        )
        
        # Use TF-IDF for text features
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        text_features = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Normalize numerical features
        numerical_cols = ['popularity', 'vote_average', 'vote_count']
        numerical_features = self.movies_df[numerical_cols].fillna(0)
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)
        
        # Combine features
        self.feature_matrix = np.hstack([
            text_features.toarray(),
            numerical_features_scaled
        ])
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def get_movie_index(self, movie_title):
        """Get index of movie by title"""
        matches = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
        if len(matches) > 0:
            return matches.index[0]
        
        # Try partial match
        partial_matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
        if len(partial_matches) > 0:
            return partial_matches.index[0]
        
        return None
    
    def get_recommendations(self, movie_title, n_recommendations=8):
        """Get movie recommendations based on similarity"""
        movie_idx = self.get_movie_index(movie_title)
        
        if movie_idx is None:
            return []
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity (excluding the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Prepare recommendations
        recommendations = []
        for idx, (movie_idx, score) in enumerate(zip(movie_indices, sim_scores)):
            movie = self.movies_df.iloc[movie_idx]
            recommendations.append({
                'id': movie['id'],
                'title': movie['title'],
                'rating': movie['vote_average'],
                'votes': movie['vote_count'],
                'genres': movie['genre'],
                'similarity': score[1],
                'year': movie['release_date'][:4] if pd.notna(movie['release_date']) else None
            })
        
        return recommendations