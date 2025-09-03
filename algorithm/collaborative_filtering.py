"""
Collaborative Filtering Algorithms for Movie Recommendation System

This module implements various collaborative filtering techniques:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering 
- Matrix Factorization (SVD, NMF)
- Hybrid Approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class UserBasedCollaborativeFiltering:
    """User-Based Collaborative Filtering Implementation"""
    
    def __init__(self, similarity_threshold=0.1, k_neighbors=50):
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.user_similarity_matrix = None
        self.user_item_matrix = None
        self.user_means = None
        
    def fit(self, user_interactions):
        """Train the user-based collaborative filtering model"""
        print("Building user-item matrix...")
        
        # Create user-item matrix
        self.user_item_matrix = user_interactions.pivot(index='user_id', 
                                                        columns='movie_id', 
                                                        values='rating').fillna(0)
        
        # Calculate user means (for mean-centered ratings)
        self.user_means = self.user_item_matrix.replace(0, np.nan).mean(axis=1)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print("Computing user similarity matrix...")
        
        # Compute user similarity using cosine similarity
        user_matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)
        self.user_similarity_matrix = cosine_similarity(user_matrix_centered)
        
        # Set diagonal to 0 (user shouldn't be most similar to themselves)
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        print(f"User similarity matrix shape: {self.user_similarity_matrix.shape}")
        print(f"Average similarity: {self.user_similarity_matrix.mean():.3f}")
        
    def predict(self, user_id, movie_id, top_k=None):
        """Predict rating for a specific user-movie pair"""
        if top_k is None:
            top_k = self.k_neighbors
            
        # Get user similarities
        user_similarities = self.user_similarity_matrix[user_id]
        
        # Find users who have rated this movie and are similar
        movie_raters = self.user_item_matrix[movie_id] > 0
        valid_similarities = user_similarities * movie_raters
        
        # Get top-k similar users
        top_similar_users = np.argsort(valid_similarities)[::-1][:top_k]
        top_similarities = valid_similarities[top_similar_users]
        
        # Filter by similarity threshold
        above_threshold = top_similarities > self.similarity_threshold
        top_similar_users = top_similar_users[above_threshold]
        top_similarities = top_similarities[above_threshold]
        
        if len(top_similar_users) == 0:
            return self.user_means[user_id] if user_id in self.user_means else 5.0
        
        # Calculate weighted average rating
        similar_ratings = self.user_item_matrix.iloc[top_similar_users, movie_id]
        similar_user_means = self.user_means.iloc[top_similar_users]
        
        # Mean-centered prediction
        numerator = np.sum(top_similarities * (similar_ratings - similar_user_means))
        denominator = np.sum(np.abs(top_similarities))
        
        if denominator == 0:
            return self.user_means[user_id] if user_id in self.user_means else 5.0
        
        prediction = self.user_means[user_id] + (numerator / denominator)
        return np.clip(prediction, 1, 10)
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Recommend movies for a specific user"""
        # Get movies the user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class ItemBasedCollaborativeFiltering:
    """Item-Based Collaborative Filtering Implementation"""
    
    def __init__(self, similarity_threshold=0.1, k_neighbors=50):
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.item_means = None
        
    def fit(self, user_interactions):
        """Train the item-based collaborative filtering model"""
        print("Building user-item matrix...")
        
        # Create user-item matrix
        self.user_item_matrix = user_interactions.pivot(index='user_id', 
                                                        columns='movie_id', 
                                                        values='rating').fillna(0)
        
        # Calculate item means
        self.item_means = self.user_item_matrix.replace(0, np.nan).mean(axis=0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print("Computing item similarity matrix...")
        
        # Compute item similarity using cosine similarity
        item_matrix = self.user_item_matrix.T
        item_matrix_centered = item_matrix.sub(self.item_means, axis=0).fillna(0)
        self.item_similarity_matrix = cosine_similarity(item_matrix_centered)
        
        # Set diagonal to 0 (item shouldn't be most similar to itself)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print(f"Item similarity matrix shape: {self.item_similarity_matrix.shape}")
        print(f"Average similarity: {self.item_similarity_matrix.mean():.3f}")
        
    def predict(self, user_id, movie_id, top_k=None):
        """Predict rating for a specific user-movie pair"""
        if top_k is None:
            top_k = self.k_neighbors
            
        # Get item similarities for the target movie
        item_similarities = self.item_similarity_matrix[movie_id]
        
        # Find items that the user has rated and are similar to target movie
        user_ratings = self.user_item_matrix.iloc[user_id]
        rated_items = user_ratings > 0
        valid_similarities = item_similarities * rated_items
        
        # Get top-k similar items
        top_similar_items = np.argsort(valid_similarities)[::-1][:top_k]
        top_similarities = valid_similarities[top_similar_items]
        
        # Filter by similarity threshold
        above_threshold = top_similarities > self.similarity_threshold
        top_similar_items = top_similar_items[above_threshold]
        top_similarities = top_similarities[above_threshold]
        
        if len(top_similar_items) == 0:
            return self.item_means[movie_id] if movie_id in self.item_means else 5.0
        
        # Calculate weighted average rating
        user_ratings_similar = user_ratings.iloc[top_similar_items]
        similar_item_means = self.item_means.iloc[top_similar_items]
        
        # Mean-centered prediction
        numerator = np.sum(top_similarities * (user_ratings_similar - similar_item_means))
        denominator = np.sum(np.abs(top_similarities))
        
        if denominator == 0:
            return self.item_means[movie_id] if movie_id in self.item_means else 5.0
        
        prediction = self.item_means[movie_id] + (numerator / denominator)
        return np.clip(prediction, 1, 10)
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Recommend movies for a specific user"""
        # Get movies the user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get movies similar to a given movie"""
        similarities = self.item_similarity_matrix[movie_id]
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
        similar_scores = similarities[similar_indices]
        
        return list(zip(similar_indices, similar_scores))


class MatrixFactorizationCF:
    """Matrix Factorization Collaborative Filtering (SVD/NMF)"""
    
    def __init__(self, n_factors=50, method='svd'):
        self.n_factors = n_factors
        self.method = method
        self.model = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, user_interactions):
        """Train matrix factorization model"""
        print(f"Training {self.method.upper()} Matrix Factorization...")
        
        # Create user-item matrix
        self.user_item_matrix = user_interactions.pivot(index='user_id', 
                                                        columns='movie_id', 
                                                        values='rating').fillna(0)
        
        self.global_mean = user_interactions['rating'].mean()
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Number of factors: {self.n_factors}")
        
        if self.method == 'svd':
            # Truncated SVD for collaborative filtering
            self.model = TruncatedSVD(n_components=self.n_factors, random_state=42)
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.item_factors = self.model.components_
            
        elif self.method == 'nmf':
            # Non-negative Matrix Factorization
            self.model = NMF(n_components=self.n_factors, random_state=42, max_iter=200)
            self.user_factors = self.model.fit_transform(self.user_item_matrix)
            self.item_factors = self.model.components_
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
        
        if hasattr(self.model, 'explained_variance_ratio_'):
            print(f"Explained variance: {self.model.explained_variance_ratio_.sum():.3f}")
        elif hasattr(self.model, 'reconstruction_err_'):
            print(f"Reconstruction error: {self.model.reconstruction_err_:.3f}")
    
    def predict(self, user_id, movie_id):
        """Predict rating for user-movie pair"""
        if user_id >= len(self.user_factors) or movie_id >= self.item_factors.shape[1]:
            return self.global_mean
        
        # Dot product of user and item factors
        prediction = np.dot(self.user_factors[user_id], self.item_factors[:, movie_id])
        
        return np.clip(prediction, 1, 10)
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Recommend movies for a user"""
        if user_id >= len(self.user_factors):
            return []
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            if movie_id < self.item_factors.shape[1]:
                predicted_rating = self.predict(user_id, movie_id)
                predictions.append((movie_id, predicted_rating))
        
        # Sort and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class HybridCollaborativeFiltering:
    """Hybrid Collaborative Filtering combining multiple approaches"""
    
    def __init__(self, user_weight=0.3, item_weight=0.3, matrix_weight=0.4):
        self.user_weight = user_weight
        self.item_weight = item_weight  
        self.matrix_weight = matrix_weight
        self.user_cf = None
        self.item_cf = None
        self.matrix_cf = None
        self.global_mean = None
        
    def fit(self, user_interactions):
        """Train all component models"""
        print("Training Hybrid Collaborative Filtering System...")
        print(f"Weights: User={self.user_weight}, Item={self.item_weight}, Matrix={self.matrix_weight}")
        
        self.global_mean = user_interactions['rating'].mean()
        
        # Train user-based CF
        print("\n1. Training User-Based CF...")
        self.user_cf = UserBasedCollaborativeFiltering(similarity_threshold=0.05, k_neighbors=20)
        self.user_cf.fit(user_interactions)
        
        # Train item-based CF
        print("\n2. Training Item-Based CF...")
        self.item_cf = ItemBasedCollaborativeFiltering(similarity_threshold=0.05, k_neighbors=20)
        self.item_cf.fit(user_interactions)
        
        # Train matrix factorization
        print("\n3. Training Matrix Factorization...")
        self.matrix_cf = MatrixFactorizationCF(n_factors=25, method='svd')
        self.matrix_cf.fit(user_interactions)
        
        print("\n✅ All component models trained!")
    
    def predict(self, user_id, movie_id):
        """Hybrid prediction combining all methods"""
        predictions = []
        weights = []
        
        # User-based prediction
        try:
            user_pred = self.user_cf.predict(user_id, movie_id)
            predictions.append(user_pred)
            weights.append(self.user_weight)
        except:
            pass
        
        # Item-based prediction
        try:
            item_pred = self.item_cf.predict(user_id, movie_id)
            predictions.append(item_pred)
            weights.append(self.item_weight)
        except:
            pass
        
        # Matrix factorization prediction
        try:
            matrix_pred = self.matrix_cf.predict(user_id, movie_id)
            predictions.append(matrix_pred)
            weights.append(self.matrix_weight)
        except:
            pass
        
        # Weighted average
        if predictions:
            weights = np.array(weights[:len(predictions)])
            weights = weights / weights.sum()  # Normalize weights
            hybrid_prediction = np.average(predictions, weights=weights)
            return np.clip(hybrid_prediction, 1, 10)
        
        return self.global_mean
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Hybrid recommendations"""
        # Get recommendations from all methods
        user_recs = []
        item_recs = []
        matrix_recs = []
        
        try:
            user_recs = self.user_cf.recommend_movies(user_id, n_recommendations * 2)
        except:
            pass
        
        try:
            item_recs = self.item_cf.recommend_movies(user_id, n_recommendations * 2)
        except:
            pass
        
        try:
            matrix_recs = self.matrix_cf.recommend_movies(user_id, n_recommendations * 2)
        except:
            pass
        
        # Combine and score all recommended movies
        all_movies = set()
        for recs in [user_recs, item_recs, matrix_recs]:
            all_movies.update([movie_id for movie_id, _ in recs])
        
        # Calculate hybrid scores for all candidate movies
        hybrid_scores = []
        for movie_id in all_movies:
            hybrid_score = self.predict(user_id, movie_id)
            hybrid_scores.append((movie_id, hybrid_score))
        
        # Sort by hybrid score and return top N
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:n_recommendations]


class CollaborativeFilteringRecommender:
    """Main interface for collaborative filtering recommendations"""
    
    def __init__(self):
        self.models = {}
        self.movies_df = None
        self.user_interactions = None
        
    def load_data(self):
        """Load movie and interaction data"""
        try:
            self.movies_df = pd.read_csv('movies_clean.csv')
            print(f"✅ Movies loaded: {len(self.movies_df):,}")
            
            # Try to load existing user interactions
            try:
                self.user_interactions = pd.read_csv('data/user_interactions.csv')
                print(f"✅ User interactions loaded: {len(self.user_interactions):,}")
            except FileNotFoundError:
                print("Creating simulated user interactions...")
                self.create_simulated_interactions()
                
            return True
        except FileNotFoundError as e:
            print(f"❌ Data file not found: {e}")
            return False
    
    def create_simulated_interactions(self):
        """Create realistic user-movie interactions if not available"""
        np.random.seed(42)
        n_users = 1000
        n_interactions = 50000
        
        # Generate user interactions with realistic patterns
        user_ids = np.random.randint(0, n_users, n_interactions)
        movie_indices = np.random.choice(len(self.movies_df), n_interactions)
        
        # Create more realistic rating distribution
        ratings = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_interactions, 
                                   p=[0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.08, 0.02])
        
        self.user_interactions = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_indices,
            'rating': ratings
        })
        
        # Save for future use
        os.makedirs('data', exist_ok=True)
        self.user_interactions.to_csv('data/user_interactions.csv', index=False)
        print(f"✅ Simulated interactions created: {len(self.user_interactions):,}")
    
    def train_models(self):
        """Train all collaborative filtering models"""
        if self.user_interactions is None:
            print("❌ No user interaction data available")
            return False
        
        print("\n🔧 Training Collaborative Filtering Models...")
        print("=" * 50)
        
        # Train User-Based CF
        print("\n1. Training User-Based Collaborative Filtering...")
        self.models['user_based'] = UserBasedCollaborativeFiltering(
            similarity_threshold=0.05, k_neighbors=30
        )
        self.models['user_based'].fit(self.user_interactions)
        
        # Train Item-Based CF
        print("\n2. Training Item-Based Collaborative Filtering...")
        self.models['item_based'] = ItemBasedCollaborativeFiltering(
            similarity_threshold=0.05, k_neighbors=30
        )
        self.models['item_based'].fit(self.user_interactions)
        
        # Train Matrix Factorization (SVD)
        print("\n3. Training SVD Matrix Factorization...")
        self.models['svd'] = MatrixFactorizationCF(n_factors=30, method='svd')
        self.models['svd'].fit(self.user_interactions)
        
        # Train Matrix Factorization (NMF)
        print("\n4. Training NMF Matrix Factorization...")
        self.models['nmf'] = MatrixFactorizationCF(n_factors=30, method='nmf')
        self.models['nmf'].fit(self.user_interactions)
        
        # Train Hybrid Model
        print("\n5. Training Hybrid Collaborative Filtering...")
        self.models['hybrid'] = HybridCollaborativeFiltering(
            user_weight=0.3, item_weight=0.3, matrix_weight=0.4
        )
        self.models['hybrid'].fit(self.user_interactions)
        
        print("\n✅ All models trained successfully!")
        return True
    
    def get_recommendations(self, user_id, method='hybrid', n_recommendations=10):
        """Get movie recommendations for a user"""
        if method not in self.models:
            print(f"Method '{method}' not available. Available: {list(self.models.keys())}")
            return []
        
        model = self.models[method]
        recommendations = model.recommend_movies(user_id, n_recommendations)
        
        # Add movie details
        detailed_recs = []
        for movie_id, score in recommendations:
            if movie_id < len(self.movies_df):
                movie_info = self.movies_df.iloc[movie_id]
                detailed_recs.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info.get('genres_str', ''),
                    'rating': movie_info.get('vote_average', 0),
                    'year': movie_info.get('release_year', 'N/A'),
                    'predicted_score': score
                })
        
        return detailed_recs
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get movies similar to a given movie (using item-based CF)"""
        if 'item_based' in self.models:
            similar = self.models['item_based'].get_similar_movies(movie_id, n_similar)
            
            detailed_similar = []
            for sim_movie_id, similarity_score in similar:
                if sim_movie_id < len(self.movies_df):
                    movie_info = self.movies_df.iloc[sim_movie_id]
                    detailed_similar.append({
                        'movie_id': sim_movie_id,
                        'title': movie_info['title'],
                        'genres': movie_info.get('genres_str', ''),
                        'rating': movie_info.get('vote_average', 0),
                        'year': movie_info.get('release_year', 'N/A'),
                        'similarity_score': similarity_score
                    })
            
            return detailed_similar
        return []
    
    def evaluate_models(self, test_size=0.2):
        """Evaluate all models and return performance metrics"""
        if self.user_interactions is None:
            print("❌ No user interaction data available")
            return {}
        
        print("\n📊 Evaluating Models...")
        print("=" * 30)
        
        # Split data
        train_data, test_data = train_test_split(
            self.user_interactions, test_size=test_size, random_state=42
        )
        
        # Train models on training data
        eval_models = {}
        
        # User-based CF
        eval_models['User-Based'] = UserBasedCollaborativeFiltering(
            similarity_threshold=0.05, k_neighbors=20
        )
        eval_models['User-Based'].fit(train_data)
        
        # Item-based CF
        eval_models['Item-Based'] = ItemBasedCollaborativeFiltering(
            similarity_threshold=0.05, k_neighbors=20
        )
        eval_models['Item-Based'].fit(train_data)
        
        # SVD Matrix Factorization
        eval_models['SVD'] = MatrixFactorizationCF(n_factors=25, method='svd')
        eval_models['SVD'].fit(train_data)
        
        # Hybrid
        eval_models['Hybrid'] = HybridCollaborativeFiltering(
            user_weight=0.3, item_weight=0.3, matrix_weight=0.4
        )
        eval_models['Hybrid'].fit(train_data)
        
        # Evaluate each model
        results = {}
        for model_name, model in eval_models.items():
            print(f"\nEvaluating {model_name}...")
            
            predictions = []
            actuals = []
            
            # Sample subset for faster evaluation
            test_sample = test_data.sample(min(1000, len(test_data)), random_state=42)
            
            for _, row in test_sample.iterrows():
                user_id = row['user_id']
                movie_id = row['movie_id']
                actual_rating = row['rating']
                
                try:
                    predicted_rating = model.predict(user_id, movie_id)
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
                except:
                    continue
            
            if predictions:
                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mse)
                
                results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Predictions': len(predictions)
                }
                
                print(f"   RMSE: {rmse:.3f}")
                print(f"   MAE: {mae:.3f}")
                print(f"   Predictions: {len(predictions)}")
        
        return results
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs('algorithm', exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f'algorithm/{model_name}_cf.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ {model_name} model saved")
        
        # Save the recommender interface
        with open('algorithm/cf_recommender.pkl', 'wb') as f:
            pickle.dump(self, f)
        print("✅ Recommender interface saved")
    
    def load_models(self):
        """Load all trained models"""
        model_files = {
            'user_based': 'algorithm/user_based_cf.pkl',
            'item_based': 'algorithm/item_based_cf.pkl',
            'svd': 'algorithm/svd_cf.pkl',
            'nmf': 'algorithm/nmf_cf.pkl',
            'hybrid': 'algorithm/hybrid_cf.pkl'
        }
        
        for model_name, filename in model_files.items():
            try:
                with open(filename, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"✅ {model_name} model loaded")
            except FileNotFoundError:
                print(f"⚠️ {model_name} model file not found")
        
        return len(self.models) > 0


def main():
    """Main function to demonstrate collaborative filtering"""
    print("🎬 Collaborative Filtering for Movie Recommendations")
    print("=" * 60)
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender()
    
    # Load data
    if not recommender.load_data():
        return
    
    # Display dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Users: {recommender.user_interactions['user_id'].nunique():,}")
    print(f"   Movies: {recommender.user_interactions['movie_id'].nunique():,}")
    print(f"   Interactions: {len(recommender.user_interactions):,}")
    
    sparsity = (1 - len(recommender.user_interactions) / 
                (recommender.user_interactions['user_id'].nunique() * 
                 recommender.user_interactions['movie_id'].nunique())) * 100
    print(f"   Sparsity: {sparsity:.2f}%")
    print(f"   Rating range: {recommender.user_interactions['rating'].min():.1f} - {recommender.user_interactions['rating'].max():.1f}")
    
    # Train models
    if not recommender.train_models():
        return
    
    # Test recommendations
    test_user = 0
    print(f"\n🎬 Testing Recommendations for User {test_user}:")
    print("-" * 50)
    
    for method in ['user_based', 'item_based', 'svd', 'hybrid']:
        print(f"\n{method.replace('_', ' ').title()} Recommendations:")
        recommendations = recommender.get_recommendations(test_user, method=method, n_recommendations=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} ({rec['year']}) - Score: {rec['predicted_score']:.2f}")
    
    # Test similar movies
    test_movie = 0
    movie_title = recommender.movies_df.iloc[test_movie]['title']
    print(f"\n🎭 Movies Similar to '{movie_title}':")
    print("-" * 50)
    
    similar_movies = recommender.get_similar_movies(test_movie, n_similar=5)
    for i, sim in enumerate(similar_movies, 1):
        print(f"  {i}. {sim['title']} - Similarity: {sim['similarity_score']:.3f}")
    
    # Evaluate models
    print(f"\n📊 Model Evaluation:")
    print("-" * 30)
    results = recommender.evaluate_models()
    
    if results:
        print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8}")
        print("-" * 31)
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['RMSE']:<8.3f} {metrics['MAE']:<8.3f}")
        
        best_model = min(results.keys(), key=lambda k: results[k]['RMSE'])
        print(f"\n🏆 Best model: {best_model} (RMSE: {results[best_model]['RMSE']:.3f})")
    
    # Save models
    recommender.save_models()
    


if __name__ == "__main__":
    main()