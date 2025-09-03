"""
Debug script to test movie data loading and display
"""
import pandas as pd
import numpy as np

def test_movie_data():
    """Test loading and accessing movie data"""
    print("Testing Movie Data Loading...")
    
    try:
        # Load both datasets
        features_df = pd.read_csv('data/dataset/movie_features_engineered.csv')
        movies_df = pd.read_csv('data/dataset/movies_clean.csv')
        
        print(f"Features DF loaded: {features_df.shape}")
        print(f"Movies DF loaded: {movies_df.shape}")
        
        # Check columns
        print(f"\nFeatures DF columns:")
        print(list(features_df.columns[:10]))  # First 10 columns
        
        print(f"\nMovies DF columns:")
        print(list(movies_df.columns))
        
        # Test first few movies from features_df
        print(f"\nFirst 3 movies from features_df:")
        for i in range(min(3, len(features_df))):
            movie = features_df.iloc[i]
            print(f"{i+1}. ID: {movie['id']}, Title: '{movie['title']}'")
        
        # Test first few movies from movies_df
        print(f"\nFirst 3 movies from movies_df:")
        for i in range(min(3, len(movies_df))):
            movie = movies_df.iloc[i]
            print(f"{i+1}. ID: {movie['id']}, Title: '{movie['title']}'")
        
        # Test different access methods
        print(f"\nTesting data access methods:")
        test_movie = movies_df.iloc[0]
        
        print(f"Type of test_movie: {type(test_movie)}")
        print(f"Is Series? {isinstance(test_movie, pd.Series)}")
        print(f"Has 'get' method? {hasattr(test_movie, 'get')}")
        
        # Try different access methods
        try:
            title1 = test_movie['title']
            print(f"test_movie['title']: '{title1}'")
        except Exception as e:
            print(f"test_movie['title'] failed: {e}")
        
        try:
            title2 = test_movie.get('title', 'FALLBACK')
            print(f"test_movie.get('title'): '{title2}'")
        except Exception as e:
            print(f"test_movie.get('title') failed: {e}")
        
        try:
            title3 = getattr(test_movie, 'title', 'FALLBACK')
            print(f"getattr(test_movie, 'title'): '{title3}'")
        except Exception as e:
            print(f"getattr failed: {e}")
        
        # Test .to_dict()
        try:
            movie_dict = test_movie.to_dict()
            title4 = movie_dict.get('title', 'FALLBACK')
            print(f"test_movie.to_dict()['title']: '{title4}'")
        except Exception as e:
            print(f"to_dict() failed: {e}")
        
        # Test the create_movie_card function simulation
        print(f"\nTesting movie card data preparation:")
        
        # Method 1: Direct Series
        print("Method 1 - Direct Series:")
        test_card_data_1(test_movie)
        
        # Method 2: Convert to dict
        print("Method 2 - Convert to dict:")
        test_card_data_2(test_movie.to_dict())
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_card_data_1(movie_data):
    """Test with pandas Series"""
    try:
        title = movie_data['title'] if 'title' in movie_data else 'Unknown'
        year = movie_data['release_year'] if 'release_year' in movie_data else 'N/A'
        rating = movie_data['vote_average'] if 'vote_average' in movie_data else 0
        print(f"   Title: '{title}', Year: {year}, Rating: {rating}")
    except Exception as e:
        print(f"   Series access failed: {e}")

def test_card_data_2(movie_data):
    """Test with dictionary"""
    try:
        title = movie_data.get('title', 'Unknown')
        year = movie_data.get('release_year', 'N/A')
        rating = movie_data.get('vote_average', 0)
        print(f"   Title: '{title}', Year: {year}, Rating: {rating}")
    except Exception as e:
        print(f"   Dict access failed: {e}")

if __name__ == "__main__":
    test_movie_data()