# TMDB API Configuration
import os

def get_tmdb_api_key():
    """Get TMDB API key from api file or environment variable"""
    try:
        # Try to read from api file first
        with open('api', 'r') as f:
            api_key = f.read().strip()
            if api_key:
                return api_key
    except FileNotFoundError:
        pass
    
    # Fallback to environment variable
    return os.getenv('TMDB_API_KEY', '')

# TMDB API endpoints
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"  # For poster images

# API key
TMDB_API_KEY = get_tmdb_api_key()