import requests
import streamlit as st
from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_movie_poster_url(movie_id, title=None):
    """
    Get movie poster URL from TMDB API
    
    Args:
        movie_id: TMDB movie ID
        title: Movie title (for fallback search if needed)
    
    Returns:
        str: Full URL to poster image, or None if not found
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        # First try to get movie details by ID
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US'
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        
        # If ID lookup fails and we have a title, try search
        if title and response.status_code == 404:
            return search_movie_poster(title)
            
    except requests.RequestException as e:
        st.warning(f"Error fetching poster for movie {movie_id}: {e}")
        
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_movie_poster(title):
    """
    Search for movie by title and get poster URL
    
    Args:
        title: Movie title to search for
    
    Returns:
        str: Full URL to poster image, or None if not found
    """
    if not TMDB_API_KEY or not title:
        return None
    
    try:
        # Search for movie by title
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'language': 'en-US',
            'page': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                # Get the first result's poster
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                    
    except requests.RequestException as e:
        st.warning(f"Error searching for poster for '{title}': {e}")
        
    return None

def get_fallback_poster():
    """
    Get a fallback poster image when TMDB poster is not available
    
    Returns:
        str: URL or data URI for fallback image
    """
    # You can use a placeholder image service or a local image
    return "https://via.placeholder.com/500x750/333333/ffffff?text=No+Poster"