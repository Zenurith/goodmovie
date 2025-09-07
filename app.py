import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import json
import os
import time
from datetime import datetime
from algorithm import get_recommendations, get_content_based_recommendations
from algorithm.collaborative_filtering import get_collaborative_confidence
from algorithm.tfidf_content import search_movies_tfidf, create_tfidf_search_engine
from user_manager import get_user_manager

# Page configuration
st.set_page_config(
    page_title="Good Movie - Top Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    
    .stApp {
        background-color: #0f0f0f;
    }
    
    .top-header {
        text-align: center;
        font-size: 4rem;
        font-weight: bold;
        color: #ff4444;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        border: 2px solid #ff4444;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff4444;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #ff4444;
        padding-left: 1rem;
    }
    
    .movie-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        margin: 0.5rem;
        border: 1px solid #333;
        overflow: hidden;
        position: relative;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 25px rgba(255, 68, 68, 0.4);
        border-color: #ff4444;
    }
    
    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .poster-placeholder {
        width: 100%;
        height: 400px;
        background-color: #333;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
        font-size: 0.9rem;
    }
    
    .movie-poster-container {
        position: relative;
        width: 100%;
        height: 400px;
        overflow: hidden;
    }
    
    .movie-poster-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    
    .movie-info-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(transparent, rgba(0,0,0,0.8) 50%, rgba(0,0,0,0.95));
        padding: 60px 15px 15px;
        color: white;
    }
    
    .search-container {
        background-color: #1a1a1a;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        border: 1px solid #333;
    }
    
    .rating-container {
        background-color: #2a2a2a;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        text-align: center;
    }
    
    .user-rating {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .movie-card-clickable {
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .movie-card-clickable:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.3);
    }
    
    .modal-content {
        background-color: #1a1a1a;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #ff4444;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .recommendations-section {
        background-color: #2a2a2a;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        border-left: 4px solid #ff4444;
    }
    
    .back-button {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        background-color: #ff4444;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 15px;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4);
    }
    
    .back-button:hover {
        background-color: #ff6666;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.6);
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #333;
    }
    
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .movie-info {
        font-size: 0.9rem;
        color: #cccccc;
        margin-bottom: 0.3rem;
    }
    
    .rating {
        color: #ffd700;
        font-weight: bold;
    }
    
    .genre {
        background-color: #333;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 3rem 0;
        gap: 2rem;
    }
    
    .pagination-button {
        background-color: #ff4444;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
        text-decoration: none;
        display: inline-block;
    }
    
    .pagination-button:hover {
        background-color: #ff6666;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.5);
    }
    
    .pagination-button:disabled {
        background-color: #666;
        cursor: not-allowed;
        transform: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .pagination-info {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=7200)  # Cache for 2 hours
def load_movie_data():
    try:
        df = pd.read_csv('dataset/movies.csv')
        # Optimize data types for better performance
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
        if 'vote_average' in df.columns:
            df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
        if 'vote_count' in df.columns:
            df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Movies dataset not found. Please make sure 'dataset/movies.csv' exists.")
        return pd.DataFrame()


@st.cache_data(ttl=3600, max_entries=500)  # Cache for 1 hour, limit cache size
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d33748abb9bc67b4691fcc92d60d189c&language=en-US"
        response = requests.get(url, timeout=2)  # Further reduced timeout for better UX
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w400{poster_path}"  # Reduced image size for faster loading
            return full_path
        else:
            return "https://via.placeholder.com/400x600/333333/ffffff?text=No+Poster"
    except Exception as e:
        return "https://via.placeholder.com/400x600/333333/ffffff?text=No+Poster"


def load_user_ratings():
    """Load ratings for the current logged-in user"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return {}
    
    user_manager = get_user_manager()
    return user_manager.get_user_ratings(st.session_state.current_user)

def track_session_interaction(interaction_type, movie_id=None, search_term=None, genre=None):
    """Track user interactions for session-based recommendations"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return
    
    # Initialize session data if not exists
    if 'session_data' not in st.session_state:
        st.session_state.session_data = {
            'movie_views': [],
            'modal_opens': [],
            'search_queries': [],
            'genre_interests': {},
            'session_start': time.time()
        }
    
    current_time = time.time()
    
    if interaction_type == 'movie_view' and movie_id:
        st.session_state.session_data['movie_views'].append({
            'movie_id': movie_id,
            'timestamp': current_time
        })
    elif interaction_type == 'modal_open' and movie_id:
        st.session_state.session_data['modal_opens'].append({
            'movie_id': movie_id,
            'timestamp': current_time
        })
    elif interaction_type == 'search' and search_term:
        st.session_state.session_data['search_queries'].append({
            'query': search_term.lower().strip(),
            'timestamp': current_time
        })
    elif interaction_type == 'genre_interest' and genre:
        if genre not in st.session_state.session_data['genre_interests']:
            st.session_state.session_data['genre_interests'][genre] = 0
        st.session_state.session_data['genre_interests'][genre] += 1

def calculate_implicit_signals():
    """Calculate implicit feedback signals from session data"""
    if 'session_data' not in st.session_state:
        return {}
    
    session_data = st.session_state.session_data
    implicit_signals = {}
    
    # Weight different interactions
    for view in session_data.get('movie_views', []):
        movie_id = str(view['movie_id'])
        if movie_id not in implicit_signals:
            implicit_signals[movie_id] = 0
        implicit_signals[movie_id] += 0.2  # Light interest
    
    for modal in session_data.get('modal_opens', []):
        movie_id = str(modal['movie_id'])
        if movie_id not in implicit_signals:
            implicit_signals[movie_id] = 0
        implicit_signals[movie_id] += 0.5  # Moderate interest
    
    return implicit_signals

def save_user_rating(movie_id, rating):
    """Save a single rating for the current user"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return False
    
    user_manager = get_user_manager()
    return user_manager.save_user_rating(st.session_state.current_user, movie_id, rating)

@st.cache_data(ttl=1800)  # Cache search results for 30 minutes
def search_movies(df, search_term):
    """Enhanced movie search using TF-IDF content-based search with fallback to basic search."""
    if not search_term:
        return df
    
    search_term = search_term.lower().strip()
    # Track search interaction
    track_session_interaction('search', search_term=search_term)
    
    try:
        # Use TF-IDF search for intelligent content-based search
        tfidf_results = search_movies_tfidf(df, search_term, max_results=100)
        
        if not tfidf_results.empty:
            # Sort by final_score if available, otherwise by search_similarity
            sort_column = 'final_score' if 'final_score' in tfidf_results.columns else 'search_similarity'
            tfidf_results = tfidf_results.sort_values(sort_column, ascending=False)
            
            # Remove the extra scoring columns before returning (keep original df structure)
            columns_to_keep = [col for col in df.columns if col in tfidf_results.columns]
            result = tfidf_results[columns_to_keep]
            
            return result
    
    except Exception as e:
        # Log the error but continue with fallback search
        import logging
        logging.warning(f"TF-IDF search failed, using fallback: {e}")
    
    # Fallback to basic search if TF-IDF fails
    mask = (
        df['title'].str.lower().str.contains(search_term, na=False, regex=False) |
        df['genre'].str.lower().str.contains(search_term, na=False, regex=False) |
        df['overview'].str.lower().str.contains(search_term, na=False, regex=False)
    )
    return df[mask]

def get_search_suggestions(df, partial_query, max_suggestions=5):
    """Get intelligent search suggestions using TF-IDF search engine."""
    if not partial_query or len(partial_query.strip()) < 2:
        return []
    
    try:
        # Use the TF-IDF search engine to get suggestions
        search_engine = create_tfidf_search_engine(df)
        suggestions = search_engine.get_search_suggestions(partial_query, max_suggestions)
        return suggestions
    except Exception as e:
        # Fallback to simple suggestions
        import logging
        logging.warning(f"TF-IDF suggestions failed, using fallback: {e}")
        
        # Simple fallback suggestions based on movie titles
        suggestions = []
        query_lower = partial_query.lower()
        
        for title in df['title'][:100]:  # Check first 100 movies for performance
            if query_lower in title.lower():
                suggestions.append(title)
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions



def get_hybrid_recommendations(df, user_ratings, n_recommendations=10):
    """
    Get hybrid recommendations combining collaborative filtering and content-based filtering.
    Uses progressive confidence scoring instead of hard thresholds.
    
    Args:
        df: Movie dataframe
        user_ratings: Dictionary of user ratings
        n_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame with recommended movies from both algorithms
    """
    # Use session state caching for recommendations with TTL
    implicit_signals = calculate_implicit_signals()
    cache_key = f"recommendations_{hash(str(user_ratings))}_{hash(str(implicit_signals))}_{n_recommendations}"
    cache_timestamp_key = f"{cache_key}_timestamp"
    
    # Check if cache is still valid (15 minutes TTL for dynamic recommendations)
    current_time = time.time()
    if (cache_key in st.session_state and 
        cache_timestamp_key in st.session_state and
        current_time - st.session_state[cache_timestamp_key] < 900):  # 15 minutes
        return st.session_state[cache_key]
    
    num_ratings = len([r for r in user_ratings.values() if r > 0])
    
    if not user_ratings or num_ratings == 0:
        # For completely new users, check for implicit signals
        if implicit_signals:
            # Use implicit signals to bootstrap content-based recommendations
            try:
                result = get_content_based_recommendations_with_implicit(df, {}, implicit_signals, n_recommendations)
                st.session_state[cache_key] = result
                st.session_state[cache_timestamp_key] = current_time
                return result
            except:
                pass
        
        # Fallback to pure content-based with default profile
        try:
            result = get_content_based_recommendations(df, {}, n_recommendations, implicit_signals)
            st.session_state[cache_key] = result
            st.session_state[cache_timestamp_key] = current_time
            return result
        except Exception as e:
            st.error(f"Error getting content-based recommendations: {e}")
            result = df.nlargest(n_recommendations, 'vote_average')
            st.session_state[cache_key] = result
            st.session_state[cache_timestamp_key] = current_time
            return result
    
    # Progressive confidence scoring instead of hard thresholds
    collab_confidence = get_collaborative_confidence(num_ratings, implicit_signals)
    content_confidence = 1.0 - collab_confidence
    
    # Ensure minimum contribution from each algorithm when both are viable
    if collab_confidence > 0.2 and content_confidence > 0.2:
        collab_ratio = max(0.2, collab_confidence)
        content_ratio = max(0.2, content_confidence)
        # Normalize to sum to 1
        total = collab_ratio + content_ratio
        collab_ratio /= total
        content_ratio /= total
    else:
        collab_ratio = collab_confidence
        content_ratio = content_confidence
    
    # Calculate how many recommendations from each algorithm
    content_count = max(1, int(n_recommendations * content_ratio))
    collab_count = max(1, int(n_recommendations * collab_ratio))
    
    # Ensure we don't exceed the requested number
    if content_count + collab_count > n_recommendations:
        if content_ratio > collab_ratio:
            content_count = n_recommendations - collab_count
        else:
            collab_count = n_recommendations - content_count
    
    recommendations = []
    seen_movies = set()
    
    try:
        # Get content-based recommendations
        content_recs = get_content_based_recommendations(df, user_ratings, content_count * 2)  # Get more to allow for deduplication
        
        for _, movie in content_recs.iterrows():
            if len(recommendations) >= content_count:
                break
            movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
            if movie_id not in seen_movies and movie_id not in user_ratings:
                movie['recommendation_source'] = 'content_based'
                recommendations.append(movie)
                seen_movies.add(movie_id)
    except Exception as e:
        st.error(f"Error getting content-based recommendations: {e}")
    
    try:
        # Get collaborative filtering recommendations with implicit signals
        collab_recs = get_recommendations(df, user_ratings, collab_count * 2, None, implicit_signals)  # Get more to allow for deduplication
        
        for _, movie in collab_recs.iterrows():
            if len(recommendations) >= n_recommendations:
                break
            movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
            if movie_id not in seen_movies and movie_id not in user_ratings:
                movie['recommendation_source'] = 'collaborative'
                recommendations.append(movie)
                seen_movies.add(movie_id)
    except Exception as e:
        st.error(f"Error getting collaborative recommendations: {e}")
    
    # If we still don't have enough recommendations, fill with popular movies
    if len(recommendations) < n_recommendations:
        popular_movies = df.nlargest(n_recommendations * 2, 'vote_average')
        for _, movie in popular_movies.iterrows():
            if len(recommendations) >= n_recommendations:
                break
            movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
            if movie_id not in seen_movies and movie_id not in user_ratings:
                movie['recommendation_source'] = 'popular'
                recommendations.append(movie)
                seen_movies.add(movie_id)
    
    if recommendations:
        result = pd.DataFrame(recommendations[:n_recommendations])
        st.session_state[cache_key] = result
        st.session_state[cache_timestamp_key] = current_time
        return result
    else:
        # Fallback to top-rated movies
        result = df.nlargest(n_recommendations, 'vote_average')
        st.session_state[cache_key] = result
        st.session_state[cache_timestamp_key] = current_time
        return result

def display_pagination_controls(total_items, current_page, items_per_page, key_prefix=""):
    """Display pagination controls with Previous/Next buttons"""
    total_pages = (total_items - 1) // items_per_page + 1
    
    if total_pages <= 1:
        return
    
    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page > 1:
            if st.button("◀ Previous", key=f"{key_prefix}prev_page", type="primary"):
                st.session_state.current_page = current_page - 1
                st.rerun()
        else:
            st.button("◀ Previous", key=f"{key_prefix}prev_page_disabled", disabled=True)
    
    with col2:
        st.markdown(f'<div class="pagination-info">Page {current_page} of {total_pages}</div>', unsafe_allow_html=True)
    
    with col3:
        if current_page < total_pages:
            if st.button("Next ▶", key=f"{key_prefix}next_page", type="primary"):
                st.session_state.current_page = current_page + 1
                st.rerun()
        else:
            st.button("Next ▶", key=f"{key_prefix}next_page_disabled", disabled=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def paginate_dataframe(df, page, items_per_page):
    """Get a subset of dataframe for the current page"""
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    return df.iloc[start_idx:end_idx]

def show_login_page():
    """Display login/registration interface"""
    user_manager = get_user_manager()
    
    st.markdown('<div class="top-header">🎬 Good Movie</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Login/Register tabs
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        
        st.markdown("---")
        login_username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        
        if st.button("Login", type="primary"):
            if login_username.strip():
                if user_manager.login_user(login_username):
                    st.session_state.current_user = login_username.strip().lower()
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    st.error("Username not found. Please register first or check your spelling.")
            else:
                st.error("Please enter a username")
    
    with tab2:
        st.markdown("### Create New Account")
        new_username = st.text_input("Choose Username", placeholder="Enter a unique username", key="register_username")
        
        if st.button("Register", type="primary"):
            if new_username.strip():
                if len(new_username.strip()) < 2:
                    st.error("Username must be at least 2 characters long")
                elif user_manager.create_user(new_username):
                    st.session_state.current_user = new_username.strip().lower()
                    st.success(f"Account created! Welcome, {new_username}!")
                    st.rerun()
                else:
                    st.error("Username already exists. Please choose a different username.")
            else:
                st.error("Please enter a username")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System stats
    stats = user_manager.get_system_stats()
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", stats["total_users"])
    with col2:
        st.metric("Total Ratings", stats["total_ratings"])
    with col3:
        st.metric("Avg Ratings/User", stats["average_ratings_per_user"])

def display_movie_card(movie, clickable=True, context=""):
    movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
    
    # Track movie view for session-based recommendations
    if 'id' in movie and pd.notna(movie['id']):
        track_session_interaction('movie_view', movie_id=int(movie['id']))
        # Track genre interests
        if pd.notna(movie.get('genre')):
            genres = [g.strip() for g in str(movie['genre']).split(',') if g.strip()]
            for genre in genres[:2]:  # Track top 2 genres to avoid spam
                track_session_interaction('genre_interest', genre=genre)
    
    # Get poster URL if movie has an ID (with lazy loading)
    poster_url = None
    if 'id' in movie and pd.notna(movie['id']):
        # Use session state to cache poster URLs
        poster_cache_key = f"poster_{int(movie['id'])}"
        if poster_cache_key not in st.session_state:
            st.session_state[poster_cache_key] = fetch_poster(int(movie['id']))
        poster_url = st.session_state[poster_cache_key]
    
    # Load user ratings
    user_ratings = load_user_ratings()
    current_rating = user_ratings.get(movie_id, 0)
    
    # Create movie card with overlay design matching reference  
    if poster_url and not poster_url.endswith('No+Poster'):
        poster_html = f'<img src="{poster_url}" class="movie-poster-img" alt="{movie["title"]}" />'
    else:
        poster_html = '<div class="poster-placeholder">No Poster Available</div>'
    
    # Create star rating display
    vote_avg = float(movie['vote_average'])
    full_stars = int(vote_avg)
    half_star = 1 if (vote_avg - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    star_rating = '★' * full_stars + ('☆' if half_star else '') + '☆' * empty_stars
    
    # User rating stars if exists - ensure clean HTML
    user_stars_html = ""
    if current_rating > 0:
        user_stars_html = f'<div style="color: #ff4444; font-size: 0.8rem; margin-top: 3px;">Your Rating: {"★" * current_rating}</div>'
    
    # Prepare safe content
    safe_title = str(movie['title']).replace('"', '&quot;').replace("'", "&#39;")
    
    # Process genres safely
    genres_html = ""
    try:
        genre_str = str(movie['genre']) if pd.notna(movie['genre']) else ""
        if genre_str and genre_str != 'nan':
            genres = [genre.strip() for genre in genre_str.split(',') if genre.strip()][:3]
            genres_html = ''.join([f'<span class="genre">{genre}</span>' for genre in genres])
    except:
        genres_html = ""
    
    # Build the movie card HTML step by step to avoid any HTML issues
    overlay_content = f"""
                <div class="movie-title" style="font-size: 1.1rem; font-weight: bold; margin-bottom: 8px; line-height: 1.2;">{safe_title}</div>
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span style="color: #ffd700; margin-right: 8px; font-size: 0.9rem;">{star_rating[:5]}</span>
                    <span style="color: #ffd700; font-size: 0.9rem;">{movie['vote_average']}</span>
                    <span style="color: #999; margin-left: 8px; font-size: 0.8rem;">({movie['vote_count']} votes)</span>
                </div>
                <div style="color: #ccc; font-size: 0.8rem; margin-bottom: 5px;">{str(movie['release_date'])[:4]} • Popularity: {movie['popularity']:.0f}</div>
                <div style="margin-bottom: 8px;">{genres_html}</div>"""
    
    # Only add user stars if they exist
    if user_stars_html:
        overlay_content += user_stars_html
    
    # Complete the HTML structure
    card_html = f"""<div class="movie-card">
        <div class="movie-poster-container">
            {poster_html}
            <div class="movie-info-overlay">{overlay_content}
            </div>
        </div>
    </div>"""
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Add the rate movie button
    if clickable:
        unique_key = f"rate_{movie_id}_{context}" if context else f"rate_{movie_id}"
        if st.button("Rate Movie", key=unique_key, type="primary", use_container_width=True):
            # Track modal opening for session-based recommendations
            if 'id' in movie and pd.notna(movie['id']):
                track_session_interaction('modal_open', movie_id=int(movie['id']))
            st.session_state.selected_movie = movie
            st.session_state.show_modal = True
            st.rerun()

def display_movie_modal(movie, df):
    movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
    
    # Get poster URL (with lazy loading)
    poster_url = None
    if 'id' in movie and pd.notna(movie['id']):
        # Use session state to cache poster URLs
        poster_cache_key = f"poster_{int(movie['id'])}"
        if poster_cache_key not in st.session_state:
            st.session_state[poster_cache_key] = fetch_poster(int(movie['id']))
        poster_url = st.session_state[poster_cache_key]
    
    # Load current rating
    user_ratings = load_user_ratings()
    current_rating = user_ratings.get(movie_id, 0)
    
    # Modal content with back button
    st.markdown(f"""
    <div class="modal-content">
        <div class="modal-header">
            <h1 style="color: #ff4444; margin: 0;">{movie['title']}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Back to main page button (top-left corner)
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        if st.button("🏠 Back to Main", key="back_to_main", type="secondary"):
            if 'show_modal' in st.session_state:
                del st.session_state.show_modal
            if 'selected_movie' in st.session_state:
                del st.session_state.selected_movie
            st.rerun()
    
    # Two columns layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display poster
        if poster_url and not poster_url.endswith('No+Poster'):
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown('<div class="poster-placeholder">No Poster Available</div>', unsafe_allow_html=True)
    
    with col2:
        # Movie details
        st.markdown(f"""
        <div style="padding: 1rem;">
            <div style="font-size: 1.2rem; margin-bottom: 1rem;">
                <span style="color: #ffd700;">⭐ {movie['vote_average']}</span> ({movie['vote_count']} votes)
            </div>
            <div style="margin-bottom: 0.5rem;"><strong>Release Date:</strong> {movie['release_date']}</div>
            <div style="margin-bottom: 0.5rem;"><strong>Popularity:</strong> {movie['popularity']:.1f}</div>
            <div style="margin-bottom: 1rem;">
                <strong>Genres:</strong><br>
                {' '.join([f'<span class="genre">{genre.strip()}</span>' for genre in str(movie['genre']).split(',')])}
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>Overview:</strong><br>
                <p style="text-align: justify; line-height: 1.5;">{movie['overview']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Star rating system
    st.markdown("<h3 style='color: #ff4444; text-align: center; margin-top: 2rem;'>⭐ Rate this Movie (1-10 stars)</h3>", unsafe_allow_html=True)
    
    # Create star rating interface
    rating_cols = st.columns(10)
    new_rating = current_rating
    
    for i in range(10):
        with rating_cols[i]:
            if i+1 <= current_rating:
                if st.button(f"{i+1}⭐", key=f"star_{movie_id}_{i+1}", type="primary"):
                    new_rating = i + 1
            else:
                if st.button(f"{i+1}☆", key=f"star_{movie_id}_{i+1}"):
                    new_rating = i + 1
    
    # Update rating if changed
    if new_rating != current_rating:
        if save_user_rating(movie_id, new_rating):
            st.success(f"Rated {movie['title']}: {new_rating}/10 stars!")
            st.rerun()
        else:
            st.error("Failed to save rating. Please try again.")
    
    # Display current rating
    if current_rating > 0:
        stars_display = '⭐' * current_rating + '☆' * (10 - current_rating)
        st.markdown(f"<div style='text-align: center; font-size: 1.5rem; color: #ffd700; margin: 1rem 0;'>{stars_display} ({current_rating}/10)</div>", unsafe_allow_html=True)
    
    # Show content-based "You may also like" recommendations
    st.markdown('<div class="recommendations-section">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #ff4444; margin-bottom: 1rem;'>🎬 You May Also Like</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #cccccc; margin-bottom: 1rem; font-style: italic;'>Movies similar to <strong>{movie['title']}</strong></p>", unsafe_allow_html=True)
    
    # Get similar movies using content-based algorithm
    from algorithm.content_based import create_content_based_recommender
    content_recommender = create_content_based_recommender(df)
    similar_movies = content_recommender.get_similar_movies(movie_id, 4)
    
    if not similar_movies.empty:
        # Display 4 similar movies in a single row
        rec_cols = st.columns(4)
        for idx, (_, rec_movie) in enumerate(similar_movies.iterrows()):
            with rec_cols[idx]:
                # Add similarity score if available
                similarity_score = rec_movie.get('similarity_score', 0)
                if similarity_score > 0:
                    st.markdown(f"<div style='text-align: center; color: #ff4444; font-size: 0.8rem; margin-bottom: 0.5rem;'>Match: {similarity_score:.2f}</div>", unsafe_allow_html=True)
                display_movie_card(rec_movie, clickable=True, context=f"similar_{idx}")
    else:
        st.markdown("<p style='color: #999; text-align: center; padding: 2rem;'>No similar movies found. Try rating this movie to get personalized recommendations!</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show personalized recommendations based on progressive confidence
    # Only show if user has rated this specific movie
    if current_rating > 0:
        num_ratings = len([rating for rating in user_ratings.values() if rating > 0])
        implicit_signals = calculate_implicit_signals()
        collab_confidence = get_collaborative_confidence(num_ratings, implicit_signals)
        
        if collab_confidence > 0.4:  # Dynamic threshold instead of hard 3-rating rule
            st.markdown('<div class="recommendations-section" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown("<h3 style='color: #ff4444; margin-bottom: 1rem;'>🎯 More Recommendations for You</h3>", unsafe_allow_html=True)
            
            recommendations = get_hybrid_recommendations(df, user_ratings, 4)
            
            if not recommendations.empty:
                rec_cols = st.columns(4)
                for idx, (_, rec_movie) in enumerate(recommendations.iterrows()):
                    with rec_cols[idx]:
                        display_movie_card(rec_movie, clickable=True, context=f"rec_{idx}")
            else:
                st.write("Rate more movies to get better recommendations!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        elif num_ratings > 0:
            st.markdown('<div class="recommendations-section" style="margin-top: 1rem; text-align: center; padding: 1rem;">', unsafe_allow_html=True)
            st.markdown(f"<p style='color: #ff4444; font-style: italic;'>Rate {3 - num_ratings} more movies to unlock personalized collaborative filtering recommendations!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show message encouraging user to rate this movie first
        st.markdown('<div class="recommendations-section" style="margin-top: 1rem; text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

def main():
    # Initialize session state with performance optimizations
    if 'show_modal' not in st.session_state:
        st.session_state.show_modal = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'movies_per_page' not in st.session_state:
        st.session_state.movies_per_page = 12
    if 'movie_df_cache' not in st.session_state:
        st.session_state.movie_df_cache = None
    if 'poster_cache' not in st.session_state:
        st.session_state.poster_cache = {}
    if 'session_data' not in st.session_state:
        st.session_state.session_data = {
            'movie_views': [],
            'modal_opens': [],
            'search_queries': [],
            'genre_interests': {},
            'session_start': time.time()
        }
    
    # Check if user is logged in
    if not st.session_state.current_user:
        show_login_page()
        return
    
    # Load data with caching
    if st.session_state.movie_df_cache is None:
        df = load_movie_data()
        st.session_state.movie_df_cache = df
    else:
        df = st.session_state.movie_df_cache
    
    if df.empty:
        st.error("No movie data available")
        return
    
    # User header with logout
    user_manager = get_user_manager()
    user_stats = user_manager.get_user_stats(st.session_state.current_user)
    
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<div class="top-header">🎬 Good Movie</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown(f"**Welcome, {st.session_state.current_user}!**")
        st.caption(f"Rated: {user_stats.get('total_rated', 0)} movies | Avg: {user_stats.get('average_rating', 0.0)}/10")
        if st.button("Logout", type="secondary"):
            st.session_state.current_user = None
            st.rerun()
        
        # Delete account option with confirmation
        if 'confirm_delete' not in st.session_state:
            st.session_state.confirm_delete = False
        
        if not st.session_state.confirm_delete:
            if st.button("🗑️ Delete Account", type="secondary"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning("⚠️ This will permanently delete your account and all ratings!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", type="primary"):
                    # Delete the user account
                    user_manager.delete_user(st.session_state.current_user)
                    st.session_state.current_user = None
                    st.session_state.confirm_delete = False
                    st.success("Account deleted successfully!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()
    
    # Show modal if a movie is selected
    if st.session_state.show_modal and 'selected_movie' in st.session_state:
        display_movie_modal(st.session_state.selected_movie, df)
        return
    
    # Search section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        # Auto-focus search if coming from modal
        placeholder_text = "Try: 'action movies', 'Marvel', 'comedy 2020'..."
        if 'focus_search' in st.session_state and st.session_state.focus_search:
            st.session_state.focus_search = False
            
        # Use selected suggestion if available, otherwise use current search value
        suggestion_value = st.session_state.get('selected_suggestion', '')
        current_search = st.session_state.get('movie_search', '')
        
        # Use suggestion if available, otherwise keep current search
        default_value = suggestion_value if suggestion_value else current_search
        
        search_term = st.text_input(
            "🔍 Search movies by title, genre, or description",
            placeholder=placeholder_text,
            value=default_value,
            key="movie_search"
        )
        
        # Clear suggestion only after it has been applied to the search input
        if suggestion_value and search_term == suggestion_value:
            st.session_state.selected_suggestion = ''
        
        # Show search suggestions if user has typed something
        if search_term and len(search_term) >= 2:
            suggestions = get_search_suggestions(df, search_term, max_suggestions=3)
            if suggestions and search_term.lower() not in [s.lower() for s in suggestions]:
                with st.expander("💡 Search Suggestions", expanded=False):
                    for suggestion in suggestions:
                        if st.button(f"🎬 {suggestion}", key=f"suggest_{suggestion}", use_container_width=True):
                            st.session_state.selected_suggestion = suggestion
                            st.rerun()
    
    with search_col2:
        # Add empty label to align button with input field
        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle search
    search_results = None
    if search_term or search_button:
        # Reset to page 1 when new search is made
        if search_button or ('last_search_term' not in st.session_state or st.session_state.last_search_term != search_term):
            st.session_state.current_page = 1
            st.session_state.last_search_term = search_term
        
        search_results = search_movies(df, search_term)
        
        if not search_results.empty:
            # Show enhanced search results header with TF-IDF indicator
            results_text = f"Results ({len(search_results)} movies found)"
            if 'search_similarity' in search_results.columns:
                avg_relevance = search_results['search_similarity'].mean() * 100
                results_text += f" - Avg Relevance: {avg_relevance:.1f}%"
            
            st.markdown(f'<div class="section-header">{results_text}</div>', unsafe_allow_html=True)
            
            # Show search method used
            search_method = "TF-IDF Content Search" if 'search_similarity' in search_results.columns else "Basic Text Search"
            
            # Paginate search results
            paginated_search = paginate_dataframe(search_results, st.session_state.current_page, st.session_state.movies_per_page)
            
            # Display search results with relevance info
            search_cols = st.columns(4)
            for idx, (_, movie) in enumerate(paginated_search.iterrows()):
                with search_cols[idx % 4]:
                    # Add relevance score if available
                    if 'search_similarity' in movie and pd.notna(movie['search_similarity']):
                        relevance_score = movie['search_similarity'] * 100
                        st.markdown(f'<div style="text-align: center; color: #ff4444; font-size: 0.8rem; margin-bottom: 0.5rem;">Relevance: {relevance_score:.1f}%</div>', unsafe_allow_html=True)
                    
                    display_movie_card(movie, clickable=True, context=f"search_{idx}")
            
            # Add pagination controls for search results
            display_pagination_controls(len(search_results), st.session_state.current_page, st.session_state.movies_per_page, "search_")
            
        else:
            st.warning(f"No movies found for '{search_term}'")
            
            # Suggest alternative searches
            if len(search_term) >= 3:
                st.markdown("### 💡 Try these suggestions:")
                
                # Get trending searches or popular genres
                try:
                    search_engine = create_tfidf_search_engine(df)
                    trending = search_engine.get_trending_searches()[:5]
                    
                    suggestion_cols = st.columns(len(trending) if trending else 3)
                    for idx, suggestion in enumerate(trending):
                        with suggestion_cols[idx % len(suggestion_cols)]:
                            if st.button(f"🎭 {suggestion}", key=f"trending_{idx}"):
                                st.session_state.movie_search = suggestion
                                st.rerun()
                except:
                    # Fallback suggestions
                    fallback_suggestions = ["Action", "Comedy", "Drama", "Horror", "Romance"]
                    suggestion_cols = st.columns(5)
                    for idx, genre in enumerate(fallback_suggestions):
                        with suggestion_cols[idx]:
                            if st.button(f"🎭 {genre}", key=f"fallback_{idx}"):
                                st.session_state.movie_search = f"{genre} movies"
                                st.rerun()
    
    # Only show trending if no search is active
    if not search_term:
        # Trending Movies section with pagination
        st.markdown('<div class="section-header">📈 Trending Movies</div>', unsafe_allow_html=True)
        
        # Sort by vote_average and popularity for all trending movies
        trending_movies = df.nlargest(100, ['vote_average', 'popularity'])
        
        # Paginate trending movies
        paginated_trending = paginate_dataframe(trending_movies, st.session_state.current_page, st.session_state.movies_per_page)
        
        # Display in a grid layout (4 columns)
        cols = st.columns(4)
        for idx, (_, movie) in enumerate(paginated_trending.iterrows()):
            with cols[idx % 4]:
                display_movie_card(movie, clickable=True, context=f"trending_{idx}")
        
        # Add pagination controls for trending movies
        display_pagination_controls(len(trending_movies), st.session_state.current_page, st.session_state.movies_per_page, "trending_")
    
    # Add recommendations section for users who have rated movies
    user_ratings = load_user_ratings()
    rated_movies = [movie_id for movie_id, rating in user_ratings.items() if rating > 0]
    
    if len(rated_movies) >= 2 and not search_term:
        st.markdown('<div class="section-header">🎯 Recommended for You</div>', unsafe_allow_html=True)
        recommendations = get_hybrid_recommendations(df, user_ratings, 8)
        
        if not recommendations.empty:
            rec_cols = st.columns(4)
            for idx, (_, rec_movie) in enumerate(recommendations.head(8).iterrows()):
                with rec_cols[idx % 4]:
                    display_movie_card(rec_movie, clickable=True, context=f"home_rec_{idx}")
    
    # Sidebar profile (always show)
    st.sidebar.title("🎬 My Profile")
    
    # User stats
    user_stats = user_manager.get_user_stats(st.session_state.current_user)
    if user_stats:
        st.sidebar.metric("Movies Rated", user_stats.get('total_rated', 0))
        if user_stats.get('total_rated', 0) > 0:
            st.sidebar.metric("Average Rating", f"{user_stats.get('average_rating', 0.0)}/10")
            st.sidebar.write(f"**Highest:** {user_stats.get('highest_rating', 0)}/10")
            st.sidebar.write(f"**Lowest:** {user_stats.get('lowest_rating', 0)}/10")
        
        st.sidebar.write(f"**Member since:** {user_stats.get('created_at', '')[:10]}")
    
    if rated_movies:
        st.sidebar.markdown("---")
        st.sidebar.write("**Recent Ratings:**")
        # Show last 5 ratings
        recent_ratings = dict(list(load_user_ratings().items())[-5:])
        for movie_id, rating in recent_ratings.items():
            movie_row = df[df['id'].astype(str) == str(movie_id)]
            if not movie_row.empty:
                movie_title = movie_row.iloc[0]['title']
                st.sidebar.write(f"{'★' * rating} {movie_title[:20]}...")
        
        st.sidebar.markdown("---")
        if st.sidebar.button("Clear All My Ratings"):
            # Clear ratings for current user
            for movie_id in list(load_user_ratings().keys()):
                user_manager.remove_user_rating(st.session_state.current_user, movie_id)
            st.rerun()
        
    else:
        st.sidebar.write("No ratings yet. Click on movies to rate them!")
    

if __name__ == "__main__":
    main()