import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import json
import os
from algorithm import get_recommendations
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

@st.cache_data
def load_movie_data():
    try:
        df = pd.read_csv('dataset/movies.csv')
        return df
    except FileNotFoundError:
        st.error("Movies dataset not found. Please make sure 'dataset/movies.csv' exists.")
        return pd.DataFrame()

@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d33748abb9bc67b4691fcc92d60d189c&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
        else:
            return "https://via.placeholder.com/500x750/333333/ffffff?text=No+Poster"
    except Exception as e:
        return "https://via.placeholder.com/500x750/333333/ffffff?text=No+Poster"

def load_user_ratings():
    """Load ratings for the current logged-in user"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return {}
    
    user_manager = get_user_manager()
    return user_manager.get_user_ratings(st.session_state.current_user)

def save_user_ratings(ratings):
    """Save ratings for the current logged-in user"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return
    
    # This function is kept for compatibility, but individual ratings are saved differently now
    pass

def save_user_rating(movie_id, rating):
    """Save a single rating for the current user"""
    if 'current_user' not in st.session_state or not st.session_state.current_user:
        return False
    
    user_manager = get_user_manager()
    return user_manager.save_user_rating(st.session_state.current_user, movie_id, rating)

def search_movies(df, search_term):
    if not search_term:
        return df
    
    search_term = search_term.lower()
    mask = (
        df['title'].str.lower().str.contains(search_term, na=False) |
        df['genre'].str.lower().str.contains(search_term, na=False) |
        df['overview'].str.lower().str.contains(search_term, na=False)
    )
    return df[mask]

def collaborative_filtering_recommendations(df, user_ratings, n_recommendations=5):
    """
    Wrapper function for the collaborative filtering algorithm.
    
    Args:
        df: Movie dataframe
        user_ratings: Dictionary of user ratings
        n_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame with recommended movies
    """
    return get_recommendations(df, user_ratings, n_recommendations)

def display_pagination_controls(total_items, current_page, items_per_page):
    """Display pagination controls with Previous/Next buttons"""
    total_pages = (total_items - 1) // items_per_page + 1
    
    if total_pages <= 1:
        return
    
    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page > 1:
            if st.button("◀ Previous", key="prev_page", type="primary"):
                st.session_state.current_page = current_page - 1
                st.rerun()
        else:
            st.button("◀ Previous", key="prev_page_disabled", disabled=True)
    
    with col2:
        st.markdown(f'<div class="pagination-info">Page {current_page} of {total_pages}</div>', unsafe_allow_html=True)
    
    with col3:
        if current_page < total_pages:
            if st.button("Next ▶", key="next_page", type="primary"):
                st.session_state.current_page = current_page + 1
                st.rerun()
        else:
            st.button("Next ▶", key="next_page_disabled", disabled=True)
    
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
        
        # Show existing users for convenience
        existing_users = user_manager.get_all_users()
        if existing_users:
            st.markdown("**Existing users:**")
            cols = st.columns(min(len(existing_users), 4))
            for i, username in enumerate(existing_users[:8]):  # Show up to 8 users
                with cols[i % 4]:
                    if st.button(f"Login as {username}", key=f"quick_login_{username}"):
                        if user_manager.login_user(username):
                            st.session_state.current_user = username
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
        
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

def display_movie_card(movie, clickable=True):
    movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
    
    # Get poster URL if movie has an ID
    poster_url = None
    if 'id' in movie and pd.notna(movie['id']):
        poster_url = fetch_poster(int(movie['id']))
    
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
        if st.button("Rate Movie", key=f"rate_{movie_id}", type="primary", use_container_width=True):
            st.session_state.selected_movie = movie
            st.session_state.show_modal = True
            st.rerun()

def display_movie_modal(movie, df):
    movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
    
    # Get poster URL
    poster_url = None
    if 'id' in movie and pd.notna(movie['id']):
        poster_url = fetch_poster(int(movie['id']))
    
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
    
    # Show recommendations based on this rating
    if current_rating > 0:
        st.markdown('<div class="recommendations-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ff4444; margin-bottom: 1rem;'>🎯 Recommended for You</h3>", unsafe_allow_html=True)
        
        recommendations = get_recommendations(df, user_ratings, 4)
        
        if not recommendations.empty:
            rec_cols = st.columns(4)
            for idx, (_, rec_movie) in enumerate(recommendations.iterrows()):
                with rec_cols[idx]:
                    display_movie_card(rec_movie, clickable=True)
        else:
            st.write("Rate more movies to get better recommendations!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'show_modal' not in st.session_state:
        st.session_state.show_modal = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'movies_per_page' not in st.session_state:
        st.session_state.movies_per_page = 12
    
    # Check if user is logged in
    if not st.session_state.current_user:
        show_login_page()
        return
    
    # Load data
    df = load_movie_data()
    
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
        placeholder_text = "Enter movie name, genre, or keywords..."
        if 'focus_search' in st.session_state and st.session_state.focus_search:
            st.session_state.focus_search = False
            
        search_term = st.text_input(
            "🔍 Search movies by title, genre, or description",
            placeholder=placeholder_text,
            key="movie_search"
        )
    
    with search_col2:
        search_button = st.button("Search", type="primary")
    
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
            st.markdown(f'<div class="section-header">🔍 Search Results ({len(search_results)} movies found)</div>', unsafe_allow_html=True)
            
            # Paginate search results
            paginated_search = paginate_dataframe(search_results, st.session_state.current_page, st.session_state.movies_per_page)
            
            # Display search results
            search_cols = st.columns(4)
            for idx, (_, movie) in enumerate(paginated_search.iterrows()):
                with search_cols[idx % 4]:
                    display_movie_card(movie, clickable=True)
            
            # Add pagination controls for search results
            display_pagination_controls(len(search_results), st.session_state.current_page, st.session_state.movies_per_page)
        else:
            st.warning(f"No movies found for '{search_term}'")
    
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
                display_movie_card(movie, clickable=True)
        
        # Add pagination controls for trending movies
        display_pagination_controls(len(trending_movies), st.session_state.current_page, st.session_state.movies_per_page)
    
    # Add recommendations section for users who have rated movies
    user_ratings = load_user_ratings()
    rated_movies = [movie_id for movie_id, rating in user_ratings.items() if rating > 0]
    
    if len(rated_movies) >= 2 and not search_term:
        st.markdown('<div class="section-header">🎯 Recommended for You</div>', unsafe_allow_html=True)
        recommendations = get_recommendations(df, user_ratings, 8)
        
        if not recommendations.empty:
            rec_cols = st.columns(4)
            for idx, (_, rec_movie) in enumerate(recommendations.head(8).iterrows()):
                with rec_cols[idx % 4]:
                    display_movie_card(rec_movie, clickable=True)
    
    # Sidebar filters (only show if no search is active)
    if not search_term:
        st.sidebar.title("🎬 Filters")
        
        # Genre filter
        all_genres = set()
        for genres in df['genre'].dropna():
            all_genres.update([g.strip() for g in str(genres).split(',')])
        
        selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + sorted(list(all_genres)))
        
        # Rating filter
        min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.1)
        
        # Year filter
        df['year'] = pd.to_datetime(df['release_date']).dt.year
        min_year = st.sidebar.slider("From Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].min()))
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_genre != "All":
            filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, na=False)]
        
        filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
        filtered_df = filtered_df[filtered_df['year'] >= min_year]
        
        if len(filtered_df) != len(df):
            st.markdown('<div class="section-header">🔍 Filtered Results</div>', unsafe_allow_html=True)
            
            filtered_movies = filtered_df.nlargest(len(filtered_df), ['vote_average', 'popularity'])
            
            # Paginate filtered results
            paginated_filtered = paginate_dataframe(filtered_movies, st.session_state.current_page, st.session_state.movies_per_page)
            
            cols4 = st.columns(4)
            for idx, (_, movie) in enumerate(paginated_filtered.iterrows()):
                with cols4[idx % 4]:
                    display_movie_card(movie, clickable=True)
            
            # Add pagination controls for filtered results
            display_pagination_controls(len(filtered_movies), st.session_state.current_page, st.session_state.movies_per_page)
    else:
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