import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import json
import os
from algorithm import get_recommendations

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
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
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
    if os.path.exists('user_ratings.json'):
        try:
            with open('user_ratings.json', 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_ratings(ratings):
    try:
        with open('user_ratings.json', 'w') as f:
            json.dump(ratings, f)
    except:
        pass

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
        user_ratings[movie_id] = new_rating
        save_user_ratings(user_ratings)
        st.success(f"Rated {movie['title']}: {new_rating}/10 stars!")
        st.rerun()
    
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
    
    
    
    # Load data
    df = load_movie_data()
    
    if df.empty:
        st.error("No movie data available")
        return
    
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
        search_results = search_movies(df, search_term)
        
        if not search_results.empty:
            st.markdown(f'<div class="section-header">🔍 Search Results ({len(search_results)} movies found)</div>', unsafe_allow_html=True)
            
            # Display search results
            search_cols = st.columns(4)
            for idx, (_, movie) in enumerate(search_results.head(12).iterrows()):
                with search_cols[idx % 4]:
                    display_movie_card(movie, clickable=True)
        else:
            st.warning(f"No movies found for '{search_term}'")
    
    # Only show trending if no search is active
    if not search_term:
        # Top 10 section
        st.markdown('<div class="section-header">📈 Trending Today</div>', unsafe_allow_html=True)
        
        # Sort by vote_average and popularity for top movies
        top_movies = df.nlargest(10, ['vote_average', 'popularity'])
        
        # Display in a grid layout (5 columns for top 10)
        cols = st.columns(5)
        for idx, (_, movie) in enumerate(top_movies.head(5).iterrows()):
            with cols[idx]:
                display_movie_card(movie, clickable=True)
        
        # Second row of top 10
        cols2 = st.columns(5)
        for idx, (_, movie) in enumerate(top_movies.tail(5).iterrows()):
            with cols2[idx]:
                display_movie_card(movie, clickable=True)
        
        # Additional trending section
        st.markdown('<div class="section-header">🔥 More Trending</div>', unsafe_allow_html=True)
        
        # Show more movies in a different layout
        trending_movies = df.nlargest(20, 'popularity').tail(10)
        
        cols3 = st.columns(4)
        for idx, (_, movie) in enumerate(trending_movies.iterrows()):
            with cols3[idx % 4]:
                display_movie_card(movie, clickable=True)
    
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
            
            filtered_movies = filtered_df.nlargest(12, ['vote_average', 'popularity'])
            cols4 = st.columns(4)
            for idx, (_, movie) in enumerate(filtered_movies.iterrows()):
                with cols4[idx % 4]:
                    display_movie_card(movie, clickable=True)
    else:
        st.sidebar.title("🎬 My Ratings")
        if rated_movies:
            st.sidebar.write(f"You have rated {len(rated_movies)} movies")
            
            if st.sidebar.button("Clear All Ratings"):
                save_user_ratings({})
                st.rerun()
        else:
            st.sidebar.write("No ratings yet. Click on movies to rate them!")
    
    # Footer stats
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        st.metric("Average Rating", f"{df['vote_average'].mean():.1f}")
    with col3:
        st.metric("Genres Available", len(all_genres) if 'all_genres' in locals() else 0)
    with col4:
        display_results = search_results if search_results is not None else df
        st.metric("Results Shown", len(display_results))
    with col5:
        st.metric("Movies Rated", len(rated_movies))

if __name__ == "__main__":
    main()