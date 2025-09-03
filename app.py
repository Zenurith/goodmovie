import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from poster_fetcher import get_movie_poster_url, get_fallback_poster
from config import TMDB_API_KEY

# Import collaborative filtering from algorithm folder
try:
    import sys
    sys.path.append('algorithm')
    from collaborative_filtering import CollaborativeFilteringRecommender
    COLLABORATIVE_AVAILABLE = True
except ImportError:
    COLLABORATIVE_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="GoodMovie - Netflix Style",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Netflix-style UI matching references
st.markdown("""
<style>
    /* Hide Streamlit elements */
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, 
    .styles_viewerBadge__1yB5_, #MainMenu, header, footer, .stDeployButton {
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #141414 100%);
        color: #ffffff;
        font-family: 'Netflix Sans', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Top navigation bar */
    .top-nav {
        background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%);
        padding: 20px 40px;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo {
        color: #e50914;
        font-size: 32px;
        font-weight: bold;
        text-decoration: none;
    }
    
    .nav-links {
        display: flex;
        gap: 25px;
    }
    
    .nav-link {
        color: #e5e5e5;
        text-decoration: none;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .nav-link:hover, .nav-link.active {
        color: #ffffff;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.8)), 
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600"><rect fill="%23141414"/></svg>');
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 64px;
        font-weight: bold;
        margin-bottom: 20px;
        background: linear-gradient(45deg, #e50914, #f40612);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(229,9,20,0.3);
    }
    
    .hero-subtitle {
        font-size: 24px;
        color: #e5e5e5;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Section headers matching UI references */
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
        margin: 40px 40px 20px 40px;
        padding-left: 20px;
        border-left: 4px solid #e50914;
        background: linear-gradient(90deg, rgba(229,9,20,0.1) 0%, transparent 100%);
        padding: 15px 0 15px 20px;
    }
    
    .top-10-header {
        position: relative;
        font-size: 32px;
        font-weight: 900;
        color: #ffffff;
        margin: 40px 40px 20px 40px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .top-10-badge {
        background: linear-gradient(45deg, #e50914, #f40612);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Movie carousel container */
    .movie-carousel {
        padding: 0 40px;
        margin-bottom: 50px;
    }
    
    /* Movie card styling to match Netflix */
    .movie-card {
        background: #181818;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        position: relative;
        aspect-ratio: 16/9;
        margin: 10px 5px;
        border: 2px solid transparent;
    }
    
    .movie-card:hover {
        transform: scale(1.05) translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
        border-color: #e50914;
        z-index: 100;
    }
    
    .movie-poster {
        width: 100%;
        height: 200px;
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        position: relative;
        overflow: hidden;
    }
    
    .movie-poster::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(229,9,20,0.1), rgba(0,0,0,0.3));
    }
    
    .movie-info {
        padding: 15px;
        background: linear-gradient(180deg, transparent 0%, #181818 100%);
    }
    
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.3;
        height: 42px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .movie-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .movie-year {
        color: #b3b3b3;
        font-size: 14px;
    }
    
    .movie-rating {
        display: flex;
        align-items: center;
        gap: 4px;
        color: #46d369;
        font-weight: bold;
        font-size: 14px;
    }
    
    .movie-genres {
        color: #e50914;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .match-score {
        background: linear-gradient(45deg, #46d369, #00ff41);
        color: #000;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Search section */
    .search-section {
        background: rgba(20, 20, 20, 0.9);
        padding: 30px 40px;
        margin: 40px 40px 60px 40px;
        border-radius: 12px;
        border: 1px solid #333;
        backdrop-filter: blur(10px);
    }
    
    .search-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Genre filter tabs */
    .genre-tabs {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 30px 40px;
        flex-wrap: wrap;
    }
    
    .genre-tab {
        background: rgba(255,255,255,0.1);
        color: #e5e5e5;
        padding: 10px 20px;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid transparent;
    }
    
    .genre-tab:hover {
        background: rgba(229,9,20,0.2);
        color: #ffffff;
        border-color: #e50914;
    }
    
    .genre-tab.active {
        background: linear-gradient(45deg, #e50914, #f40612);
        color: #ffffff;
        font-weight: bold;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #141414 0%, #0a0a0a 100%);
        border-right: 1px solid #333;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #e50914, #f40612);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(229,9,20,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(229,9,20,0.5);
        background: linear-gradient(45deg, #f40612, #e50914);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(20, 20, 20, 0.9);
        border: 1px solid #333;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Custom metrics */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #e50914;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #e50914;
        margin-bottom: 8px;
    }
    
    .metric-label {
        color: #b3b3b3;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Recommendation types */
    .rec-type-tabs {
        display: flex;
        gap: 10px;
        margin: 20px 40px;
        justify-content: center;
    }
    
    .rec-tab {
        padding: 12px 24px;
        background: rgba(255,255,255,0.1);
        color: #e5e5e5;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .rec-tab.active {
        background: linear-gradient(45deg, #e50914, #f40612);
        color: #ffffff;
        font-weight: bold;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #333;
        border-top: 4px solid #e50914;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds to allow debugging
def load_movie_data():
    """Load movie data from available files"""
    try:
        # Load movie features from feature engineering output
        features_df = pd.read_csv('data/dataset/movie_features_engineered.csv')
        movies_df = pd.read_csv('data/dataset/movies_clean.csv')
        
        # Extract features and metadata
        feature_columns = [col for col in features_df.columns if col not in ['id', 'title']]
        movie_features = features_df[feature_columns].values
        movie_ids = features_df['id'].values
        movie_titles = features_df['title'].values
        
        # Create metadata dictionary
        metadata = {
            'n_features': len(feature_columns),
            'n_movies': len(features_df)
        }
        
        return movie_features, movie_ids, movie_titles, metadata, movies_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.error("Make sure you have run feature_engineering.ipynb notebook first.")
        st.error("Required files: data/dataset/movie_features_engineered.csv and data/dataset/movies_clean.csv")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_data
def get_content_based_recommendations(_movie_features, movie_idx, top_k=10):
    """Get content-based recommendations using cosine similarity"""
    target_features = _movie_features[movie_idx].reshape(1, -1)
    similarities = cosine_similarity(target_features, _movie_features).flatten()
    
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    similar_scores = similarities[similar_indices]
    
    return similar_indices, similar_scores

@st.cache_resource
def load_collaborative_filtering():
    """Load collaborative filtering recommender"""
    if not COLLABORATIVE_AVAILABLE:
        return None
    
    try:
        # Try to load existing trained models
        if os.path.exists('algorithm/cf_recommender.pkl'):
            with open('algorithm/cf_recommender.pkl', 'rb') as f:
                recommender = pickle.load(f)
            if recommender.load_models():
                return recommender
        
        # If no trained models, create and train new ones
        recommender = CollaborativeFilteringRecommender()
        if recommender.load_data():
            recommender.train_models()
            recommender.save_models()
            return recommender
        
    except Exception as e:
        st.warning(f"Collaborative filtering not available: {e}")
    
    return None

def create_movie_card(movie_data, show_similarity=False, similarity_score=None, show_match=False, match_score=None):
    """Create a Netflix-style movie card"""
    
    # Convert pandas Series to dictionary for easier handling
    if hasattr(movie_data, 'to_dict'):
        movie_dict = movie_data.to_dict()
    elif isinstance(movie_data, dict):
        movie_dict = movie_data
    else:
        # Try to convert to dict
        movie_dict = dict(movie_data)
    
    # Extract movie information safely
    title = str(movie_dict.get('title', 'Unknown Movie')).strip()
    year = movie_dict.get('release_year', 'N/A')
    rating = movie_dict.get('vote_average', 0)
    genres = movie_dict.get('genres_str', '')
    
    # Process genres
    if isinstance(genres, str) and genres:
        genres_list = genres.split('|')
        main_genre = genres_list[0] if genres_list[0] else 'Unknown'
    else:
        main_genre = 'Unknown'
    
    # Ensure title is not empty
    if not title or title == '':
        title = 'Unknown Movie'
    
    # Create match score display
    match_display = ""
    if show_match and match_score:
        match_display = f'<div class="match-score">{match_score:.0%} Match</div>'
    elif show_similarity and similarity_score:
        match_display = f'<div class="match-score">{similarity_score:.0%} Match</div>'
    
    # Ensure year is properly formatted
    year_display = year if year != 'N/A' and year else 'N/A'
    rating_display = rating if rating and rating > 0 else 0
    
    # Get poster URL if API key is available
    movie_id = movie_dict.get('id', None)
    poster_url = None
    
    if TMDB_API_KEY and movie_id:
        poster_url = get_movie_poster_url(movie_id, title)
    
    # Create simple movie card without complex HTML
    card_html = f"""<div style="background: #181818; border-radius: 8px; margin: 10px 5px; padding: 15px; border: 1px solid #333; text-align: center;">
    <div style="font-size: 48px; margin-bottom: 15px;">🎬</div>
    <h4 style="color: white; font-size: 16px; margin: 0 0 8px 0; font-weight: bold;">{title}</h4>
    <div style="color: #ccc; font-size: 14px; margin-bottom: 5px;">{year_display}</div>
    <div style="color: #ffd700; font-size: 14px; margin-bottom: 5px;">⭐ {rating_display:.1f}</div>
    <div style="color: #999; font-size: 12px;">{main_genre}</div>
    </div>"""
    return card_html

def create_movie_carousel(movies_data, title, show_match=False):
    """Create a horizontal movie carousel"""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    
    # Display movies in rows of 5
    for i in range(0, min(len(movies_data), 20), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(movies_data):
                movie_data = movies_data[i + j]
                match_score = movie_data.get('predicted_score', 0) / 10 if show_match else None
                with col:
                    st.markdown(
                        create_movie_card(movie_data, show_match=show_match, match_score=match_score),
                        unsafe_allow_html=True
                    )

def main():
    """Main application"""
    # Load data
    movie_features, movie_ids, movie_titles, metadata, movies_df = load_movie_data()
    cf_recommender = load_collaborative_filtering()
    
    # Create movie lookup dictionaries
    title_to_index = {title: idx for idx, title in enumerate(movie_titles)}
    
    # Debug info to check data loading
    st.sidebar.write("🔍 **Debug Info:**")
    st.sidebar.write(f"Movies loaded: {len(movies_df)}")
    st.sidebar.write(f"Features loaded: {len(movie_titles)}")
    st.sidebar.write("**Sample movie titles:**")
    for i, title in enumerate(movie_titles[:5]):
        st.sidebar.write(f"{i+1}. {title}")
    
    st.sidebar.write("**Movies DF columns:**")
    st.sidebar.write(list(movies_df.columns))
    
    st.sidebar.write("**Sample from movies_df:**")
    if len(movies_df) > 0:
        st.sidebar.write(f"First movie: {movies_df.iloc[0]['title']}")
    
    # Top Navigation
    st.markdown("""
    <div class="top-nav">
        <div class="logo">GoodMovie</div>
        <div class="nav-links">
            <a href="#" class="nav-link active">Home</a>
            <a href="#" class="nav-link">Movies</a>
            <a href="#" class="nav-link">Series</a>
            <a href="#" class="nav-link">Top Rated</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">GoodMovie</div>
        <div class="hero-subtitle">AI-Powered Movie Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Section
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">Find Your Perfect Movie</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "Select a movie to get recommendations:",
            options=[""] + list(movie_titles),
            key="movie_search",
            help="Choose a movie you like to get similar recommendations"
        )
    
    with col2:
        num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    with col3:
        rec_type = "Content-Based"  # Simplified to content-based only
        st.write("**Recommendation method:** Content-Based Filtering")
        st.write("*Finds movies similar to your selection*")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Genre Filter Tabs
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation']
    genre_tabs_html = '<div class="genre-tabs">'
    for genre in genres:
        genre_tabs_html += f'<div class="genre-tab">{genre}</div>'
    genre_tabs_html += '</div>'
    st.markdown(genre_tabs_html, unsafe_allow_html=True)
    
    # Main content based on selection
    if selected_movie and selected_movie != "":
        movie_idx = title_to_index[selected_movie]
        selected_movie_data = movies_df.iloc[movie_idx]
        
        # Display selected movie
        st.markdown('<div class="top-10-header"><div class="top-10-badge">Selected</div>Your Movie Choice</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col3:
            st.markdown(create_movie_card(selected_movie_data), unsafe_allow_html=True)
        
        # Get content-based recommendations
        similar_indices, similar_scores = get_content_based_recommendations(
            movie_features, movie_idx, num_recs
        )
        
        recommendations = []
        for i, (idx, score) in enumerate(zip(similar_indices, similar_scores)):
            movie_data = movies_df.iloc[idx].to_dict()  # Convert Series to dict
            movie_data['predicted_score'] = score * 10  # Convert to 0-10 scale
            recommendations.append(movie_data)
        
        create_movie_carousel(recommendations, "🎯 Movies You'll Love", show_match=True)
    
    else:
        # Show default content when no movie is selected
        
        # TOP 10 Section (matching UI reference)
        st.markdown("""
        <div class="top-10-header">
            <div class="top-10-badge">TOP 10</div>
            Content Today
        </div>
        """, unsafe_allow_html=True)
        
        # Get top-rated movies
        top_movies = movies_df.nlargest(10, 'vote_average')
        top_movies_list = [top_movies.iloc[i].to_dict() for i in range(len(top_movies))]
        
        # Display in carousel format
        for i in range(0, min(10, len(top_movies_list)), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                if i + j < len(top_movies_list):
                    movie_data = top_movies_list[i + j]
                    with col:
                        st.markdown(create_movie_card(movie_data), unsafe_allow_html=True)
        
        # Trending Today section
        trending_movies = movies_df.nlargest(10, 'popularity')
        trending_list = [trending_movies.iloc[i].to_dict() for i in range(len(trending_movies))]
        create_movie_carousel(trending_list, "🔥 Trending Today")
        
        # Top Rated section
        top_rated = movies_df[movies_df['vote_count'] >= 500].nlargest(15, 'vote_average')
        top_rated_list = [top_rated.iloc[i].to_dict() for i in range(len(top_rated))]
        create_movie_carousel(top_rated_list, "⭐ Top Rated")
        
        # Genres section (matching UI reference)
        st.markdown('<div class="section-header">🎭 Genres</div>', unsafe_allow_html=True)
        
        # Get movies by popular genres
        popular_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Science Fiction']
        for genre in popular_genres[:2]:  # Show first 2 genres
            genre_movies = movies_df[movies_df['genres_str'].str.contains(genre, na=False)]
            if len(genre_movies) >= 5:
                genre_movies = genre_movies.nlargest(10, 'vote_average')
                genre_list = [genre_movies.iloc[i].to_dict() for i in range(min(10, len(genre_movies)))]
                create_movie_carousel(genre_list, f"🎬 {genre}")
    
    # Sidebar with stats
    with st.sidebar:
        st.markdown("### 📊 Platform Stats")
        
        # Create custom metric cards
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(movies_df):,}</div>
            <div class="metric-label">Movies</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">{movies_df['vote_average'].mean():.1f}</div>
            <div class="metric-label">Avg Rating</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">{metadata['n_features']:,}</div>
            <div class="metric-label">AI Features</div>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status
        st.markdown("### 🔑 API Status")
        if TMDB_API_KEY:
            st.success("✅ TMDB API Connected")
            st.caption("Movie posters enabled")
        else:
            st.warning("⚠️ TMDB API Not Configured")
            st.caption("Add your API key to the `api` file")
            with st.expander("📋 Setup Instructions"):
                st.markdown("""
                1. Get API key from [TMDB](https://www.themoviedb.org/settings/api)
                2. Add key to `api` file in the app directory
                3. Restart the app
                """)
        
        # Debug info
        if st.checkbox("🔍 Debug Info"):
            sample_movie = movies_df.iloc[0]
            st.write(f"Sample movie ID: {sample_movie['id']}")
            st.write(f"Sample title: {sample_movie['title']}")
            if TMDB_API_KEY:
                poster_url = get_movie_poster_url(sample_movie['id'], sample_movie['title'])
                st.write(f"Poster URL: {poster_url}")
            else:
                st.write("Poster URL: Not available (no API key)")
        
        st.markdown("### 🎯 Recommendation Engine")
        st.write("• Content-based filtering")
        st.write("• Advanced feature analysis") 
        if cf_recommender:
            st.write("• Collaborative filtering")
            st.write("• Hybrid recommendations")
        
        st.markdown("### 🚀 Powered By")
        st.write("• Machine Learning")
        st.write("• Cosine Similarity")
        st.write("• Matrix Factorization")
        st.write("• Netflix-style UI")

if __name__ == "__main__":
    main()