import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import json
import os
from algorithm.content_based_recommender import ContentBasedRecommender, RecommendationConfig

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
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #333;
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
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
        height: 300px;
        background-color: #333;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
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
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .search-suggestions {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .match-button {
        background-color: #2a2a2a;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.2rem;
        color: #fff;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .match-button:hover {
        background-color: #ff4444;
        border-color: #ff4444;
    }
    
    .search-hint {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Initialize and cache the content-based recommender"""
    try:
        config = RecommendationConfig(
            dataset_path='dataset/movies.csv',
            similarity_method="cosine",
            default_n_recommendations=8,
            default_min_rating=0.0,
            default_min_votes=0,
            fuzzy_search_threshold=70
        )
        recommender = ContentBasedRecommender(config)
        return recommender
    except Exception as e:
        st.error(f"Failed to load recommender: {e}")
        return None

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

def fuzzy_search_movies(df, search_query, top_n=5):
    """
    Fuzzy search for movies using multiple matching strategies
    Returns list of (movie_title, similarity_score) tuples
    """
    import difflib
    
    if not search_query or len(search_query.strip()) < 2:
        return []
    
    search_query = search_query.lower().strip()
    matches = []
    
    for _, movie in df.iterrows():
        title = str(movie['title']).lower()
        
        # Strategy 1: Exact substring match (highest priority)
        if search_query in title:
            # Calculate how much of the title the query covers
            coverage = len(search_query) / len(title) * 100
            similarity_score = min(95 + coverage, 100)  # Bonus for exact matches
            matches.append((movie['title'], similarity_score))
            continue
        
        # Strategy 2: Word-based matching
        search_words = search_query.split()
        title_words = title.split()
        
        word_matches = 0
        for search_word in search_words:
            for title_word in title_words:
                if search_word in title_word or title_word in search_word:
                    word_matches += 1
                    break
        
        if word_matches > 0:
            word_score = (word_matches / len(search_words)) * 85
            matches.append((movie['title'], word_score))
            continue
        
        # Strategy 3: Fuzzy string matching (for typos)
        similarity = difflib.SequenceMatcher(None, search_query, title).ratio()
        if similarity > 0.4:  # Threshold for fuzzy matches
            fuzzy_score = similarity * 70  # Max 70 for fuzzy matches
            matches.append((movie['title'], fuzzy_score))
        
        # Strategy 4: Check if search query words appear in any order
        if len(search_words) > 1:
            all_words_found = all(any(sword in tword for tword in title_words) for sword in search_words)
            if all_words_found:
                partial_score = 60  # Lower score for partial matches
                # Check if we already added this movie with a higher score
                existing_match = next((m for m in matches if m[0] == movie['title']), None)
                if not existing_match or existing_match[1] < partial_score:
                    if existing_match:
                        matches.remove(existing_match)
                    matches.append((movie['title'], partial_score))
    
    # Sort by similarity score (descending) and return top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_n]

def display_movie_card(movie, show_rating=True):
    with st.container():
        # Get poster URL if movie has an ID
        poster_url = None
        if 'id' in movie and pd.notna(movie['id']):
            poster_url = fetch_poster(int(movie['id']))
        
        # Display poster if available
        if poster_url and not poster_url.endswith('No+Poster'):
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown('<div class="poster-placeholder">No Poster Available</div>', unsafe_allow_html=True)
        
        # Load user ratings
        user_ratings = load_user_ratings()
        movie_id = str(movie['id']) if 'id' in movie and pd.notna(movie['id']) else str(movie['title'])
        
        # Movie details
        st.markdown(f"""
        <div class="movie-card" style="padding-top: 0;">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-info"><span class="rating">⭐ {movie['vote_average']}</span> ({movie['vote_count']} votes)</div>
            <div class="movie-info">📅 {movie['release_date']}</div>
            <div class="movie-info">🌟 Popularity: {movie['popularity']:.1f}</div>
            <div style="margin-top: 0.5rem;">
                {' '.join([f'<span class="genre">{genre.strip()}</span>' for genre in str(movie['genre']).split(',')])}
            </div>
            <div class="movie-info" style="margin-top: 0.5rem; font-size: 0.8rem; flex-grow: 1;">
                {movie['overview'][:150]}{'...' if len(str(movie['overview'])) > 150 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rating system
        if show_rating:
            current_rating = user_ratings.get(movie_id, 0)
            user_rating = st.slider(
                "Rate this movie",
                min_value=0,
                max_value=10,
                value=current_rating,
                step=1,
                key=f"rating_{movie_id}",
                help="Rate from 1-10 (0 = not rated)"
            )
            
            if user_rating != current_rating:
                user_ratings[movie_id] = user_rating
                save_user_ratings(user_ratings)
                if user_rating > 0:
                    st.success(f"Rated {movie['title']}: {user_rating}/10")
            
            if user_rating > 0:
                st.markdown(f'<div class="rating-container"><span class="user-rating">Your Rating: {user_rating}/10</span></div>', unsafe_allow_html=True)

def get_content_based_recommendations(recommender, movie_title, n_recommendations=5):
    """Get content-based recommendations for a movie"""
    try:
        recommendations = recommender.get_movie_recommendations(
            movie_title, 
            n_recommendations=n_recommendations,
            min_rating=0.0,
            min_votes=0
        )
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def main():
    # Header
    st.markdown('<div class="top-header">CONTENT-BASED MOVIE RECOMMENDATIONS</div>', unsafe_allow_html=True)
    
    # Load data and recommender
    df = load_movie_data()
    recommender = load_recommender()
    
    if df.empty:
        st.error("No movie data available")
        return
    
    if recommender is None:
        st.error("Recommendation system not available")
        return
    
    # Recommendation section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Get Personalized Recommendations")
    
    rec_col1, rec_col2 = st.columns([3, 1])
    
    with rec_col1:
        search_query = st.text_input(
            "🎬 Search for a movie:",
            placeholder="Type movie name (e.g., 'spider', 'iron man', 'matrix')...",
            help="Start typing and we'll find the best match!"
        )
    
    with rec_col2:
        get_recs_button = st.button("Get Recommendations", type="primary")
    
    # Auto-select best match from search query
    selected_movie = ""
    if search_query:
        # Find matching movies using fuzzy search
        matching_movies = fuzzy_search_movies(df, search_query, top_n=1)
        
        if matching_movies:
            selected_movie = matching_movies[0][0]  # Best match
            st.info(f"🎯 Found: **{selected_movie}**")
        else:
            st.warning(f"No movies found matching '{search_query}'. Try different keywords!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle recommendations
    if (selected_movie and selected_movie != "") or get_recs_button:
        if selected_movie and selected_movie != "":
            with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
                recommendations = get_content_based_recommendations(recommender, selected_movie, n_recommendations=8)
            
            if recommendations:
                st.markdown(f'<div class="section-header">🎯 Movies Similar to "{selected_movie}"</div>', unsafe_allow_html=True)
                
                # Display recommendations in grid
                rec_cols = st.columns(4)
                for idx, rec in enumerate(recommendations):
                    with rec_cols[idx % 4]:
                        # Create a movie dict compatible with display_movie_card
                        movie_data = {
                            'id': rec.get('id', ''),
                            'title': rec['title'],
                            'vote_average': rec['rating'],
                            'vote_count': rec['votes'],
                            'release_date': rec.get('year', ''),
                            'popularity': rec.get('similarity', 0) * 100,  # Use similarity as popularity
                            'genre': rec.get('genres', ''),
                            'overview': f"Similarity Score: {rec.get('similarity', 0):.3f}"
                        }
                        display_movie_card(movie_data, show_rating=True)
            else:
                st.warning(f"No recommendations found for '{selected_movie}'. Try a different movie.")
        else:
            st.warning("Please select a movie first!")
    
    # Top 10 section (show when no recommendations are displayed)
    if not (selected_movie and selected_movie != ""):
        st.markdown('<div class="section-header">📈 Trending Today</div>', unsafe_allow_html=True)
        
        # Sort by vote_average and popularity for top movies
        top_movies = df.nlargest(10, ['vote_average', 'popularity'])
        
        # Display in a grid layout (5 columns for top 10)
        cols = st.columns(5)
        for idx, (_, movie) in enumerate(top_movies.head(5).iterrows()):
            with cols[idx]:
                display_movie_card(movie, show_rating=False)
        
        # Second row of top 10
        cols2 = st.columns(5)
        for idx, (_, movie) in enumerate(top_movies.tail(5).iterrows()):
            with cols2[idx]:
                display_movie_card(movie, show_rating=False)
    
        # Additional trending section
        st.markdown('<div class="section-header">🔥 More Trending</div>', unsafe_allow_html=True)
        
        # Show more movies in a different layout
        trending_movies = df.nlargest(20, 'popularity').tail(10)
        
        cols3 = st.columns(4)
        for idx, (_, movie) in enumerate(trending_movies.iterrows()):
            with cols3[idx % 4]:
                display_movie_card(movie, show_rating=False)
    
    # Sidebar info
    st.sidebar.title("🎬 My Ratings")
    user_ratings = load_user_ratings()
    if user_ratings:
        rated_movies = [movie_id for movie_id, rating in user_ratings.items() if rating > 0]
        st.sidebar.write(f"You have rated {len(rated_movies)} movies")
        
        if st.sidebar.button("Clear All Ratings"):
            save_user_ratings({})
            st.experimental_rerun()
    else:
        st.sidebar.write("No ratings yet. Rate some movies!")
    
    # Footer stats
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Get all genres for stats
    all_genres = set()
    for genres in df['genre'].dropna():
        all_genres.update([g.strip() for g in str(genres).split(',')])
    
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        st.metric("Average Rating", f"{df['vote_average'].mean():.1f}")
    with col3:
        st.metric("Genres Available", len(all_genres))
    with col4:
        st.metric("Results Shown", len(df))
    with col5:
        user_ratings = load_user_ratings()
        st.metric("Movies Rated", len([r for r in user_ratings.values() if r > 0]))

if __name__ == "__main__":
    main()