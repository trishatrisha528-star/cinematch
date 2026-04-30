import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import json
import time

# Page configuration
st.set_page_config(
    page_title="🎬 CineMatch - Movie Recommender",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced movie-themed styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Title styling with enhanced glow */
    .main-title {
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FF6B6B, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5em;
        font-weight: 800;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        margin: 30px 0 10px 0;
        animation: glowPulse 3s ease-in-out infinite;
        letter-spacing: 2px;
    }
    
    @keyframes glowPulse {
        0%, 100% { 
            filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.6));
            transform: scale(1);
        }
        50% { 
            filter: drop-shadow(0 0 25px rgba(255, 107, 107, 0.8));
            transform: scale(1.02);
        }
    }
    
    /* Enhanced card styling */
    .movie-card {
        background: linear-gradient(135deg, rgba(42, 42, 42, 0.9) 0%, rgba(58, 58, 58, 0.9) 100%);
        border: 2px solid #FFD700;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.15), inset 0 0 10px rgba(255, 215, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(255, 215, 0, 0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(255, 215, 0, 0.3), inset 0 0 15px rgba(255, 215, 0, 0.1);
        border-color: #FF6B6B;
    }
    
    /* Title styling */
    .movie-title {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.6em;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    
    .movie-info {
        color: #E8E8E8;
        font-size: 0.98em;
        line-height: 1.8;
    }
    
    /* Genre badge with gradient */
    .genre-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 25px;
        margin: 4px;
        font-size: 0.85em;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.2);
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .genre-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    
    /* Rating styling */
    .rating-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        padding: 10px 18px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        margin: 5px 5px 5px 0;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .rating-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5);
    }
    
    /* Similarity meter */
    .similarity-meter {
        background: linear-gradient(90deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid #FFD700;
        border-radius: 15px;
        height: 12px;
        overflow: hidden;
        margin: 12px 0;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    
    .similarity-fill {
        background: linear-gradient(90deg, #00FF00 0%, #FFD700 50%, #FF6B6B 100%);
        height: 100%;
        border-radius: 15px;
        animation: fillFlow 2s ease-in-out;
    }
    
    @keyframes fillFlow {
        0% { width: 0%; }
        100% { width: var(--width); }
    }
    
    /* Sidebar styling */
    [data-testid="sidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        border-right: 3px solid #FFD700;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        font-size: 1em;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(42, 42, 42, 0.8);
        color: #FFD700;
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 12px;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF6B6B;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        background: rgba(42, 42, 42, 0.95);
    }
    
    /* Enhanced metric styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(42, 42, 42, 0.8) 0%, rgba(58, 58, 58, 0.8) 100%);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 18px;
        margin: 12px 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(255, 215, 0, 0.2);
        border-color: #FF6B6B;
    }
    
    .metric-value {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2em;
        font-weight: 800;
    }
    
    .metric-label {
        color: #B0B0B0;
        font-size: 0.9em;
        margin-top: 5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Recommendation number styling */
    .rec-number {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B6B, #FFD700);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: 800;
        font-size: 1.2em;
        margin-right: 10px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    /* Divider styling */
    hr {
        border: 1px solid rgba(255, 215, 0, 0.3);
        margin: 20px 0;
    }
    
    /* Landing page styling */
    .landing-container {
        text-align: center;
        margin-top: 60px;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .landing-title {
        background: linear-gradient(135deg, #FFD700, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2em;
        font-weight: 800;
        margin: 20px 0;
    }
    
    .landing-subtitle {
        color: #B0B0B0;
        font-size: 1.2em;
        margin: 20px 0;
    }
    
    /* Suggestion box */
    .suggestion-box {
        margin-top: 40px;
        padding: 25px;
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.08) 0%, rgba(255, 107, 107, 0.08) 100%);
        border: 2px solid #FFD700;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .suggestion-title {
        color: #FFD700;
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 15px;
    }
    
    .suggestion-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00FF00, #FFD700, #FF6B6B);
    }
    </style>
    """, unsafe_allow_html=True)

# Session state for caching
@st.cache_resource
def load_model_and_data():
    """Load the trained model and data"""
    # Load the TMDB movies dataset
    df = pd.read_csv('data/tmdb_5000_movies.csv')
    
    # Clean data
    df_clean = df.dropna(subset=['genres', 'vote_average', 'budget', 'revenue'])
    
    # Parse genres
    def extract_genres(genres_str):
        try:
            genres_list = json.loads(genres_str)
            return [genre['name'] for genre in genres_list]
        except:
            return []
    
    df_clean['genres_list'] = df_clean['genres'].apply(extract_genres)
    
    # Create genre columns
    all_genres = set()
    for genres_list in df_clean['genres_list']:
        all_genres.update(genres_list)
    
    for genre in all_genres:
        df_clean[f'genre_{genre}'] = df_clean['genres_list'].apply(lambda x: 1 if genre in x else 0)
    
    # Prepare features
    numerical_features = ['vote_average', 'budget', 'revenue', 'runtime']
    genre_features = [col for col in df_clean.columns if col.startswith('genre_')]
    feature_columns = numerical_features + genre_features
    
    X = df_clean[feature_columns].copy()
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN model
    knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn_model.fit(X_scaled)
    
    return df_clean, knn_model, X_scaled, feature_columns, all_genres

def recommend_movies(df_clean, knn_model, X_scaled, feature_columns, movie_title, n_recommendations=5):
    """Get movie recommendations"""
    # Find the movie
    movie_idx = df_clean[df_clean['title'].str.lower() == movie_title.lower()].index
    
    if len(movie_idx) == 0:
        return None, None, None
    
    movie_idx = movie_idx[0]
    
    # Get recommendations
    movie_features = X_scaled[movie_idx].reshape(1, -1)
    distances, indices = knn_model.kneighbors(movie_features, n_neighbors=n_recommendations+1)
    
    recommendation_indices = indices[0][1:]
    recommendation_distances = distances[0][1:]
    
    recommendations = df_clean.iloc[recommendation_indices][['title', 'vote_average', 'release_date', 'runtime', 'genres_list']].copy()
    recommendations['distance'] = recommendation_distances
    recommendations = recommendations.reset_index(drop=True)
    
    query_movie = df_clean.iloc[movie_idx]
    
    return query_movie, recommendations, recommendation_indices

# Main app
st.markdown("<div class='main-title'>🎬 CINEMATCH</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFD700; font-size: 1.3em; font-weight: 600; letter-spacing: 1px;'>✨ Discover Your Next Favorite Movie ✨</p>", unsafe_allow_html=True)

# Load data
with st.spinner("🎥 Loading movie database..."):
    df_clean, knn_model, X_scaled, feature_columns, all_genres = load_model_and_data()

# Sidebar info
with st.sidebar:
    st.markdown("<div style='text-align: center; margin: 20px 0;'><span style='font-size: 2.5em;'>🎭</span></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFD700; font-size: 1.3em;'>📊 Database Stats</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value'>{len(df_clean):,}</div>
            <div class='metric-label'>🎬 Movies</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-value'>{len(all_genres)}</div>
            <div class='metric-label'>🎨 Genres</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3 style='color: #FFD700; font-size: 1.1em;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    num_recommendations = st.slider("📊 Number of Recommendations", 3, 15, 5, 
                                   help="Select how many recommendations you want to see")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #FFD700; font-size: 1.1em;'>💡 Quick Tips</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: rgba(255, 215, 0, 0.1); border-left: 4px solid #FFD700; padding: 12px; border-radius: 5px;'>
        <p style='color: #FFD700; margin: 5px 0;'>🎯 <strong>Tips:</strong></p>
        <ul style='color: #B0B0B0; margin: 10px 0; padding-left: 20px;'>
            <li>Try blockbuster movies</li>
            <li>Mix different genres</li>
            <li>Check ratings</li>
            <li>Lower distance = more similar</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("---")
st.markdown("<h2 style='color: #FFD700; text-align: center; font-size: 1.4em; margin: 30px 0;'>🔍 Search Your Movie</h2>", unsafe_allow_html=True)

col_search = st.columns([1])[0]
movie_input = st.text_input(
    "Enter a movie title:",
    placeholder="e.g., The Avengers, Titanic, Inception...",
    label_visibility="collapsed"
)

# Search functionality
if movie_input:
    with st.spinner("🎬 Searching for similar movies..."):
        query_movie, recommendations, rec_indices = recommend_movies(
            df_clean, knn_model, X_scaled, feature_columns, movie_input, num_recommendations
        )
    
    if query_movie is None:
        st.error(f"❌ Movie '{movie_input}' not found in database. Please try another title!")
        st.markdown("""
        <div style='text-align: center; margin-top: 30px;'>
            <p style='color: #B0B0B0; font-size: 1.1em;'>💡 Try one of these popular titles:</p>
        </div>
        """, unsafe_allow_html=True)
        
        popular_movies = ["Avatar", "The Avengers", "Titanic", "Inception", "The Shawshank Redemption"]
        cols = st.columns(5)
        for i, movie in enumerate(popular_movies):
            with cols[i]:
                if st.button(f"🎬 {movie}", use_container_width=True):
                    st.rerun()
    else:
        # Display query movie info with enhanced styling
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 107, 107, 0.1)); 
                    border: 2px solid #FFD700; border-radius: 15px; padding: 25px; margin: 20px 0;'>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='color: #FFD700; margin: 0 0 15px 0;'>📽️ Your Selection</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"<div class='movie-title' style='font-size: 1.8em;'>{query_movie['title']}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='rating-badge'>⭐ {query_movie['vote_average']:.1f}</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div style='color: #FFD700; font-weight: 600; text-align: center;'>⏱️<br>{query_movie['runtime']:.0f}m</div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div style='color: #FFD700; font-weight: 600; text-align: center;'>📅<br>{query_movie['release_date'][:4]}</div>""", unsafe_allow_html=True)
        
        # Query movie details
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style='background: rgba(255, 215, 0, 0.05); border-left: 4px solid #FFD700; padding: 12px; border-radius: 5px;'>
                <p style='color: #B0B0B0; margin: 5px 0; font-size: 0.9em;'><strong>Release Date:</strong></p>
                <p style='color: #FFD700; margin: 5px 0; font-weight: 600;'>{query_movie['release_date']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            budget_display = f"${query_movie['budget']/1e6:.0f}M"
            st.markdown(f"""
            <div style='background: rgba(255, 215, 0, 0.05); border-left: 4px solid #FFD700; padding: 12px; border-radius: 5px;'>
                <p style='color: #B0B0B0; margin: 5px 0; font-size: 0.9em;'><strong>Budget:</strong></p>
                <p style='color: #FFD700; margin: 5px 0; font-weight: 600;'>{budget_display}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            revenue_display = f"${query_movie['revenue']/1e9:.1f}B"
            st.markdown(f"""
            <div style='background: rgba(255, 215, 0, 0.05); border-left: 4px solid #FFD700; padding: 12px; border-radius: 5px;'>
                <p style='color: #B0B0B0; margin: 5px 0; font-size: 0.9em;'><strong>Revenue:</strong></p>
                <p style='color: #FFD700; margin: 5px 0; font-weight: 600;'>{revenue_display}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Genres
        genres_html = " ".join([f"<span class='genre-badge'>{g}</span>" for g in query_movie['genres_list']])
        st.markdown(f"<div style='margin-top: 15px;'><strong style='color: #FFD700;'>🎨 Genres:</strong><br>{genres_html}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("---")
        st.markdown(f"<h2 style='color: #FFD700; text-align: center; font-size: 1.5em; margin: 30px 0;'>🎯 Top {len(recommendations)} Similar Movies</h2>", unsafe_allow_html=True)
        
        # Animate recommendations
        for idx, (_, rec) in enumerate(recommendations.iterrows()):
            # Calculate similarity score
            max_distance = recommendations['distance'].max()
            normalized_similarity = 1 - (rec['distance'] / max_distance)
            
            st.markdown(f"""
            <div class='movie-card' style='animation: fadeInUp {0.3 + idx*0.1}s ease-out;'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span class='rec-number'>{idx+1}</span>
                    <div class='movie-title' style='margin: 0; flex: 1;'>{rec['title']}</div>
                    <span style='background: linear-gradient(135deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700; font-size: 1.1em;'>{normalized_similarity*100:.0f}% Match</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Movie details in grid
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div style='color: #B0B0B0; font-size: 0.9em;'><strong>⭐ Rating</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #FFD700; font-weight: 600; font-size: 1.1em;'>{rec['vote_average']:.1f}/10</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='color: #B0B0B0; font-size: 0.9em;'><strong>⏱️ Runtime</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #FFD700; font-weight: 600; font-size: 1.1em;'>{rec['runtime']:.0f} min</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div style='color: #B0B0B0; font-size: 0.9em;'><strong>📅 Release</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #FFD700; font-weight: 600; font-size: 1.1em;'>{rec['release_date'][:4]}</div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div style='color: #B0B0B0; font-size: 0.9em;'><strong>📏 Distance</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color: #FFD700; font-weight: 600; font-size: 1.1em;'>{rec['distance']:.3f}</div>", unsafe_allow_html=True)
            
            # Genres
            genres_html = " ".join([f"<span class='genre-badge'>{g}</span>" for g in rec['genres_list']])
            st.markdown(f"<div style='margin-top: 12px;'><strong style='color: #FFD700; font-size: 0.9em;'>🎨 Genres:</strong><br>{genres_html}</div>", unsafe_allow_html=True)
            
            # Similarity bar
            st.markdown(f"""
            <div class='similarity-meter' style='--width: {normalized_similarity*100}%;'>
                <div class='similarity-fill' style='width: {normalized_similarity*100}%;'></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Statistics footer
        st.markdown("---")
        st.markdown("<h3 style='color: #FFD700; text-align: center; font-size: 1.3em; margin: 20px 0;'>📊 Recommendation Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{recommendations['vote_average'].mean():.1f}</div>
                <div class='metric-label'>Avg Rating ⭐</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{recommendations['runtime'].mean():.0f}</div>
                <div class='metric-label'>Avg Runtime ⏱️</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{recommendations['distance'].min():.2f}</div>
                <div class='metric-label'>Best Match 🎯</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{len(recommendations)}</div>
                <div class='metric-label'>Recommendations 🎬</div>
            </div>
            """, unsafe_allow_html=True)

# Landing page when no search
if not movie_input:
    st.markdown("""
    <div class='landing-container'>
        <div style='margin-bottom: 40px;'>
            <div style='font-size: 4em; text-align: center; animation: bounce 2s infinite;'>🎥</div>
        </div>
        
        <h1 class='landing-title'>Welcome to CineMatch!</h1>
        <p class='landing-subtitle'>🎬 Powered by KNN Machine Learning</p>
        <p style='color: #B0B0B0; font-size: 1.05em; margin: 20px auto; max-width: 600px; line-height: 1.8;'>
            Discover your next favorite movie with our advanced AI recommendation engine. 
            Simply search for a movie you love, and we'll find perfect matches tailored to your taste.
        </p>
    </div>
    
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    </style>
    
    <div class='suggestion-box'>
        <div class='suggestion-title'>🎬 Popular Movies to Try</div>
        <div class='suggestion-grid'>
    """, unsafe_allow_html=True)
    
    popular_movies = [
        ("Avatar", "🚀"),
        ("The Avengers", "🦸"),
        ("Titanic", "🚢"),
        ("Inception", "💭"),
        ("The Shawshank Redemption", "🏆"),
        ("The Dark Knight", "🦇"),
        ("Jurassic World", "🦕"),
        ("Forrest Gump", "🏃")
    ]
    
    cols = st.columns(len(popular_movies))
    for i, (movie, emoji) in enumerate(popular_movies):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, rgba(255,107,107,0.2), rgba(255,215,0,0.2)); 
                        border-radius: 10px; cursor: pointer; transition: all 0.3s;'>
                <div style='font-size: 1.5em; margin-bottom: 5px;'>{emoji}</div>
                <div style='color: #FFD700; font-weight: 600; font-size: 0.9em;'>{movie}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("---")
    st.markdown("<h2 style='color: #FFD700; text-align: center; font-size: 1.5em; margin: 40px 0;'>✨ Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-container' style='height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 2.5em; text-align: center; margin-bottom: 10px;'>🤖</div>
            <div class='metric-label' style='color: #FFD700; font-size: 1.1em; margin-bottom: 5px;'>AI-Powered</div>
            <div style='color: #B0B0B0; text-align: center; font-size: 0.9em;'>Advanced KNN algorithm for intelligent recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-container' style='height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 2.5em; text-align: center; margin-bottom: 10px;'>📊</div>
            <div class='metric-label' style='color: #FFD700; font-size: 1.1em; margin-bottom: 5px;'>Data-Driven</div>
            <div style='color: #B0B0B0; text-align: center; font-size: 0.9em;'>Analyzes genres, ratings, runtime & more</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-container' style='height: 200px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 2.5em; text-align: center; margin-bottom: 10px;'>⚡</div>
            <div class='metric-label' style='color: #FFD700; font-size: 1.1em; margin-bottom: 5px;'>Lightning Fast</div>
            <div style='color: #B0B0B0; text-align: center; font-size: 0.9em;'>Get recommendations instantly</div>
        </div>
        """, unsafe_allow_html=True)
