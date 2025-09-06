#!/usr/bin/env python3
"""
TF-IDF Content-Based Movie Search Engine

Specialized TF-IDF search algorithm for intelligent movie search functionality.
Optimized for real-time search with semantic understanding and fuzzy matching.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import difflib
from functools import lru_cache
import streamlit as st


class TFIDFMovieSearch:
    """
    TF-IDF-based movie search engine with semantic understanding and fuzzy matching.
    Optimized for real-time search performance and intelligent query interpretation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TF-IDF search engine.
        
        Args:
            df: Movie DataFrame with columns: id, title, genre, overview, etc.
        """
        self.df = df.copy()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_corpus = None
        self._genre_cache = {}
        self._search_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the search engine
        self._build_search_index()
        
    def _build_search_index(self) -> None:
        """Build TF-IDF search index from movie corpus."""
        try:
            self.logger.info("Building TF-IDF search index...")
            
            # Create comprehensive text corpus for each movie
            self.processed_corpus = []
            
            for _, movie in self.df.iterrows():
                # Combine multiple fields for comprehensive search
                text_components = []
                
                # Title (highest weight - repeat 3 times)
                title = self._clean_text(str(movie.get('title', '')))
                text_components.extend([title] * 3)
                
                # Genres (high weight - repeat 2 times)
                genres = self._process_genres(str(movie.get('genre', '')))
                text_components.extend([genres] * 2)
                
                # Overview/description (moderate weight)
                overview = self._clean_text(str(movie.get('overview', '')))
                if overview:
                    text_components.append(overview)
                
                # Additional searchable fields
                for field in ['tagline', 'original_title']:
                    if field in movie and pd.notna(movie[field]):
                        field_text = self._clean_text(str(movie[field]))
                        text_components.append(field_text)
                
                # Keywords from title for franchise matching
                title_keywords = self._extract_title_keywords(title)
                text_components.append(title_keywords)
                
                # Combine all text components
                combined_text = ' '.join(filter(None, text_components))
                self.processed_corpus.append(combined_text)
            
            # Create TF-IDF vectorizer with optimized parameters for movie search
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Balanced for performance and accuracy
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better phrase matching
                min_df=1,  # Include rare terms for specific movie titles
                max_df=0.8,  # Remove overly common terms
                lowercase=True,
                strip_accents='ascii',
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens, min 2 chars
            )
            
            # Fit and transform the corpus
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_corpus)
            
            self.logger.info(f"TF-IDF index built: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"Error building TF-IDF search index: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for TF-IDF processing."""
        if not text or text == 'nan':
            return ''
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Handle common abbreviations and expansions
        expansions = {
            'sci fi': 'science fiction',
            'scifi': 'science fiction',
            'rom com': 'romantic comedy',
            'romcom': 'romantic comedy',
        }
        
        text_lower = text.lower()
        for abbrev, expansion in expansions.items():
            text_lower = text_lower.replace(abbrev, expansion)
        
        return text_lower
    
    def _process_genres(self, genre_str: str) -> str:
        """Process genre string for better matching."""
        if not genre_str or genre_str == 'nan':
            return ''
        
        # Split and clean genres
        genres = [self._clean_text(g.strip()) for g in str(genre_str).split(',') if g.strip()]
        
        # Expand genre synonyms
        genre_expansions = {
            'action': 'action adventure thriller',
            'comedy': 'comedy funny humor',
            'drama': 'drama dramatic',
            'horror': 'horror scary fear',
            'romance': 'romance romantic love',
            'thriller': 'thriller suspense mystery',
            'adventure': 'adventure action',
            'fantasy': 'fantasy magical supernatural',
            'crime': 'crime criminal detective',
            'family': 'family kids children',
        }
        
        expanded_genres = []
        for genre in genres:
            expanded_genres.append(genre)
            if genre.lower() in genre_expansions:
                expanded_genres.append(genre_expansions[genre.lower()])
        
        return ' '.join(expanded_genres)
    
    def _extract_title_keywords(self, title: str) -> str:
        """Extract key terms from movie title for franchise/series matching."""
        if not title:
            return ''
        
        # Remove common subtitle patterns
        title = re.sub(r':\s*part\s+\d+', '', title, flags=re.IGNORECASE)
        title = re.sub(r':\s*chapter\s+\d+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+\d+$', '', title)  # Remove trailing numbers
        title = re.sub(r'\s+\(\d{4}\)$', '', title)  # Remove year in parentheses
        
        # Extract meaningful keywords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower() for word in title.split() if word.lower() not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    @lru_cache(maxsize=512)  # Cache search results for performance
    def search_movies(self, query: str, top_k: int = 50, min_similarity: float = 0.1) -> pd.DataFrame:
        """
        Search movies using TF-IDF similarity with intelligent query processing.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            DataFrame with search results sorted by relevance
        """
        if not query or len(query.strip()) < 2:
            return pd.DataFrame()
        
        try:
            # Process query for better matching
            processed_query = self._process_search_query(query)
            
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top matches above minimum threshold
            match_indices = np.where(similarities >= min_similarity)[0]
            
            if len(match_indices) == 0:
                # Try fuzzy search as fallback
                return self._fuzzy_search_fallback(query, top_k)
            
            # Sort by similarity
            sorted_indices = match_indices[np.argsort(similarities[match_indices])[::-1]]
            top_indices = sorted_indices[:top_k]
            
            # Create result DataFrame
            result_df = self.df.iloc[top_indices].copy()
            result_df['search_similarity'] = similarities[top_indices]
            result_df['search_rank'] = range(1, len(result_df) + 1)
            
            # Apply post-processing for better ranking
            result_df = self._post_process_results(result_df, query)
            
            self.logger.info(f"TF-IDF search: '{query}' -> {len(result_df)} results")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in TF-IDF search: {e}")
            return pd.DataFrame()
    
    def _process_search_query(self, query: str) -> str:
        """Process and expand search query for better matching."""
        # Clean the query
        processed_query = self._clean_text(query)
        
        # Query expansions for common search patterns
        query_expansions = {
            'action movie': 'action adventure thriller',
            'funny movie': 'comedy funny humor',
            'scary movie': 'horror scary fear',
            'romantic movie': 'romance romantic love',
            'kids movie': 'family kids children animation',
            'superhero': 'superhero action adventure comic',
            'disney': 'disney animation family kids',
            'marvel': 'marvel superhero action adventure',
            'star wars': 'star wars space science fiction',
            'harry potter': 'harry potter fantasy magical wizard',
        }
        
        # Apply expansions
        for pattern, expansion in query_expansions.items():
            if pattern in processed_query.lower():
                processed_query += f' {expansion}'
        
        # Add genre-specific expansions
        for genre in ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller']:
            if genre in processed_query.lower():
                processed_query += f' {self._process_genres(genre)}'
        
        return processed_query
    
    def _post_process_results(self, result_df: pd.DataFrame, original_query: str) -> pd.DataFrame:
        """Post-process search results for better ranking."""
        if result_df.empty:
            return result_df
        
        # Calculate additional scoring factors
        result_df = result_df.copy()
        
        # Exact title match bonus
        query_lower = original_query.lower().strip()
        result_df['exact_match_bonus'] = result_df['title'].apply(
            lambda title: 0.5 if query_lower in title.lower() else 0
        )
        
        # Popularity boost for ambiguous queries
        if 'popularity' in result_df.columns:
            max_pop = result_df['popularity'].max()
            if max_pop > 0:
                result_df['popularity_boost'] = (result_df['popularity'] / max_pop) * 0.2
            else:
                result_df['popularity_boost'] = 0
        else:
            result_df['popularity_boost'] = 0
        
        # Quality boost
        if 'vote_average' in result_df.columns and 'vote_count' in result_df.columns:
            # Weighted rating considering both average and count
            result_df['quality_boost'] = (
                (result_df['vote_average'] / 10.0) * 0.7 +
                np.log1p(result_df['vote_count']) / 10.0 * 0.3
            ) * 0.1
        else:
            result_df['quality_boost'] = 0
        
        # Calculate final score
        result_df['final_score'] = (
            result_df['search_similarity'] +
            result_df['exact_match_bonus'] +
            result_df['popularity_boost'] +
            result_df['quality_boost']
        )
        
        # Re-sort by final score
        result_df = result_df.sort_values('final_score', ascending=False)
        result_df['search_rank'] = range(1, len(result_df) + 1)
        
        return result_df
    
    def _fuzzy_search_fallback(self, query: str, top_k: int) -> pd.DataFrame:
        """Fallback fuzzy search for when TF-IDF returns no results."""
        try:
            matches = []
            query_lower = query.lower().strip()
            
            for idx, title in enumerate(self.df['title']):
                title_lower = title.lower()
                
                # Calculate string similarity
                similarity = difflib.SequenceMatcher(None, query_lower, title_lower).ratio()
                
                # Check for substring matches
                if query_lower in title_lower or any(word in title_lower for word in query_lower.split() if len(word) > 2):
                    similarity += 0.3  # Boost for substring matches
                
                if similarity > 0.3:  # Lower threshold for fuzzy matching
                    matches.append((idx, similarity))
            
            if matches:
                # Sort by similarity and take top results
                matches.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in matches[:top_k]]
                
                result_df = self.df.iloc[top_indices].copy()
                result_df['search_similarity'] = [sim for _, sim in matches[:top_k]]
                result_df['search_rank'] = range(1, len(result_df) + 1)
                
                self.logger.info(f"Fuzzy search fallback: '{query}' -> {len(result_df)} results")
                return result_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in fuzzy search fallback: {e}")
            return pd.DataFrame()
    
    def get_search_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        if len(partial_query) < 2:
            return []
        
        suggestions = set()
        query_lower = partial_query.lower()
        
        # Movie title suggestions
        for title in self.df['title']:
            title_lower = title.lower()
            if query_lower in title_lower:
                suggestions.add(title)
                if len(suggestions) >= max_suggestions * 2:
                    break
        
        # Genre suggestions
        unique_genres = set()
        for genre_str in self.df['genre'].dropna():
            for genre in str(genre_str).split(','):
                genre = genre.strip()
                if genre and query_lower in genre.lower():
                    unique_genres.add(genre)
        
        suggestions.update(list(unique_genres)[:3])
        
        return sorted(list(suggestions))[:max_suggestions]
    
    def get_trending_searches(self) -> List[str]:
        """Get trending/popular search terms based on movie data."""
        trending = []
        
        # Popular genres
        if 'genre' in self.df.columns:
            genre_counts = {}
            for genre_str in self.df['genre'].dropna():
                for genre in str(genre_str).split(','):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            trending.extend([genre for genre, _ in top_genres])
        
        # Popular franchises/keywords from titles
        popular_keywords = ['Marvel', 'Star Wars', 'Disney', 'DC', 'Batman', 'Spider-Man']
        for keyword in popular_keywords:
            if any(keyword.lower() in title.lower() for title in self.df['title']):
                trending.append(keyword)
        
        return trending[:10]
    
    def clear_cache(self) -> None:
        """Clear search cache."""
        self._search_cache.clear()
        # Clear LRU cache for search_movies method
        self.search_movies.cache_clear()
        self.logger.info("Search cache cleared")


# Factory function for easy integration
@st.cache_resource
def create_tfidf_search_engine(df: pd.DataFrame) -> TFIDFMovieSearch:
    """
    Factory function to create and cache TF-IDF search engine.
    
    Args:
        df: Movie DataFrame
        
    Returns:
        Initialized TFIDFMovieSearch instance
    """
    return TFIDFMovieSearch(df)


def search_movies_tfidf(df: pd.DataFrame, query: str, max_results: int = 20) -> pd.DataFrame:
    """
    Convenient function for TF-IDF movie search.
    
    Args:
        df: Movie DataFrame
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        DataFrame with search results
    """
    search_engine = create_tfidf_search_engine(df)
    return search_engine.search_movies(query, top_k=max_results)