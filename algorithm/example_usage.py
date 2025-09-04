#!/usr/bin/env python3
"""
Example usage of the Content-Based Movie Recommender
"""

from content_based_recommender import ContentBasedRecommender, RecommendationConfig

def main():
    print("🎬 Enhanced Movie Recommender Demo")
    print("=" * 40)
    
    # Configure the recommender
    config = RecommendationConfig(
        similarity_method="cosine",
        default_n_recommendations=5,
        default_min_rating=6.0,
        fuzzy_search_threshold=75,
        cache_enabled=True
    )
    
    # Initialize recommender
    try:
        recommender = ContentBasedRecommender(config)
        recommender.load_data()
        print("✓ Recommender initialized successfully!")
        
        # Show dataset info
        stats = recommender.get_dataset_stats()
        print(f"\n📊 Dataset: {stats['total_movies']:,} movies, {stats['features']} features")
        
        # Example 1: Fuzzy search demonstration
        print(f"\n🔍 Example 1: Fuzzy Search")
        search_terms = ["avata", "dark night", "toy stor"]  # Intentionally misspelled
        
        for term in search_terms:
            try:
                matched_title, score, idx = recommender.fuzzy_search_movie(term)
                print(f"'{term}' → '{matched_title}' ({score}% match)")
            except Exception as e:
                print(f"'{term}' → {e}")
        
        # Example 2: Get recommendations
        print(f"\n🎯 Example 2: Recommendations for 'The Matrix'")
        try:
            recs = recommender.get_movie_recommendations(
                "The Matrix", 
                n_recommendations=3,
                min_rating=7.0
            )
            
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['title']} ({rec.get('year', 'N/A')})")
                print(f"   Rating: {rec['rating']:.1f} ⭐ | Similarity: {rec['similarity']:.3f}")
                if rec['genres']:
                    print(f"   Genres: {rec['genres']}")
                print()
                
        except Exception as e:
            print(f"Error: {e}")
        
        # Example 3: Genre recommendations
        print(f"🎭 Example 3: Top Sci-Fi Movies")
        try:
            sci_fi_recs = recommender.get_genre_recommendations(
                "Science Fiction",
                n_recommendations=3,
                min_rating=7.5
            )
            
            for i, rec in enumerate(sci_fi_recs, 1):
                print(f"{i}. {rec['title']} ({rec.get('year', 'N/A')}) - {rec['rating']:.1f} ⭐")
                
        except Exception as e:
            print(f"Available genres: {recommender.get_available_genres()}")
            print(f"Error: {e}")
        
        # Example 4: Batch recommendations
        print(f"\n🔄 Example 4: Batch Processing")
        try:
            # Get some movie IDs
            sample_ids = list(recommender.movies_df['id'].head(3))
            batch_results = recommender.get_similar_movies_batch(sample_ids, n_each=2)
            
            for movie_id, recs in batch_results.items():
                movie_title = recommender.movies_df[recommender.movies_df['id'] == movie_id]['title'].iloc[0]
                print(f"Similar to '{movie_title}': {len(recs)} recommendations")
                
        except Exception as e:
            print(f"Error: {e}")
            
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the dataset file exists")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Check that you're running from the correct directory")


if __name__ == "__main__":
    main()