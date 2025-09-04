"""
User Management System for Movie Recommendation App
Simple username-only authentication with file-based storage
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class UserManager:
    """Simple user management system with file-based storage"""
    
    def __init__(self, users_file: str = "users_data.json"):
        self.users_file = users_file
        self.users_data = self._load_users_data()
    
    def _load_users_data(self) -> Dict:
        """Load users data from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Default structure
        return {
            "users": {},  # {username: {ratings: {}, created_at: "", last_login: ""}}
            "metadata": {
                "total_users": 0,
                "created_at": datetime.now().isoformat()
            }
        }
    
    def _save_users_data(self) -> bool:
        """Save users data to JSON file"""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving users data: {e}")
            return False
    
    def create_user(self, username: str) -> bool:
        """
        Create a new user account
        
        Args:
            username: Username (must be unique)
            
        Returns:
            bool: True if user created successfully, False if username exists
        """
        username = username.strip().lower()
        
        if not username or len(username) < 2:
            return False
        
        if self.user_exists(username):
            return False
        
        # Create new user
        self.users_data["users"][username] = {
            "ratings": {},
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "total_movies_rated": 0
        }
        
        self.users_data["metadata"]["total_users"] += 1
        
        return self._save_users_data()
    
    def user_exists(self, username: str) -> bool:
        """Check if username exists"""
        return username.strip().lower() in self.users_data["users"]
    
    def login_user(self, username: str) -> bool:
        """
        Login user (update last login time)
        
        Args:
            username: Username to login
            
        Returns:
            bool: True if login successful, False if user doesn't exist
        """
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return False
        
        # Update last login
        self.users_data["users"][username]["last_login"] = datetime.now().isoformat()
        self._save_users_data()
        
        return True
    
    def get_user_ratings(self, username: str) -> Dict[str, int]:
        """Get all ratings for a specific user"""
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return {}
        
        return self.users_data["users"][username]["ratings"].copy()
    
    def save_user_rating(self, username: str, movie_id: str, rating: int) -> bool:
        """
        Save a rating for a specific user
        
        Args:
            username: Username
            movie_id: Movie ID
            rating: Rating (1-10)
            
        Returns:
            bool: True if saved successfully
        """
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return False
        
        if not (1 <= rating <= 10):
            return False
        
        # Save rating
        old_rating = self.users_data["users"][username]["ratings"].get(str(movie_id))
        self.users_data["users"][username]["ratings"][str(movie_id)] = rating
        
        # Update total count if it's a new rating
        if old_rating is None:
            self.users_data["users"][username]["total_movies_rated"] += 1
        
        return self._save_users_data()
    
    def remove_user_rating(self, username: str, movie_id: str) -> bool:
        """Remove a rating for a specific user"""
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return False
        
        if str(movie_id) in self.users_data["users"][username]["ratings"]:
            del self.users_data["users"][username]["ratings"][str(movie_id)]
            self.users_data["users"][username]["total_movies_rated"] -= 1
            return self._save_users_data()
        
        return False
    
    def get_user_stats(self, username: str) -> Dict:
        """Get statistics for a specific user"""
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return {}
        
        user_data = self.users_data["users"][username]
        ratings = user_data["ratings"]
        
        if not ratings:
            return {
                "username": username,
                "total_rated": 0,
                "average_rating": 0.0,
                "created_at": user_data["created_at"],
                "last_login": user_data["last_login"]
            }
        
        rating_values = list(ratings.values())
        
        return {
            "username": username,
            "total_rated": len(ratings),
            "average_rating": round(sum(rating_values) / len(rating_values), 1),
            "highest_rating": max(rating_values),
            "lowest_rating": min(rating_values),
            "created_at": user_data["created_at"],
            "last_login": user_data["last_login"]
        }
    
    def get_all_users(self) -> List[str]:
        """Get list of all usernames"""
        return list(self.users_data["users"].keys())
    
    def get_users_count(self) -> int:
        """Get total number of users"""
        return len(self.users_data["users"])
    
    def delete_user(self, username: str) -> bool:
        """Delete a user account"""
        username = username.strip().lower()
        
        if not self.user_exists(username):
            return False
        
        del self.users_data["users"][username]
        self.users_data["metadata"]["total_users"] -= 1
        
        return self._save_users_data()
    
    def get_system_stats(self) -> Dict:
        """Get system-wide statistics"""
        total_ratings = 0
        total_users = len(self.users_data["users"])
        
        for user_data in self.users_data["users"].values():
            total_ratings += len(user_data["ratings"])
        
        return {
            "total_users": total_users,
            "total_ratings": total_ratings,
            "average_ratings_per_user": round(total_ratings / total_users, 1) if total_users > 0 else 0,
            "system_created": self.users_data["metadata"]["created_at"]
        }


# Global instance
user_manager = UserManager()


def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    return user_manager