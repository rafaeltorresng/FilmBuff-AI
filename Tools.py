import os
import re
import time
import requests
from typing import Dict, List, Any, Optional, Type, Union
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def make_tmdb_request(endpoint, params=None):
    if params is None:
        params = {}
    
    clean_params = {k: v for k, v in params.items() if v is not None and v != "None"}
    
    clean_params["api_key"] = TMDB_API_KEY
    
    try:
        response = requests.get(f"{TMDB_BASE_URL}/{endpoint}", params=clean_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        response_status = getattr(e.response, 'status_code', None)
        if response_status == 429:
            retry_after = int(e.response.headers.get("Retry-After", 1))
            time.sleep(retry_after)
            return make_tmdb_request(endpoint, params)
        
        print(f"API request error: {str(e)}")
        return {"error": str(e), "results": []}

@lru_cache(maxsize=128)
def get_genre_id(genre_name: str) -> Optional[int]:
    if not genre_name or genre_name == "None":
        return None
        
    genres = {
        "Action": 28, "Adventure": 12, "Animation": 16,
        "Comedy": 35, "Crime": 80, "Documentary": 99,
        "Drama": 18, "Family": 10751, "Fantasy": 14,
        "History": 36, "Horror": 27, "Mystery": 9648,
        "Romance": 10749, "Science Fiction": 878, "Sci-Fi": 878,
        "Thriller": 53, "War": 10752, "Western": 37
    }
    
    normalized_genre = genre_name.title()
    if normalized_genre in genres:
        return genres[normalized_genre]
    
    for key, value in genres.items():
        if key.lower() in genre_name.lower():
            return value
            
    return None

class IntentClassifierSchema(BaseModel):
    query: str = Field(description="The user query to classify")

class IntentClassifierTool(BaseTool):
    name: str = "intent_classifier_tool"
    description: str = "Classifies the intent of a user query"
    args_schema: Type[BaseModel] = IntentClassifierSchema
    
    def _run(self, query: str):
        query = query.lower()
        
        if any(pattern in query for pattern in ["tell me about", "information about", "details of", "details about", 
                                              "what is", "who is", "synopsis of", "plot of", 
                                              "describe", "rating of", "how long is", "when was", 
                                              "who directed", "who played", "what genre", "rated"]):
            
            if any(term in query for term in ["actor", "actress", "director", "who played", "cast of"]):
                specific_type = next((term for term in ["actor", "actress", "director", "who played", "cast"] 
                                    if term in query), "person")
                return {
                    "intent": "person_info",
                    "confidence": 0.9,
                    "target_agent": "information_agent",
                    "specific_query": True,
                    "query_type": specific_type
                }
            
            specific_queries = {
                "director": ["who directed", "who was the director", "director of"],
                "rating": ["what rating", "how well rated", "imdb rating", "rotten tomatoes", "score of"],
                "cast": ["who starred in", "who plays in", "actors in", "cast of"],
                "release": ["when was released", "release date", "when did it come out"],
                "duration": ["how long is", "runtime of", "duration of", "how many seasons"],
                "genre": ["what genre is", "which genre", "type of movie"]
            }
            
            for query_type, patterns in specific_queries.items():
                if any(pattern in query for pattern in patterns):
                    return {
                        "intent": "movie_info_specific",
                        "confidence": 0.95,
                        "target_agent": "information_agent",
                        "specific_query": True,
                        "query_type": query_type
                    }
            
            return {
                "intent": "movie_info_general",
                "confidence": 0.9,
                "target_agent": "information_agent",
                "specific_query": False
            }
                    
        if any(pattern in query for pattern in ["recommend", "suggestion", "similar to", "like", 
                                              "movies about", "shows about", "watch", "good movies", 
                                              "best movies", "movies with"]):
            return {
                "intent": "recommendation",
                "confidence": 0.85,
                "target_agent": "recommendation_agent"
            }
        
        if any(pattern in query for pattern in ["trending", "popular", "top rated", "best of", 
                                              "this week", "this month", "new releases", 
                                              "what's hot", "what is popular"]):
            return {
                "intent": "trends",
                "confidence": 0.95,
                "target_agent": "trends_agent"
            }
            
        return {
            "intent": "recommendation", 
            "confidence": 0.6,
            "target_agent": "recommendation_agent"
        }

class FetchMovieInfoSchema(BaseModel):
    query: str = Field(description="Movie or TV show title to search for")
    year: Optional[int] = Field(default=None, description="Release year (optional)")

class FetchMovieReviewsSchema(BaseModel):
    movie_id: int = Field(description="Movie or TV show ID to fetch reviews for")

class FetchMovieInfoTool(BaseTool):
    name: str = "fetch_movie_info"
    description: str = "Fetches detailed information about movies or TV shows by title"
    args_schema: Type[BaseModel] = FetchMovieInfoSchema
    
    def _run(self, query: str, year: Optional[int] = None):
        params = {"query": query, "language": "en-US"}
        if year:
            params["year"] = year
            
        search_data = make_tmdb_request("search/movie", params)
        results = search_data.get("results", [])
        
        if not results:
            search_data = make_tmdb_request("search/tv", params)
            results = search_data.get("results", [])
            content_type = "tv"
        else:
            content_type = "movie"
            
        if not results:
            return {
                "status": "not_found",
                "message": f"Could not find '{query}'"
            }
            
        content_id = results[0]["id"]
        
        if content_type == "movie":
            details = make_tmdb_request(
                f"movie/{content_id}", 
                {"language": "en-US", "append_to_response": "credits,similar,videos,release_dates"}
            )
            
            rating = "Not rated"
            if "release_dates" in details:
                for country in details["release_dates"]["results"]:
                    if country["iso_3166_1"] == "US":
                        for release in country["release_dates"]:
                            if release.get("certification"):
                                rating = release["certification"]
                                break
            
            director = next((c["name"] for c in details.get("credits", {}).get("crew", []) 
                          if c.get("job") == "Director"), "Unknown")
            
            writers = [c["name"] for c in details.get("credits", {}).get("crew", [])
                    if c.get("job") in ["Writer", "Screenplay"]]
            
            trailer = ""
            for video in details.get("videos", {}).get("results", []):
                if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                    trailer = f"https://www.youtube.com/watch?v={video['key']}"
                    break
                    
            return {
                "status": "success",
                "content_type": "movie",
                "id": details.get("id"),
                "title": details.get("title"),
                "original_title": details.get("original_title"),
                "tagline": details.get("tagline"),
                "overview": details.get("overview"),
                "release_date": details.get("release_date"),
                "runtime": details.get("runtime"),
                "rating": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
                "popularity": details.get("popularity"),
                "genres": [g["name"] for g in details.get("genres", [])],
                "content_rating": rating,
                "director": director,
                "writers": writers[:3],
                "cast": [{"name": c["name"], "character": c["character"]} 
                       for c in details.get("credits", {}).get("cast", [])[:10]],
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "poster_path": details.get("poster_path"),
                "trailer": trailer,
                "similar": [{"id": m["id"], "title": m["title"]} 
                          for m in details.get("similar", {}).get("results", [])[:5]]
            }
        else:
            details = make_tmdb_request(
                f"tv/{content_id}", 
                {"language": "en-US", "append_to_response": "credits,similar,videos,content_ratings"}
            )
            
            rating = "Not rated"
            if "content_ratings" in details:
                for country in details["content_ratings"]["results"]:
                    if country["iso_3166_1"] == "US":
                        rating = country["rating"]
                        break
            
            creators = [p["name"] for p in details.get("created_by", [])]
            
            trailer = ""
            for video in details.get("videos", {}).get("results", []):
                if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                    trailer = f"https://www.youtube.com/watch?v={video['key']}"
                    break
                    
            return {
                "status": "success",
                "content_type": "tv",
                "id": details.get("id"),
                "title": details.get("name"),
                "original_title": details.get("original_name"),
                "tagline": details.get("tagline"),
                "overview": details.get("overview"),
                "first_air_date": details.get("first_air_date"),
                "last_air_date": details.get("last_air_date"),
                "rating": details.get("vote_average"),
                "vote_count": details.get("vote_count"),
                "popularity": details.get("popularity"),
                "status": details.get("status"),
                "genres": [g["name"] for g in details.get("genres", [])],
                "content_rating": rating,
                "creators": creators,
                "seasons": len(details.get("seasons", [])),
                "episodes": sum(s.get("episode_count", 0) for s in details.get("seasons", [])),
                "cast": [{"name": c["name"], "character": c["character"]} 
                       for c in details.get("credits", {}).get("cast", [])[:10]],
                "poster_path": details.get("poster_path"),
                "trailer": trailer,
                "networks": [n["name"] for n in details.get("networks", [])],
                "similar": [{"id": s["id"], "title": s["name"]} 
                          for s in details.get("similar", {}).get("results", [])[:5]]
            }

class FetchMovieReviewsTool(BaseTool):
    name: str = "fetch_movie_reviews"
    description: str = "Fetches reviews and critiques for a specific movie or TV show"
    args_schema: Type[BaseModel] = FetchMovieReviewsSchema
    
    def _run(self, movie_id: int):
        reviews = make_tmdb_request(f"movie/{movie_id}/reviews")
        content_type = "movie"
        
        if "status_code" in reviews and reviews["status_code"] == 34:
            reviews = make_tmdb_request(f"tv/{movie_id}/reviews")
            content_type = "tv"
            
        formatted_reviews = []
        for review in reviews.get("results", [])[:5]:
            formatted_reviews.append({
                "author": review.get("author", "Anonymous"),
                "rating": review.get("author_details", {}).get("rating"),
                "content": review.get("content", "")[:300] + "..." if len(review.get("content", "")) > 300 else review.get("content", ""),
                "url": review.get("url"),
                "created_at": review.get("created_at")
            })
            
        return {
            "content_type": content_type,
            "movie_id": movie_id,
            "total_reviews": reviews.get("total_results", 0),
            "reviews": formatted_reviews
        }

class SearchSimilarMoviesSchema(BaseModel):
    query: str = Field(description="Movie name to find similar titles for")
    max_results: int = Field(default=5, description="Maximum number of results")

class RecommendByGenreSchema(BaseModel):
    genre: str = Field(description="Genre to search for recommendations")
    min_rating: float = Field(default=7.0, description="Minimum rating (0-10)")
    max_results: int = Field(default=8, description="Maximum number of results")
    year_from: Optional[int] = Field(default=None, description="Starting year (optional)")
    year_to: Optional[int] = Field(default=None, description="Ending year (optional)")

class SearchSimilarMoviesTool(BaseTool):
    name: str = "search_similar_movies"
    description: str = "Finds movies or TV shows similar to a specific title"
    args_schema: Type[BaseModel] = SearchSimilarMoviesSchema
    
    def _run(self, query: str, max_results: int = 5):
        search_results = make_tmdb_request("search/movie", {"query": query, "language": "en-US"})
        movie_results = search_results.get("results", [])
        
        if not movie_results:
            search_results = make_tmdb_request("search/tv", {"query": query, "language": "en-US"})
            tv_results = search_results.get("results", [])
            
            if not tv_results:
                return {
                    "status": "not_found",
                    "message": f"Could not find '{query}'"
                }
                
            content_id = tv_results[0]["id"]
            content_type = "tv"
            original_title = tv_results[0]["name"]
        else:
            content_id = movie_results[0]["id"]
            content_type = "movie"
            original_title = movie_results[0]["title"]
            
        similar_data = make_tmdb_request(f"{content_type}/{content_id}/similar")
        similar_results = similar_data.get("results", [])[:max_results]
        
        formatted_results = []
        for item in similar_results:
            if content_type == "movie":
                formatted_results.append({
                    "id": item["id"],
                    "title": item["title"],
                    "overview": item["overview"],
                    "year": item["release_date"].split("-")[0] if item.get("release_date") else "N/A",
                    "rating": item["vote_average"],
                    "popularity": item["popularity"]
                })
            else:
                formatted_results.append({
                    "id": item["id"],
                    "title": item["name"],
                    "overview": item["overview"],
                    "year": item["first_air_date"].split("-")[0] if item.get("first_air_date") else "N/A",
                    "rating": item["vote_average"],
                    "popularity": item["popularity"]
                })
                
        return {
            "status": "success",
            "query": query,
            "reference": {
                "id": content_id,
                "title": original_title,
                "type": content_type
            },
            "results": formatted_results
        }

class RecommendByGenreTool(BaseTool):
    name: str = "recommend_by_genre"
    description: str = "Recommends movies or TV shows by genre with additional filters"
    args_schema: Type[BaseModel] = RecommendByGenreSchema
    
    def _run(self, genre: str, min_rating: float = 7.0, max_results: int = 8, 
             year_from: Optional[int] = None, year_to: Optional[int] = None):
        genre_id = get_genre_id(genre)
        if not genre_id:
            return {
                "status": "error",
                "message": f"Genre '{genre}' not recognized"
            }
            
        params = {
            "with_genres": genre_id,
            "vote_average.gte": min_rating,
            "sort_by": "vote_average.desc",
            "language": "en-US"
        }
        
        if year_from:
            if year_from > 1900:
                params["primary_release_date.gte"] = f"{year_from}-01-01"
                
        if year_to:
            if year_to > 1900:
                params["primary_release_date.lte"] = f"{year_to}-12-31"
        
        discover_data = make_tmdb_request("discover/movie", params)
        results = discover_data.get("results", [])[:max_results]
        
        formatted_results = []
        for item in results:
            formatted_results.append({
                "id": item["id"],
                "title": item["title"],
                "overview": item["overview"],
                "year": item["release_date"].split("-")[0] if item.get("release_date") else "N/A",
                "rating": item["vote_average"],
                "popularity": item["popularity"]
            })
            
        return {
            "status": "success",
            "genre": genre,
            "min_rating": min_rating,
            "year_range": f"{year_from or 'any'} to {year_to or 'present'}",
            "results": formatted_results
        }

class FetchTrendingMoviesSchema(BaseModel):
    media_type: str = Field(default="all", description="Media type: all, movie, tv")
    time_window: str = Field(default="week", description="Time window: day, week")
    max_results: int = Field(default=10, description="Maximum number of results")

class FetchTrendingMoviesTool(BaseTool):
    name: str = "fetch_trending_movies"
    description: str = "Fetches trending movies and TV shows"
    args_schema: Type[BaseModel] = FetchTrendingMoviesSchema
    
    def _run(self, media_type: str = "all", time_window: str = "week", max_results: int = 10):
        valid_media_types = ["all", "movie", "tv"]
        if media_type not in valid_media_types:
            media_type = "all"
            
        valid_time_windows = ["day", "week"]
        if time_window not in valid_time_windows:
            time_window = "week"
            
        trending_data = make_tmdb_request(f"trending/{media_type}/{time_window}")
        results = trending_data.get("results", [])[:max_results]
        
        formatted_results = []
        for item in results:
            content_type = item.get("media_type")
            if content_type == "movie":
                formatted_results.append({
                    "id": item["id"],
                    "title": item["title"],
                    "overview": item["overview"],
                    "year": item["release_date"].split("-")[0] if item.get("release_date") else "N/A",
                    "rating": item["vote_average"],
                    "popularity": item["popularity"],
                    "media_type": "movie"
                })
            elif content_type == "tv":
                formatted_results.append({
                    "id": item["id"],
                    "title": item["name"],
                    "overview": item["overview"],
                    "year": item["first_air_date"].split("-")[0] if item.get("first_air_date") else "N/A",
                    "rating": item["vote_average"],
                    "popularity": item["popularity"],
                    "media_type": "tv"
                })
                
        return {
            "status": "success",
            "media_type": media_type,
            "time_window": time_window,
            "results": formatted_results
        }

# For convenient importing
all_tools = {
    'intent_classifier': IntentClassifierTool(),
    'fetch_movie_info': FetchMovieInfoTool(),
    'fetch_movie_reviews': FetchMovieReviewsTool(),
    'search_similar_movies': SearchSimilarMoviesTool(),
    'recommend_by_genre': RecommendByGenreTool(),
    'fetch_trending_movies': FetchTrendingMoviesTool()
}