import os
import re
import json
import time
import requests
from typing import Dict, List, Any, Optional, Type, Union
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

class QueryCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def save_cache(self):
        try:
            import pickle
            with open("query_cache.pkl", "wb") as f:
                pickle.dump(self.cache, f)
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

query_cache = QueryCache()

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
        
        if any(pattern in query for pattern in ["tell me about", "information about", "details of", 
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

manager_agent = Agent(
    role="Manager Agent",
    goal="Analyze user query and delegate to the appropriate specialist agent",
    backstory="""You are the manager of a team of film and TV show specialists. 
                Your role is to understand what the user is asking for and direct 
                the query to the most appropriate specialist.""",
    tools=[IntentClassifierTool()],
    verbose=True,
    llm=llm
)

information_agent = Agent(
    role="Information Agent",
    goal="Provide detailed and accurate information about movies and TV shows",
    backstory="""You are an encyclopedia of knowledge about cinema and television.
                Your role is to provide detailed, contextualized, and well-formatted 
                information about movies, TV shows, cast, and production teams.""",
    tools=[FetchMovieInfoTool(), FetchMovieReviewsTool()],
    verbose=True,
    llm=llm
)

recommendation_agent = Agent(
    role="Recommendation Agent",
    goal="Recommend personalized movies and TV shows based on user preferences",
    backstory="""You are a film curator specialized in making personalized recommendations.
                Your role is to understand user preferences and suggest relevant 
                and high-quality content.""",
    tools=[SearchSimilarMoviesTool(), RecommendByGenreTool()],
    verbose=True,
    llm=llm
)

trends_agent = Agent(
    role="Trends Agent",
    goal="Inform about currently trending movies and TV shows",
    backstory="""You are a specialist in entertainment world trends.
                Your role is to keep users updated on the most popular
                and trending content at the moment.""",
    tools=[FetchTrendingMoviesTool()],
    verbose=True,
    llm=llm
)

def create_manager_task(query: str) -> Task:
    return Task(
        description=f"""
        # User Query Analysis
        
        Analyze the following user query: "{query}"
        
        Your role is to:
        1. Use the intent_classifier_tool to determine the query type
        2. Decide which specialist agent should process this query
        
        Please return:
        - The identified query type
        - The agent that should process this query
        - A brief explanation of why this classification
        """,
        expected_output="Query analysis with type and responsible agent",
        agent=manager_agent
    )

def create_information_task(query: str) -> Task:
    return Task(
        description=f"""
        # Content Information Search
        
        The user is looking for information about: "{query}"
        
        ## QUERY ANALYSIS:
        1. First, determine if this is a general or specific query:
        *For queries like "best", "highest rated", or "top" movies/TV shows:
           - Limit API calls to a maximum of 5 attempts
           - If you can't find specific information, provide a general response
           - Consider using your knowledge of widely acclaimed films
        
        2. For specific queries, identify the exact type of information requested:
           - Direction/Director
           - Cast/Specific actors
           - Rating/Score
           - Content rating
           - Duration/Seasons
           - Release year
           - Genre
           - Nominations/Awards
        
        ## HOW TO ACT:
        1. Use the fetch_movie_info tool to find the content
        2. For specific queries:
           - Answer ONLY the requested information with minimal context
           - Keep the response short and direct
           - Example: "The director of Interstellar is Christopher Nolan."
        
        3. For general queries:
           - Present complete and well-organized information
           - Use fetch_movie_reviews if needed to complement
           - Include title, year, synopsis, genres, cast, direction, etc.
        
        Format your response using markdown to improve readability.
        """,
        expected_output="Information about the content - specific or general as requested",
        agent=information_agent
    )

def create_recommendation_task(query: str) -> Task:
    return Task(
        description=f"""
        # Personalized Recommendations
        
        The user is looking for recommendations: "{query}"
        
        Your role is to:
        1. Analyze the query to identify preferences (genres, reference movies)
        2. Use search_similar_movies if the user mentioned a specific title
        3. Use recommend_by_genre if the user mentioned a specific genre
        4. If the query is generic, identify an implicit genre and use recommend_by_genre
        
        Include in your response:
        - A list of 5-8 high-quality recommendations
        - Brief explanation of each recommendation
        - Rating and year of each recommendation
        - Why you believe these choices align with the query
        
        Format your response using markdown to improve readability.
        """,
        expected_output="Personalized list of recommendations with explanations",
        agent=recommendation_agent
    )

def create_trends_task(query: str) -> Task:
    return Task(
        description=f"""
        # Current Trends in Movies and TV Shows
        
        The user wants to know about trends: "{query}"
        
        Your role is to:
        1. Determine if the user is interested in movies, TV shows, or both
        2. Decide if the focus should be daily or weekly trends
        3. Use fetch_trending_movies with the appropriate parameters
        4. Present the results in an organized and informative way
        
        ## CRITICAL FORMATTING INSTRUCTIONS:
        After receiving the tool results, you MUST:
        - Format your response as an organized list with markdown
        - Include a title indicating that these are trends for the week/day
        - For EACH item in the results list, include title, type (movie/TV show), rating, year, and a brief description
        - Highlight the 3-5 most popular or highest-rated items
        
        ## EXPECTED FORMAT EXAMPLE:
        ```
        # Trending Movies and TV Shows This Week
        
        Here are the most popular content currently:
        
        ## Highlights:
        
        1. **[Movie/TV Show Title]** (Year) - ⭐ 8.5/10
           Type: Movie/TV Show
           Brief description of the content...
           
        2. **[Other Movie/TV Show Title]** (Year) - ⭐ 9.0/10
           Type: Movie/TV Show
           Brief description of the content...
           
        ## Other Trending Content:
        
        3. **[Title]** (Year) - ⭐ 7.8/10
           Type: Movie/TV Show
           Description...
        ```
        
        IMPORTANT: You MUST format a complete response with all items from the tool results.
        """,
        expected_output="Detailed and formatted list of trending content",
        agent=trends_agent
    )

def process_film_buff_query(query: str) -> str:
    cached_result = query_cache.get(query)
    if cached_result:
        print("Using cached result")
        return cached_result
    
    try:
        print(f"Processing query: '{query}'")
        
        manager_crew = Crew(
            agents=[manager_agent],
            tasks=[create_manager_task(query)],
            process=Process.sequential,
            verbose=True
        )
        
        manager_analysis = str(manager_crew.kickoff())
        print(f"Manager analysis: {manager_analysis}")
        
        target_agent = None
        
        if any(term in manager_analysis.lower() for term in ["information_agent", "information agent", "movie info", "film information"]):
            target_agent = "information_agent"
            print("Delegating to Information Agent")
            specialist_crew = Crew(
                agents=[information_agent],
                tasks=[create_information_task(query)],
                process=Process.sequential,
                verbose=True
            )
        elif any(term in manager_analysis.lower() for term in ["trends_agent", "trends agent", "trending"]):
            target_agent = "trends_agent"
            print("Delegating to Trends Agent")
            specialist_crew = Crew(
                agents=[trends_agent],
                tasks=[create_trends_task(query)],
                process=Process.sequential,
                verbose=True
            )
        elif any(term in manager_analysis.lower() for term in ["recommendation_agent", "recommendation agent", "recommendations"]):
            target_agent = "recommendation_agent"
            print("Delegating to Recommendation Agent")
            specialist_crew = Crew(
                agents=[recommendation_agent],
                tasks=[create_recommendation_task(query)],
                process=Process.sequential,
                verbose=True
            )
        else:
            print("Target agent not clearly identified, determining from context")
            
            if "information" in query.lower() or "details" in query.lower() or "about" in query.lower():
                target_agent = "information_agent"
                print("Context suggests Information Agent")
                specialist_crew = Crew(
                    agents=[information_agent],
                    tasks=[create_information_task(query)],
                    process=Process.sequential,
                    verbose=True
                )
            else:
                target_agent = "recommendation_agent"
                print("Using Recommendation Agent as default")
                specialist_crew = Crew(
                    agents=[recommendation_agent],
                    tasks=[create_recommendation_task(query)],
                    process=Process.sequential,
                    verbose=True
                )
        
        specialist_result = str(specialist_crew.kickoff())
        
        if len(specialist_result.strip()) < 50:
            print(f"Result from {target_agent} too short, trying to improve")
            
            if target_agent == "trends_agent":
                print("Additional attempt with Trends Agent")
                retry_task = Task(
                    description=f"""
                    # CRITICAL TASK: Format Movie/TV Show Trends
                    
                    The user asked: "{query}"
                    
                    ## EXPLICIT AND MANDATORY INSTRUCTIONS:
                    
                    1. Use the fetch_trending_movies tool to get current data
                    2. With the results, you MUST create a complete and detailed response
                    3. Your response MUST follow exactly this format:
                    
                    # Trending Movies and TV Shows This Week
                    
                    Here are the most popular content currently:
                    
                    ## Highlights:
                    
                    1. **[MOVIE/TV SHOW TITLE]** (YEAR) - ⭐ [RATING]/10
                       Type: [Movie/TV Show]
                       [BRIEF DESCRIPTION]
                       
                    2. **[MOVIE/TV SHOW TITLE]** (YEAR) - ⭐ [RATING]/10
                       Type: [Movie/TV Show]
                       [BRIEF DESCRIPTION]
                       
                    ## Other Trending Content:
                    
                    [LIST ALL OTHER MOVIES/TV SHOWS IN THE SAME FORMAT]
                    """,
                    expected_output="Complete and formatted list of trending movies and TV shows",
                    agent=trends_agent
                )
                
                retry_crew = Crew(
                    agents=[trends_agent],
                    tasks=[retry_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                improved_result = str(retry_crew.kickoff())
                
                if len(improved_result.strip()) > 50:
                    specialist_result = improved_result
            elif target_agent == "information_agent":
                print("Additional attempt with Information Agent")
                retry_task = Task(
                    description=f"""
                    # CRITICAL TASK: Provide Detailed Information
                    
                    The user asked: "{query}"
                    
                    ## EXPLICIT INSTRUCTIONS:
                    1. Use the search_movie tool to find the specific movie
                    2. Use the get_movie_details tool to get detailed information
                    3. Format a complete response with ALL available details
                    
                    Include:
                    - Complete title and year
                    - Director and main cast
                    - Detailed synopsis
                    - Rating and popularity
                    - Genres and duration
                    - Important facts
                    """,
                    expected_output="Detailed information about the movie or TV show",
                    agent=information_agent
                )
                
                retry_crew = Crew(
                    agents=[information_agent],
                    tasks=[retry_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                improved_result = str(retry_crew.kickoff())
                
                if len(improved_result.strip()) > 50:
                    specialist_result = improved_result
        
        query_cache.set(query, specialist_result)
        return specialist_result
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"""# Sorry, an error occurred processing your query

Could not process: "{query}"

Error: {str(e)}

Please try:
1. Rephrasing your question
2. Being more specific about movie or TV show titles
3. Using simpler queries

Thank you for your understanding!
"""

if __name__ == "__main__":
    print("Testing Film Buff with CrewAI...")
    
    info_query = "Tell me about the movie Interstellar"
    print("\n\nTESTING INFORMATION QUERY:")
    result = process_film_buff_query(info_query)
    print(f"Result: {result[:200]}...")
    
    rec_query = "Recommend sci-fi movies similar to Blade Runner"
    print("\n\nTESTING RECOMMENDATION QUERY:")
    result = process_film_buff_query(rec_query)
    print(f"Result: {result[:200]}...")
    
    trend_query = "What movies are trending this week?"
    print("\n\nTESTING TRENDS QUERY:")
    result = process_film_buff_query(trend_query)
    print(f"Result: {result[:200]}...")