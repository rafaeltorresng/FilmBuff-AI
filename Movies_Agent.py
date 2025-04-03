import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool, BaseTool
from typing import Optional, Dict, List, Union
import requests
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# ========== API HANDLER ==========

def make_tmdb_request(endpoint, params=None):
    """Make a request to TMDB API with error handling and rate limiting"""
    if params is None:
        params = {}
    
    # Ensure API key is included
    params["api_key"] = TMDB_API_KEY
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        response_status = getattr(e.response, 'status_code', None)
        # Handle rate limiting
        if response_status == 429:
            retry_after = int(e.response.headers.get("Retry-After", 1))
            time.sleep(retry_after)
            return make_tmdb_request(endpoint, params)
        
        print(f"API request error: {str(e)}")
        return {"error": str(e), "results": []}

# ========== TOOLS ==========

@tool
def search_movies(
    query: str,
    max_results: int = 5,
    year: Optional[int] = None,
    genre: Optional[str] = None
) -> List[Dict]:
    """
    Search for movies based on keywords, year, or genre.
    Example: "thriller movies with plot twist"
    """
    params = {
        "query": query,
        "language": "en-US"
    }
    if year:
        params["year"] = year
    if genre:
        genre_id = get_genre_id(genre)
        if genre_id:
            params["with_genres"] = genre_id

    data = make_tmdb_request("search/movie", params)
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item["title"],
        "overview": item["overview"],
        "year": item["release_date"].split("-")[0] if item.get("release_date") else "N/A",
        "rating": item["vote_average"],
        "popularity": item["popularity"],
        "url": f"https://www.themoviedb.org/movie/{item['id']}"
    } for item in results]

@tool
def search_tv_shows(
    query: str,
    max_results: int = 5,
    year: Optional[int] = None,
    genre: Optional[str] = None
) -> List[Dict]:
    """
    Search for TV shows based on keywords, year, or genre.
    Example: "sci-fi shows with time travel"
    """
    params = {
        "query": query,
        "language": "en-US"
    }
    if year:
        params["first_air_date_year"] = year
    if genre:
        genre_id = get_tv_genre_id(genre)
        if genre_id:
            params["with_genres"] = genre_id

    data = make_tmdb_request("search/tv", params)
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item["name"],
        "overview": item["overview"],
        "year": item["first_air_date"].split("-")[0] if item.get("first_air_date") else "N/A",
        "rating": item["vote_average"],
        "popularity": item["popularity"],
        "url": f"https://www.themoviedb.org/tv/{item['id']}"
    } for item in results]

@tool
def get_movie_details(movie_id: int) -> Dict:
    """
    Get detailed information about a specific movie by ID.
    Example: "Get details for movie 550" (Fight Club)
    """
    data = make_tmdb_request(
        f"movie/{movie_id}",
        {"language": "en-US", "append_to_response": "credits,similar,reviews,videos"}
    )
    
    if "error" in data:
        return {"error": data["error"]}
    
    # Extract trailer if available
    trailer = ""
    for video in data.get("videos", {}).get("results", []):
        if video.get("type") == "Trailer" and video.get("site") == "YouTube":
            trailer = f"https://www.youtube.com/watch?v={video['key']}"
            break
    
    # Extract key crew members
    director = next((c["name"] for c in data.get("credits", {}).get("crew", []) 
                   if c.get("job") == "Director"), "Unknown")
    
    writers = [c["name"] for c in data.get("credits", {}).get("crew", [])
              if c.get("job") in ["Writer", "Screenplay"]]
    
    # Extract reviews
    reviews = []
    for review in data.get("reviews", {}).get("results", [])[:2]:
        reviews.append({
            "author": review.get("author", "Anonymous"),
            "excerpt": review.get("content", "")[:200] + "..." if review.get("content") else ""
        })
    
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "tagline": data.get("tagline"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "genres": [g["name"] for g in data.get("genres", [])],
        "director": director,
        "writers": writers[:2],
        "cast": [{"name": c["name"], "character": c["character"]} for c in data.get("credits", {}).get("cast", [])[:5]],
        "rating": data.get("vote_average"),
        "budget": f"${data.get('budget'):,}" if data.get('budget') else "Unknown",
        "revenue": f"${data.get('revenue'):,}" if data.get('revenue') else "Unknown",
        "trailer": trailer,
        "reviews": reviews,
        "similar": [{"id": m["id"], "title": m["title"]} for m in data.get("similar", {}).get("results", [])[:3]],
        "url": f"https://www.themoviedb.org/movie/{data.get('id')}"
    }

@tool
def get_tv_show_details(show_id: int) -> Dict:
    """
    Get detailed information about a specific TV show by ID.
    Example: "Get details for show 1399" (Game of Thrones)
    """
    data = make_tmdb_request(
        f"tv/{show_id}", 
        {"language": "en-US", "append_to_response": "credits,similar,reviews,videos"}
    )
    
    if "error" in data:
        return {"error": data["error"]}
    
    # Extract trailer if available
    trailer = ""
    for video in data.get("videos", {}).get("results", []):
        if video.get("type") == "Trailer" and video.get("site") == "YouTube":
            trailer = f"https://www.youtube.com/watch?v={video['key']}"
            break
    
    # Extract key crew members
    creators = [p["name"] for p in data.get("created_by", [])]
    
    # Extract seasons info
    seasons = []
    for season in data.get("seasons", []):
        if season.get("season_number") > 0:  # Skip specials
            seasons.append({
                "season_number": season.get("season_number"),
                "episode_count": season.get("episode_count"),
                "air_date": season.get("air_date")
            })
    
    # Extract reviews
    reviews = []
    for review in data.get("reviews", {}).get("results", [])[:2]:
        reviews.append({
            "author": review.get("author", "Anonymous"),
            "excerpt": review.get("content", "")[:200] + "..." if review.get("content") else ""
        })
    
    return {
        "id": data.get("id"),
        "title": data.get("name"),
        "tagline": data.get("tagline"),
        "overview": data.get("overview"),
        "first_air_date": data.get("first_air_date"),
        "last_air_date": data.get("last_air_date"),
        "status": data.get("status"),
        "seasons": len(seasons),
        "episodes": sum(s.get("episode_count", 0) for s in seasons),
        "genres": [g["name"] for g in data.get("genres", [])],
        "creators": creators,
        "cast": [{"name": c["name"], "character": c["character"]} for c in data.get("credits", {}).get("cast", [])[:5]],
        "rating": data.get("vote_average"),
        "trailer": trailer,
        "reviews": reviews,
        "similar": [{"id": s["id"], "title": s["name"]} for s in data.get("similar", {}).get("results", [])[:3]],
        "url": f"https://www.themoviedb.org/tv/{data.get('id')}"
    }

@tool
def discover_movies(
    genre: Optional[str] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
    min_rating: float = 0.0,
    max_results: int = 5
) -> List[Dict]:
    """
    Discover movies based on specific criteria without a keyword search.
    Example: "Find popular action movies from 2022"
    """
    params = {
        "language": "en-US",
        "sort_by": sort_by,
        "vote_average.gte": min_rating,
        "include_adult": False
    }
    
    if genre:
        genre_id = get_genre_id(genre)
        if genre_id:
            params["with_genres"] = genre_id
    
    if year:
        params["primary_release_year"] = year
    
    data = make_tmdb_request("discover/movie", params)
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item["title"],
        "overview": item["overview"],
        "year": item["release_date"].split("-")[0] if item.get("release_date") else "N/A",
        "rating": item["vote_average"],
        "popularity": item["popularity"],
        "url": f"https://www.themoviedb.org/movie/{item['id']}"
    } for item in results]

@tool
def discover_tv_shows(
    genre: Optional[str] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
    min_rating: float = 0.0,
    max_results: int = 5
) -> List[Dict]:
    """
    Discover TV shows based on specific criteria without a keyword search.
    Example: "Find popular drama series from 2021"
    """
    params = {
        "language": "en-US",
        "sort_by": sort_by,
        "vote_average.gte": min_rating,
        "include_adult": False
    }
    
    if genre:
        genre_id = get_tv_genre_id(genre)
        if genre_id:
            params["with_genres"] = genre_id
    
    if year:
        params["first_air_date_year"] = year
    
    data = make_tmdb_request("discover/tv", params)
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item["name"],
        "overview": item["overview"],
        "year": item["first_air_date"].split("-")[0] if item.get("first_air_date") else "N/A",
        "rating": item["vote_average"],
        "popularity": item["popularity"],
        "url": f"https://www.themoviedb.org/tv/{item['id']}"
    } for item in results]

@tool
def get_trending_content(
    media_type: str = "all",  # Options: "all", "movie", "tv", "person"
    time_window: str = "week",  # Options: "day", "week"
    max_results: int = 5
) -> List[Dict]:
    """
    Get trending movies, TV shows, or people.
    Example: "What movies are trending this week?"
    """
    data = make_tmdb_request(f"trending/{media_type}/{time_window}")
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item.get("title", item.get("name")),
        "media_type": item.get("media_type"),
        "overview": item.get("overview", ""),
        "rating": item.get("vote_average"),
        "popularity": item["popularity"],
        "url": f"https://www.themoviedb.org/{item.get('media_type', 'movie')}/{item['id']}"
    } for item in results]

@tool
def search_person(
    query: str,
    max_results: int = 5
) -> List[Dict]:
    """
    Search for actors, directors, or other film industry personalities.
    Example: "Find information about Christopher Nolan"
    """
    data = make_tmdb_request("search/person", {"query": query, "language": "en-US"})
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "name": item["name"],
        "popularity": item["popularity"],
        "known_for_department": item.get("known_for_department"),
        "known_for": [kf.get("title", kf.get("name", "Unknown")) for kf in item.get("known_for", [])],
        "url": f"https://www.themoviedb.org/person/{item['id']}"
    } for item in results]

@tool
def get_person_details(person_id: int) -> Dict:
    """
    Get detailed information about a specific person by ID.
    Example: "Get details for person 138" (Quentin Tarantino)
    """
    data = make_tmdb_request(
        f"person/{person_id}",
        {"language": "en-US", "append_to_response": "movie_credits,tv_credits"}
    )
    
    if "error" in data:
        return {"error": data["error"]}
    
    # Get notable movies (as director or actor)
    notable_movies = []
    if data.get("movie_credits", {}).get("cast"):
        for movie in sorted(data["movie_credits"]["cast"], key=lambda x: x.get("popularity", 0), reverse=True)[:5]:
            notable_movies.append({
                "id": movie["id"],
                "title": movie["title"],
                "character": movie.get("character", ""),
                "year": movie.get("release_date", "").split("-")[0] if movie.get("release_date") else ""
            })
    
    # Get notable movies as crew (director, writer, etc.)
    if len(notable_movies) < 5 and data.get("movie_credits", {}).get("crew"):
        for movie in sorted(data["movie_credits"]["crew"], key=lambda x: x.get("popularity", 0), reverse=True):
            if len(notable_movies) >= 5:
                break
            notable_movies.append({
                "id": movie["id"],
                "title": movie["title"],
                "job": movie.get("job", ""),
                "year": movie.get("release_date", "").split("-")[0] if movie.get("release_date") else ""
            })
    
    # Get notable TV shows
    notable_tv = []
    if data.get("tv_credits", {}).get("cast"):
        for show in sorted(data["tv_credits"]["cast"], key=lambda x: x.get("popularity", 0), reverse=True)[:3]:
            notable_tv.append({
                "id": show["id"],
                "title": show["name"],
                "character": show.get("character", ""),
                "year": show.get("first_air_date", "").split("-")[0] if show.get("first_air_date") else ""
            })
    
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "birthday": data.get("birthday"),
        "place_of_birth": data.get("place_of_birth"),
        "biography": data.get("biography"),
        "known_for_department": data.get("known_for_department"),
        "notable_movies": notable_movies,
        "notable_tv_shows": notable_tv,
        "url": f"https://www.themoviedb.org/person/{data.get('id')}"
    }

@tool
def find_similar_content(
    content_id: int,
    content_type: str,  # "movie" or "tv"
    max_results: int = 5
) -> List[Dict]:
    """
    Find content similar to a specific movie or TV show.
    Example: "Find shows similar to Breaking Bad"
    """
    data = make_tmdb_request(f"{content_type}/{content_id}/similar")
    results = data.get("results", [])[:max_results]
    
    return [{
        "id": item["id"],
        "title": item.get("title", item.get("name")),
        "overview": item["overview"],
        "year": (item.get("release_date") or item.get("first_air_date", "")).split("-")[0] or "N/A",
        "rating": item["vote_average"],
        "url": f"https://www.themoviedb.org/{content_type}/{item['id']}"
    } for item in results]

# ========== HELPER FUNCTIONS ==========

# Cache genre lookups to avoid repetitive string matching
@lru_cache(maxsize=128)
def get_genre_id(genre_name: str) -> Optional[int]:
    """Get movie genre ID from name with flexible matching"""
    # English genre names with exact IDs
    genres = {
        "Action": 28, "Adventure": 12, "Animation": 16,
        "Comedy": 35, "Crime": 80, "Documentary": 99,
        "Drama": 18, "Family": 10751, "Fantasy": 14,
        "History": 36, "Horror": 27, "Mystery": 9648,
        "Romance": 10749, "Science Fiction": 878, "Sci-Fi": 878,
        "Thriller": 53, "War": 10752, "Western": 37
    }
    
    # Try exact match
    normalized_genre = genre_name.title()
    if normalized_genre in genres:
        return genres[normalized_genre]
    
    # Try partial match
    for key, value in genres.items():
        if key.lower() in genre_name.lower():
            return value
            
    return None

@lru_cache(maxsize=128)
def get_tv_genre_id(genre_name: str) -> Optional[int]:
    """Get TV genre ID from name with flexible matching"""
    genres = {
        "Action & Adventure": 10759, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18,
        "Family": 10751, "Kids": 10762, "Mystery": 9648,
        "News": 10763, "Reality": 10764, "Sci-Fi & Fantasy": 10765,
        "Soap": 10766, "Talk": 10767, "War & Politics": 10768
    }
    
    # Try exact match
    normalized_genre = genre_name.title()
    if normalized_genre in genres:
        return genres[normalized_genre]
    
    # Try partial match
    for key, value in genres.items():
        if key.lower() in genre_name.lower():
            return value
            
    return None

# ========== AGENTS ==========
# Create explicit BaseTool objects
def create_tool(func):
    """Convert a function to a BaseTool compatible with CrewAI"""
    return {
        "name": func.__name__,
        "description": func.__doc__,
        "func": func
    }

# Create tools map for agent setup
tools_dict = {
    "search_movies": search_movies,
    "search_tv_shows": search_tv_shows,
    "discover_movies": discover_movies,
    "discover_tv_shows": discover_tv_shows,
    "get_movie_details": get_movie_details,
    "get_tv_show_details": get_tv_show_details,
    "get_trending_content": get_trending_content,
    "search_person": search_person,
    "get_person_details": get_person_details,
    "find_similar_content": find_similar_content,
}

# Research Agent - Finds content based on criteria
research_agent = Agent(
    role="Content Researcher",
    goal="Find movies and TV shows that match the user's criteria",
    tools=[
        {
            "name": "search_movies",
            "description": "Search for movies based on keywords, year, or genre.",
            "func": search_movies
        },
        {
            "name": "search_tv_shows",
            "description": "Search for TV shows based on keywords, year, or genre.",
            "func": search_tv_shows
        },
        {
            "name": "discover_movies",
            "description": "Discover movies based on specific criteria without a keyword search.",
            "func": discover_movies
        },
        {
            "name": "discover_tv_shows",
            "description": "Discover TV shows based on specific criteria without a keyword search.",
            "func": discover_tv_shows
        },
        {
            "name": "get_trending_content",
            "description": "Get trending movies, TV shows, or people.",
            "func": get_trending_content
        }
    ],
    backstory="You are a meticulous researcher specialized in finding audiovisual content based on specific criteria.",
    verbose=True
)

# Details Agent - Provides in-depth information
details_agent = Agent(
    role="Details Specialist",
    goal="Provide detailed information about movies and TV shows",
    tools=[
        {
            "name": "get_movie_details",
            "description": "Get detailed information about a specific movie by ID.",
            "func": get_movie_details
        },
        {
            "name": "get_tv_show_details",
            "description": "Get detailed information about a specific TV show by ID.",
            "func": get_tv_show_details
        },
        {
            "name": "get_person_details",
            "description": "Get detailed information about a specific person by ID.",
            "func": get_person_details
        },
        {
            "name": "find_similar_content",
            "description": "Find content similar to a specific movie or TV show.",
            "func": find_similar_content
        }
    ],
    backstory="You are a film and television expert with encyclopedic knowledge about audiovisual productions.",
    verbose=True
)

# Recommendation Agent - Creates personalized recommendations
recommendation_agent = Agent(
    role="Recommendation Consultant",
    goal="Create personalized recommendations based on user preferences",
    tools=[
        {
            "name": "search_movies",
            "description": "Search for movies based on keywords, year, or genre.",
            "func": search_movies
        },
        {
            "name": "search_tv_shows",
            "description": "Search for TV shows based on keywords, year, or genre.",
            "func": search_tv_shows
        },
        {
            "name": "discover_movies",
            "description": "Discover movies based on specific criteria without a keyword search.",
            "func": discover_movies
        },
        {
            "name": "discover_tv_shows",
            "description": "Discover TV shows based on specific criteria without a keyword search.",
            "func": discover_tv_shows
        },
        {
            "name": "find_similar_content",
            "description": "Find content similar to a specific movie or TV show.",
            "func": find_similar_content
        }
    ],
    backstory="You are a renowned film critic, known for recommending movies and shows that perfectly match audience tastes.",
    verbose=True
)

# People Agent - Specialized in actors, directors and crew
people_agent = Agent(
    role="People Specialist",
    goal="Find information about actors, directors, and other film industry personalities",
    tools=[
        {
            "name": "search_person",
            "description": "Search for actors, directors, or other film industry personalities.",
            "func": search_person
        },
        {
            "name": "get_person_details",
            "description": "Get detailed information about a specific person by ID.",
            "func": get_person_details
        }
    ],
    backstory="You are a celebrity expert with deep knowledge about film industry professionals and their careers.",
    verbose=True
)

# ========== CREW ==========

entertainment_crew = Crew(
    agents=[research_agent, details_agent, recommendation_agent, people_agent],
    tasks=[
        Task(
            description="Find 3 thriller movies with plot twists released after 2010",
            expected_output="A list of movies with title, year, synopsis, and rating",
            agent=research_agent
        ),
        Task(
            description="Analyze the found movies and provide details about cast and direction",
            expected_output="Detailed information about each movie, including cast and direction",
            agent=details_agent
        ),
        Task(
            description="Based on previous choices, recommend 2 additional movies the user might like",
            expected_output="Personalized recommendations with justification",
            agent=recommendation_agent
        )
    ],
    process=Process.sequential,
    verbose=2
)

# ========== EXECUTION ==========

def get_recommendations(user_query: str, include_people: bool = False):
    """Function to handle user queries and return recommendations"""
    # Create dynamic tasks based on user query
    tasks = [
        Task(
            description=f"Based on the user request: '{user_query}', find relevant content",
            expected_output="List of movies or TV shows that match the request",
            agent=research_agent
        ),
        Task(
            description="Analyze the found content and provide relevant details",
            expected_output="Detailed information about each item",
            agent=details_agent
        ),
        Task(
            description="Create personalized recommendations based on the results",
            expected_output="Final list with justified recommendations",
            agent=recommendation_agent
        )
    ]
    
    # Add people search if needed
    if include_people or any(keyword in user_query.lower() for keyword in ["actor", "director", "actress", "star", "cast"]):
        tasks.insert(1, Task(
            description=f"Find information about relevant people mentioned in '{user_query}'",
            expected_output="Information about actors, directors or other film professionals",
            agent=people_agent
        ))
    
    # Update crew with new tasks
    entertainment_crew.tasks = tasks
    
    # Execute the crew tasks
    result = entertainment_crew.kickoff()
    return result

# Example query - in a real application, this would come from user input
if __name__ == "__main__":
    user_query = "Recommend science fiction movies about time travel"
    result = get_recommendations(user_query)
    print(result)