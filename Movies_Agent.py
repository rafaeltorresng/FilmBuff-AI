import os
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool, BaseTool
from typing import Optional, Dict, List, Union
import requests
from functools import lru_cache
import time
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

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
    verbose=True,
    llm=llm
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
    verbose=True,
    llm=llm
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
    verbose=True,
    llm=llm
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
    verbose=True,
    llm=llm
)

# ========== CREW ==========

entertainment_crew = Crew(
    agents=[research_agent, details_agent, recommendation_agent, people_agent],
    tasks=[],  # Tasks are added dynamically based on query type
    process=Process.sequential,
    verbose=2
)

# ========== QUERY INTENT CLASSIFICATION ==========

def classify_query_intent(query: str) -> str:
    """Classifies the intention of the user's query"""
    query = query.lower()
    
    # Information about a specific movie/show
    if any(pattern in query for pattern in ["tell me about", "information about", "details of", "details about", 
                                           "what is", "tell me more", "synopsis of", "plot of", "describe"]):
        # Check if it's about a person or movie
        if any(term in query for term in ["actor", "actress", "director", "who played", "who is", "who directed"]):
            return "person_info"
        return "movie_info"
    
    # Person-related queries
    if any(pattern in query for pattern in ["who is", "actor", "actress", "director", "cast of", "stars in", 
                                          "appeared in", "filmography"]):
        return "person_info"
    
    # Trending content queries
    if any(pattern in query for pattern in ["trending", "popular", "top rated", "best of", "this week", 
                                          "this month", "new releases", "what's hot", "what is popular"]):
        return "trending"
    
    # Genre-specific queries
    if any(f"best {genre}" in query for genre in ["action", "comedy", "drama", "horror", "sci-fi", "thriller", 
                                               "romance", "documentary", "animation"]):
        return "genre_specific"
    
    # Review or rating queries
    if any(pattern in query for pattern in ["review", "rating", "score", "how good is", "worth watching"]):
        return "reviews"
    
    # Recommendation queries (default)
    return "recommendations"

# ========== SPECIALIZED QUERY HANDLERS ==========

def get_movie_information(query: str) -> str:
    """Handles queries about specific movies or TV shows"""
    # Extract the movie/show title
    title = extract_title_from_query(query)
    
    tasks = [
        Task(
            description=f"Find movies or TV shows titled '{title}' or closest match",
            expected_output="Details about the movie/show including its ID",
            agent=research_agent
        ),
        Task(
            description=f"Get comprehensive details about '{title}' including plot, cast, and reviews",
            expected_output="Detailed information formatted for user presentation",
            agent=details_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_person_information(query: str) -> str:
    """Handles queries about specific people (actors, directors, etc.)"""
    # Extract the person's name
    name = extract_person_from_query(query)
    
    tasks = [
        Task(
            description=f"Find information about '{name}' in the film/TV industry",
            expected_output="Basic profile and career information",
            agent=people_agent
        ),
        Task(
            description=f"Compile detailed information about '{name}' including their notable work",
            expected_output="Comprehensive profile with filmography highlights",
            agent=details_agent
        ),
        Task(
            description=f"Suggest notable films/shows featuring '{name}' that the user might enjoy",
            expected_output="Curated recommendations of their best work",
            agent=recommendation_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_trending_information(query: str) -> str:
    """Handles queries about trending or popular content"""
    media_type = "movie"
    time_period = "week"
    
    # Try to detect if they want TV shows specifically
    if any(term in query.lower() for term in ["tv", "television", "series", "shows"]):
        media_type = "tv"
    
    # Try to detect time period
    if any(term in query.lower() for term in ["today", "daily", "right now"]):
        time_period = "day"
    
    tasks = [
        Task(
            description=f"Find trending {media_type}s for this {time_period}",
            expected_output="List of currently popular content with basic information",
            agent=research_agent
        ),
        Task(
            description="Provide more context about why these items are trending and what makes them notable",
            expected_output="Detailed analysis of each trending item",
            agent=details_agent
        ),
        Task(
            description="Based on these trends, identify emerging patterns or themes in popular content",
            expected_output="Insights about current audience preferences and entertainment trends",
            agent=recommendation_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_genre_recommendations(query: str) -> str:
    """Handles requests for recommendations in specific genres"""
    # Extract genre from query
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "thriller", 
              "romance", "documentary", "animation", "fantasy", "mystery"]
    
    detected_genre = next((genre for genre in genres if genre in query.lower()), "popular")
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    year = int(year_match.group(1)) if year_match else None
    
    tasks = [
        Task(
            description=f"Find top-rated {detected_genre} content" + (f" from {year}" if year else ""),
            expected_output="List of highly-regarded titles in this genre",
            agent=research_agent
        ),
        Task(
            description="Analyze these selections to identify what makes them standout examples of the genre",
            expected_output="Detailed information about plot, style, and reception",
            agent=details_agent
        ),
        Task(
            description="Create personalized recommendations based on the best examples of this genre",
            expected_output="Curated suggestions with justifications",
            agent=recommendation_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_review_information(query: str) -> str:
    """Handles requests for reviews or ratings"""
    # Extract the title
    title = extract_title_from_query(query)
    
    tasks = [
        Task(
            description=f"Find '{title}' and its critical reception",
            expected_output="Basic information including ratings and review sources",
            agent=research_agent
        ),
        Task(
            description=f"Compile critical consensus and audience response for '{title}'",
            expected_output="Summary of reviews, ratings, and audience reception",
            agent=details_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_recommendations(query: str, include_people: bool = False) -> str:
    """Handles standard recommendation requests"""
    tasks = [
        Task(
            description=f"Based on '{query}', find relevant content that matches these preferences",
            expected_output="List of movies or shows matching the criteria",
            agent=research_agent
        ),
        Task(
            description="Analyze these matches to identify common themes and elements",
            expected_output="Detailed information about the suggested content",
            agent=details_agent
        ),
        Task(
            description="Generate personalized recommendations based on these findings",
            expected_output="Final recommendations with explanations of why they were chosen",
            agent=recommendation_agent
        )
    ]
    
    if include_people or any(term in query.lower() for term in ["actor", "actress", "director", "cast", "star"]):
        tasks.insert(1, Task(
            description="Find information about relevant people mentioned in the query",
            expected_output="Details about actors, directors or other key personnel",
            agent=people_agent
        ))
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

# ========== HELPER FUNCTIONS FOR ENTITY EXTRACTION ==========

def extract_title_from_query(query: str) -> str:
    """Extracts the movie/show title from the query"""
    # Common patterns that might precede a title
    patterns = ["about", "of", "for", "information on", "details of", "tell me about", 
                "what is", "how good is", "review of", "synopsis of", "plot of", "describe"]
    
    # Try to extract based on patterns
    for pattern in patterns:
        if f" {pattern} " in f" {query.lower()} ":
            parts = query.lower().split(f" {pattern} ", 1)
            if len(parts) > 1 and parts[1]:
                return parts[1].strip()
    
    # If no pattern was found, try removing common question words
    clean_query = query.lower()
    for prefix in ["tell me", "how is", "what is", "what about", "how about"]:
        clean_query = clean_query.replace(prefix, "").strip()
    
    return clean_query

def extract_person_from_query(query: str) -> str:
    """Extracts a person's name from the query"""
    # Common patterns that might indicate a person reference
    patterns = ["who is", "about", "information on", "details about", "tell me about", 
                "actor", "actress", "director", "who played", "who directed"]
    
    # Try to extract based on patterns
    for pattern in patterns:
        if f" {pattern} " in f" {query.lower()} ":
            parts = query.lower().split(f" {pattern} ", 1)
            if len(parts) > 1 and parts[1]:
                return parts[1].strip()
    
    # If no clear indicator, return the query without common prefixes
    clean_query = query.lower()
    for prefix in ["tell me about", "who is", "information about", "details about"]:
        clean_query = clean_query.replace(prefix, "").strip()
    
    return clean_query

# ========== MAIN INTERFACE FUNCTION ==========

def process_entertainment_query(user_query: str, include_people: bool = False) -> str:
    """Main function to process user queries by intent"""
    try:
        # Determine the intent of the query
        intent = classify_query_intent(user_query)
        
        # Route to appropriate handler based on intent
        if intent == "movie_info":
            return get_movie_information(user_query)
        elif intent == "person_info":
            return get_person_information(user_query)
        elif intent == "trending":
            return get_trending_information(user_query)
        elif intent == "genre_specific":
            return get_genre_recommendations(user_query)
        elif intent == "reviews":
            return get_review_information(user_query)
        else:
            # Default to standard recommendations
            return get_recommendations(user_query, include_people)
    except Exception as e:
        # Error handling
        print(f"Error processing query: {str(e)}")
        return f"""# Sorry, there was an error processing your request

I encountered a problem while trying to answer: "{user_query}"

Error details: {str(e)}

Please try:
1. Rephrasing your question
2. Being more specific about movie or show titles
3. Using simpler queries until the system is more stable

Thank you for your understanding!
"""

# Redirect original function to use the new system
def get_recommendations(user_query: str, include_people: bool = False) -> str:
    """Legacy function maintained for compatibility"""
    return process_entertainment_query(user_query, include_people)

# Example query - in a real application, this would come from user input
if __name__ == "__main__":
    user_query = "Tell me about Interstellar"
    result = process_entertainment_query(user_query)
    print(result)