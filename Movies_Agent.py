import os
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Optional, Dict, List, Union
import requests
from functools import lru_cache
import time
from langchain_openai import ChatOpenAI

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

def make_tmdb_request(endpoint, params=None):
    if params is None:
        params = {}
    
    params["api_key"] = TMDB_API_KEY
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
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

@lru_cache(maxsize=128)
def get_tv_genre_id(genre_name: str) -> Optional[int]:
    genres = {
        "Action & Adventure": 10759, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18,
        "Family": 10751, "Kids": 10762, "Mystery": 9648,
        "News": 10763, "Reality": 10764, "Sci-Fi & Fantasy": 10765,
        "Soap": 10766, "Talk": 10767, "War & Politics": 10768
    }
    
    normalized_genre = genre_name.title()
    if normalized_genre in genres:
        return genres[normalized_genre]
    
    for key, value in genres.items():
        if key.lower() in genre_name.lower():
            return value
            
    return None

class SearchMoviesTool(BaseTool):
    name: str = "search_movies"
    description: str = "Search for movies based on keywords, year, or genre."
    
    def _run(self, query: str, max_results: int = 5, year: Optional[int] = None, genre: Optional[str] = None) -> List[Dict]:
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

class SearchTVShowsTool(BaseTool):
    name: str = "search_tv_shows"
    description: str = "Search for TV shows based on keywords, year, or genre."
    
    def _run(self, query: str, max_results: int = 5, year: Optional[int] = None, genre: Optional[str] = None) -> List[Dict]:
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

class GetMovieDetailsTool(BaseTool):
    name: str = "get_movie_details"
    description: str = "Get detailed information about a specific movie by ID."
    
    def _run(self, movie_id: int) -> Dict:
        data = make_tmdb_request(
            f"movie/{movie_id}",
            {"language": "en-US", "append_to_response": "credits,similar,reviews,videos"}
        )
        
        if "error" in data:
            return {"error": data["error"]}
        
        trailer = ""
        for video in data.get("videos", {}).get("results", []):
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer = f"https://www.youtube.com/watch?v={video['key']}"
                break
        
        director = next((c["name"] for c in data.get("credits", {}).get("crew", []) 
                       if c.get("job") == "Director"), "Unknown")
        
        writers = [c["name"] for c in data.get("credits", {}).get("crew", [])
                  if c.get("job") in ["Writer", "Screenplay"]]
        
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

class GetTVShowDetailsTool(BaseTool):
    name: str = "get_tv_show_details"
    description: str = "Get detailed information about a specific TV show by ID."
    
    def _run(self, show_id: int) -> Dict:
        data = make_tmdb_request(
            f"tv/{show_id}", 
            {"language": "en-US", "append_to_response": "credits,similar,reviews,videos"}
        )
        
        if "error" in data:
            return {"error": data["error"]}
        
        trailer = ""
        for video in data.get("videos", {}).get("results", []):
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer = f"https://www.youtube.com/watch?v={video['key']}"
                break
        
        creators = [p["name"] for p in data.get("created_by", [])]
        
        seasons = []
        for season in data.get("seasons", []):
            if season.get("season_number") > 0:
                seasons.append({
                    "season_number": season.get("season_number"),
                    "episode_count": season.get("episode_count"),
                    "air_date": season.get("air_date")
                })
        
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

class DiscoverMoviesTool(BaseTool):
    name: str = "discover_movies"
    description: str = "Discover movies based on specific criteria without a keyword search."
    
    def _run(self, genre: Optional[str] = None, year: Optional[int] = None, 
             sort_by: str = "popularity.desc", min_rating: float = 0.0, max_results: int = 5) -> List[Dict]:
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

class DiscoverTVShowsTool(BaseTool):
    name: str = "discover_tv_shows"
    description: str = "Discover TV shows based on specific criteria without a keyword search."
    
    def _run(self, genre: Optional[str] = None, year: Optional[int] = None,
             sort_by: str = "popularity.desc", min_rating: float = 0.0, max_results: int = 5) -> List[Dict]:
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

class GetTrendingContentTool(BaseTool):
    name: str = "get_trending_content"
    description: str = "Get trending movies, TV shows, or people."
    
    def _run(self, media_type: str = "all", time_window: str = "week", max_results: int = 5) -> List[Dict]:
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

class SearchPersonTool(BaseTool):
    name: str = "search_person"
    description: str = "Search for actors, directors, or other film industry personalities."
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict]:
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

class GetPersonDetailsTool(BaseTool):
    name: str = "get_person_details"
    description: str = "Get detailed information about a specific person by ID."
    
    def _run(self, person_id: int) -> Dict:
        data = make_tmdb_request(
            f"person/{person_id}",
            {"language": "en-US", "append_to_response": "movie_credits,tv_credits"}
        )
        
        if "error" in data:
            return {"error": data["error"]}
        
        notable_movies = []
        if data.get("movie_credits", {}).get("cast"):
            for movie in sorted(data["movie_credits"]["cast"], key=lambda x: x.get("popularity", 0), reverse=True)[:5]:
                notable_movies.append({
                    "id": movie["id"],
                    "title": movie["title"],
                    "character": movie.get("character", ""),
                    "year": movie.get("release_date", "").split("-")[0] if movie.get("release_date") else ""
                })
        
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

class FindSimilarContentTool(BaseTool):
    name: str = "find_similar_content"
    description: str = "Find content similar to a specific movie or TV show."
    
    def _run(self, content_id: int, content_type: str, max_results: int = 5) -> List[Dict]:
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

research_agent = Agent(
    role="Content Researcher",
    goal="Find movies and TV shows that match the user's criteria",
    tools=[
        SearchMoviesTool(),
        SearchTVShowsTool(),
        DiscoverMoviesTool(),
        DiscoverTVShowsTool(),
        GetTrendingContentTool()
    ],
    backstory="You are a meticulous researcher specialized in finding audiovisual content based on specific criteria.",
    verbose=True,
    llm=llm
)

details_agent = Agent(
    role="Details Specialist",
    goal="Provide detailed information about movies and TV shows",
    tools=[
        GetMovieDetailsTool(),
        GetTVShowDetailsTool(),
        GetPersonDetailsTool(),
        FindSimilarContentTool()
    ],
    backstory="You are a film and television expert with encyclopedic knowledge about audiovisual productions.",
    verbose=True,
    llm=llm
)

recommendation_agent = Agent(
    role="Recommendation Consultant",
    goal="Create personalized recommendations based on user preferences",
    tools=[
        SearchMoviesTool(),
        SearchTVShowsTool(),
        DiscoverMoviesTool(),
        DiscoverTVShowsTool(),
        FindSimilarContentTool()
    ],
    backstory="You are a renowned film critic, known for recommending movies and shows that perfectly match audience tastes.",
    verbose=True,
    llm=llm
)

people_agent = Agent(
    role="People Specialist",
    goal="Find information about actors, directors, and other film industry personalities",
    tools=[
        SearchPersonTool(),
        GetPersonDetailsTool()
    ],
    backstory="You are a celebrity expert with deep knowledge about film industry professionals and their careers.",
    verbose=True,
    llm=llm
)

entertainment_crew = Crew(
    agents=[research_agent, details_agent, recommendation_agent, people_agent],
    tasks=[],
    process=Process.sequential,
    verbose=True
)

def classify_query_intent(query: str) -> str:
    query = query.lower()
    
    if any(pattern in query for pattern in ["tell me about", "information about", "details of", "details about", 
                                           "what is", "tell me more", "synopsis of", "plot of", "describe"]):
        if any(term in query for term in ["actor", "actress", "director", "who played", "who is", "who directed"]):
            return "person_info"
        return "movie_info"
    
    if any(pattern in query for pattern in ["who is", "actor", "actress", "director", "cast of", "stars in", 
                                          "appeared in", "filmography"]):
        return "person_info"
    
    if any(pattern in query for pattern in ["trending", "popular", "top rated", "best of", "this week", 
                                          "this month", "new releases", "what's hot", "what is popular"]):
        return "trending"
    
    if any(f"best {genre}" in query for genre in ["action", "comedy", "drama", "horror", "sci-fi", "thriller", 
                                               "romance", "documentary", "animation"]):
        return "genre_specific"
    
    if any(pattern in query for pattern in ["review", "rating", "score", "how good is", "worth watching"]):
        return "reviews"
    
    return "recommendations"

def get_movie_information(query: str) -> str:
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

def extract_title_from_query(query: str) -> str:
    patterns = ["about", "of", "for", "information on", "details of", "tell me about", 
                "what is", "how good is", "review of", "synopsis of", "plot of", "describe"]
    
    for pattern in patterns:
        if f" {pattern} " in f" {query.lower()} ":
            parts = query.lower().split(f" {pattern} ", 1)
            if len(parts) > 1 and parts[1]:
                return parts[1].strip()
    
    clean_query = query.lower()
    for prefix in ["tell me", "how is", "what is", "what about", "how about", "i want information"]:
        clean_query = clean_query.replace(prefix, "").strip()
    
    return clean_query

def extract_person_from_query(query: str) -> str:
    patterns = ["who is", "about", "information on", "details about", "tell me about", 
                "actor", "actress", "director", "who played", "who directed"]
    
    for pattern in patterns:
        if f" {pattern} " in f" {query.lower()} ":
            parts = query.lower().split(f" {pattern} ", 1)
            if len(parts) > 1 and parts[1]:
                return parts[1].strip()
    
    clean_query = query.lower()
    for prefix in ["tell me about", "who is", "information about", "details about"]:
        clean_query = clean_query.replace(prefix, "").strip()
    
    return clean_query

def get_person_information(query: str) -> str:
    person_name = extract_person_from_query(query)
    
    tasks = [
        Task(
            description=f"Find information about '{person_name}' or closest match",
            expected_output="Basic details about the person including their ID",
            agent=people_agent
        ),
        Task(
            description=f"Get comprehensive details about '{person_name}' including biography and filmography",
            expected_output="Detailed information about the person formatted for user presentation",
            agent=people_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_trending_information(query: str) -> str:
    time_window = "week"
    if "month" in query.lower() or "year" in query.lower():
        time_window = "day"
    
    media_type = "all"
    if "movie" in query.lower():
        media_type = "movie"
    elif "tv" in query.lower() or "show" in query.lower():
        media_type = "tv"
    elif "person" in query.lower() or "people" in query.lower():
        media_type = "person"
    
    tasks = [
        Task(
            description=f"Find trending {media_type} content for {time_window}",
            expected_output="List of trending content with basic details",
            agent=research_agent
        ),
        Task(
            description="Provide additional context and insights about the trending content",
            expected_output="Enhanced information about the trending content",
            agent=recommendation_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_genre_recommendations(query: str) -> str:
    genre_keywords = ["action", "comedy", "drama", "horror", "sci-fi", "thriller", 
                     "romance", "documentary", "animation"]
    
    genre = next((genre for genre in genre_keywords if genre in query.lower()), "")
    
    tasks = [
        Task(
            description=f"Find highly rated {genre} movies or shows",
            expected_output="List of content matching the genre criteria",
            agent=research_agent
        ),
        Task(
            description=f"Create personalized recommendations for {genre} content based on quality and popularity",
            expected_output="Curated recommendations with reasoning",
            agent=recommendation_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_review_information(query: str) -> str:
    title = extract_title_from_query(query)
    
    tasks = [
        Task(
            description=f"Find '{title}' and gather basic information including its ID",
            expected_output="Basic details about the content",
            agent=research_agent
        ),
        Task(
            description=f"Provide detailed review information about '{title}' including critic and audience reception",
            expected_output="Review information formatted for user presentation",
            agent=details_agent
        )
    ]
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def get_recommendations(query: str, include_people: bool = False) -> str:
    tasks = [
        Task(
            description=f"Research content related to '{query}' to understand user preferences",
            expected_output="Initial findings about relevant content",
            agent=research_agent
        ),
        Task(
            description=f"Generate personalized recommendations based on '{query}'",
            expected_output="Curated list of recommendations with reasoning",
            agent=recommendation_agent
        )
    ]
    
    if include_people:
        tasks.append(
            Task(
                description=f"Find relevant film industry professionals related to '{query}'",
                expected_output="Information about related actors, directors, or other personnel",
                agent=people_agent
            )
        )
    
    entertainment_crew.tasks = tasks
    return entertainment_crew.kickoff()

def process_entertainment_query(user_query: str, include_people: bool = False) -> str:
    try:
        intent = classify_query_intent(user_query)
        
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
            return get_recommendations(user_query, include_people)
    except Exception as e:
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

if __name__ == "__main__":
    print("Testing agents with CrewAI...")
    test_query = "Tell me about Star Wars: The Empire Strikes Back"
    result = process_entertainment_query(test_query)
    print("Resultado:")
    print(result)