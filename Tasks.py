from typing import Dict, List, Any, Optional
from crewai import Task

from Agents import (
    manager_agent, 
    information_agent, 
    recommendation_agent, 
    trends_agent
)

def create_manager_task(query: str) -> Task:
    """
    Creates a task for the Manager Agent to analyze the user's query
    and determine which specialist agent should handle it.
    """
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
    """
    Creates a task for the Information Agent to provide detailed
    information about movies or TV shows.
    """
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
    """
    Creates a task for the Recommendation Agent to provide personalized
    movie and TV show recommendations based on user preferences.
    """
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
    """
    Creates a task for the Trends Agent to provide information about
    currently trending movies and TV shows.
    """
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

def create_retry_information_task(query: str) -> Task:
    """
    Creates a retry task for the Information Agent with more specific instructions
    when the initial response was too short or incomplete.
    """
    return Task(
        description=f"""
        # CRITICAL TASK: Provide Detailed Information
        
        The user asked: "{query}"
        
        ## EXPLICIT INSTRUCTIONS:
        1. Use the fetch_movie_info tool to find the specific movie
        2. Use the fetch_movie_reviews tool if needed for additional context
        3. Format a complete response with ALL available details
        
        Include:
        - Complete title and year
        - Director and main cast
        - Detailed synopsis
        - Rating and popularity
        - Genres and duration
        - Important facts
        
        Make sure to provide a thorough and well-formatted response using markdown.
        """,
        expected_output="Detailed information about the movie or TV show",
        agent=information_agent
    )

def create_retry_trends_task(query: str) -> Task:
    """
    Creates a retry task for the Trends Agent with more specific formatting 
    instructions when the initial response was inadequate.
    """
    return Task(
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

# Dictionary for convenient access to task creation functions
task_creators = {
    'manager': create_manager_task,
    'information': create_information_task,
    'recommendation': create_recommendation_task,
    'trends': create_trends_task,
    'retry_information': create_retry_information_task,
    'retry_trends': create_retry_trends_task
}

# Export all task creation functions
__all__ = [
    'create_manager_task',
    'create_information_task',
    'create_recommendation_task',
    'create_trends_task',
    'create_retry_information_task',
    'create_retry_trends_task',
    'task_creators'
]