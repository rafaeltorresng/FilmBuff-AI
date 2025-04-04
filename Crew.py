import re
import traceback
from crewai import Crew, Process

from Agents import (
    manager_agent,
    information_agent, 
    recommendation_agent, 
    trends_agent
)

from Tasks import (
    create_manager_task,
    create_information_task,
    create_recommendation_task,
    create_trends_task,
    create_retry_information_task,
    create_retry_trends_task
)

class QueryCache:
    """Simple in-memory cache for query results to avoid redundant processing"""
    
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

# Initialize cache
query_cache = QueryCache()

def process_film_buff_query(query: str) -> str:
    """
    Processes a user query using the hierarchical agent structure.
    First routes the query through the manager agent to determine
    which specialist agent should handle it.
    
    Args:
        query: The user's question about movies or TV shows
        
    Returns:
        A formatted response to the user's query
    """
    # Check cache first
    cached_result = query_cache.get(query)
    if cached_result:
        print("Using cached result")
        return cached_result
    
    try:
        print(f"Processing query: '{query}'")
        
        # Step 1: Manager analyzes the query
        manager_crew = Crew(
            agents=[manager_agent],
            tasks=[create_manager_task(query)],
            process=Process.sequential,
            verbose=True
        )
        
        manager_analysis = str(manager_crew.kickoff())
        print(f"Manager analysis: {manager_analysis}")
        
        # Extract target agent from manager's analysis
        target_agent = None
        
        # Check for explicit mentions of agents or intentions in various forms
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
            # If the target agent cannot be clearly determined
            print("Target agent not clearly identified, determining from context")
            
            # Additional context analysis to identify the query type
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
                target_agent = "recommendation_agent"  # Default fallback
                print("Using Recommendation Agent as default")
                specialist_crew = Crew(
                    agents=[recommendation_agent],
                    tasks=[create_recommendation_task(query)],
                    process=Process.sequential,
                    verbose=True
                )
        
        # Step 2: Specialist agent processes the query
        specialist_result = str(specialist_crew.kickoff())
        
        # Step 3: Check if the result is valid
        if len(specialist_result.strip()) < 50:
            print(f"Result from {target_agent} too short, trying to improve")
            
            # Try again with the same agent but more specific instructions
            if target_agent == "trends_agent":
                print("Additional attempt with Trends Agent")
                retry_crew = Crew(
                    agents=[trends_agent],
                    tasks=[create_retry_trends_task(query)],
                    process=Process.sequential,
                    verbose=True
                )
                
                improved_result = str(retry_crew.kickoff())
                
                # Check if the new response is better
                if len(improved_result.strip()) > 50:
                    specialist_result = improved_result
            elif target_agent == "information_agent":
                # Specific retry for Information Agent
                print("Additional attempt with Information Agent")
                retry_crew = Crew(
                    agents=[information_agent],
                    tasks=[create_retry_information_task(query)],
                    process=Process.sequential,
                    verbose=True
                )
                
                improved_result = str(retry_crew.kickoff())
                
                if len(improved_result.strip()) > 50:
                    specialist_result = improved_result
        
        # Cache and return
        query_cache.set(query, specialist_result)
        return specialist_result
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        traceback.print_exc()
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