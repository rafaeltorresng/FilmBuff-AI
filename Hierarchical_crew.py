import os
import re
import hashlib
import pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from typing import Optional, Dict, List, Union, Tuple
import time

from Movies_Agent import (
    llm, make_tmdb_request,
    SearchMoviesTool, SearchTVShowsTool, GetMovieDetailsTool, GetTVShowDetailsTool,
    DiscoverMoviesTool, DiscoverTVShowsTool, GetTrendingContentTool,
    SearchPersonTool, GetPersonDetailsTool, FindSimilarContentTool,
    research_agent, details_agent, recommendation_agent, people_agent
)

load_dotenv()

class QueryCache:
    def __init__(self, cache_file="query_cache.pkl", expiry_days=7):
        self.cache_file = cache_file
        self.cache = {}
        self.expiry_days = expiry_days
        self.load_cache()
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                self._clean_expired()
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _clean_expired(self):
        now = datetime.now()
        expired_keys = []
        for key, (value, timestamp) in self.cache.items():
            if now - timestamp > timedelta(days=self.expiry_days):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_query_hash(self, query):
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query):
        query_hash = self.get_query_hash(query)
        if query_hash in self.cache:
            value, timestamp = self.cache[query_hash]
            return value
        return None
    
    def set(self, query, result):
        query_hash = self.get_query_hash(query)
        self.cache[query_hash] = (result, datetime.now())
        self.save_cache()

query_cache = QueryCache()

def classify_query_intent(query: str) -> Tuple[str, List[str]]:
    query_lower = query.lower()
    
    trending_patterns = ["trending", "popular", "best movies", "what's hot", "top rated"]
    search_patterns = ["find", "search", "look for", "list of"]
    detail_patterns = ["details", "information", "about the movie", "about the series", "synopsis", "cast", "when was it released"]
    recommendation_patterns = ["recommend", "similar", "like", "same style", "same genre"]
    person_patterns = ["actor", "actress", "director", "who", "character", "artist", "cast"]
    
    specific_title = any(title in query_lower for title in ["star wars", "avengers", "harry potter", "lord of the rings", "game of thrones"])
    
    query_type = "simple"
    required_agents = []
    
    if any(pattern in query_lower for pattern in trending_patterns):
        query_type = "trending"
        return query_type, []
    
    elif any(pattern in query_lower for pattern in search_patterns) and not specific_title:
        query_type = "search"
        return query_type, []
    
    elif any(pattern in query_lower for pattern in detail_patterns) or specific_title:
        query_type = "complex"
        required_agents.append("details_agent")
        
        if any(pattern in query_lower for pattern in recommendation_patterns):
            required_agents.append("recommendation_agent")
            
    elif any(pattern in query_lower for pattern in recommendation_patterns):
        query_type = "complex"
        required_agents.append("recommendation_agent")
        
    elif any(pattern in query_lower for pattern in person_patterns):
        query_type = "complex"
        required_agents.append("people_agent")
    
    else:
        query_type = "complex"
        required_agents.append("research_agent")
    
    if query_type == "complex" and not required_agents:
        required_agents.append("research_agent")
    
    return query_type, required_agents

manager_agent = Agent(
    role="Entertainment Concierge",
    goal="Analyze queries about movies and TV shows and provide or coordinate appropriate responses",
    tools=[
        SearchMoviesTool(),
        SearchTVShowsTool(),
        GetTrendingContentTool(),
        SearchPersonTool(),
    ],
    backstory="Entertainment specialist providing information about movies and TV shows, delegating complex queries to specialists when necessary.",
    verbose=True,
    llm=llm
)

def create_simple_manager_task(query: str) -> Task:
    return Task(
        description=f"""
        Respond directly to the user's query: "{query}"
        
        Use your available tools to provide a direct and complete response.
        Format your response clearly and in an organized way for presentation to the user.
        """,
        expected_output="A direct and informative response to the user's query",
        agent=manager_agent
    )

def create_delegation_manager_task(query: str, required_agents: List[str]) -> Task:
    agents_info = ""
    
    if "research_agent" in required_agents:
        agents_info += """
        RESEARCH AGENT: [Instructions for researching movies or TV shows relevant to the query]
        """
    
    if "details_agent" in required_agents:
        agents_info += """
        DETAILS AGENT: [Instructions for providing detailed information about the identified content]
        """
    
    if "recommendation_agent" in required_agents:
        agents_info += """
        RECOMMENDATION AGENT: [Instructions for recommending similar or related content]
        """
    
    if "people_agent" in required_agents:
        agents_info += """
        PEOPLE AGENT: [Instructions for providing information about related people]
        """
    
    return Task(
        description=f"""
        Analyze this user query: "{query}"
        
        You need to create a delegation plan for the specialized agents.
        
        Your response MUST follow this format:
        
        ```
        DELEGATION PLAN:
        [Brief description of your plan to answer the query]
        
        {agents_info}
        ```
        
        Provide specific and clear instructions for each agent listed above.
        """,
        expected_output="A delegation plan with clear instructions for specialized agents",
        agent=manager_agent
    )

def create_optimized_tasks(agent_name: str, query: str, instructions: str) -> Task:
    max_instructions_length = 300
    if len(instructions) > max_instructions_length:
        instructions = instructions[:max_instructions_length] + "..."
    
    task_description = f"""
    User query: "{query}"
    
    Instructions: {instructions}
    
    Provide a concise and informative response.
    """
    
    if agent_name == "research_agent":
        return Task(
            description=task_description,
            expected_output="List of relevant content",
            agent=research_agent
        )
    elif agent_name == "details_agent":
        return Task(
            description=task_description,
            expected_output="Detailed information about the content",
            agent=details_agent
        )
    elif agent_name == "recommendation_agent":
        return Task(
            description=task_description,
            expected_output="Personalized recommendations",
            agent=recommendation_agent
        )
    elif agent_name == "people_agent":
        return Task(
            description=task_description,
            expected_output="Information about industry people",
            agent=people_agent
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def create_synthesis_task(query: str, results: str) -> Task:
    max_results_length = 2000
    if len(results) > max_results_length:
        results = results[:max_results_length] + "\n...[results truncated to save tokens]..."
    
    return Task(
        description=f"""
        Original user query: "{query}"
        
        Results from specialized agents:
        {results}
        
        Synthesize this information into a well-formatted final response.
        Eliminate redundancies and organize the information logically.
        """,
        expected_output="Synthesized final response",
        agent=manager_agent
    )

def process_optimized_query(query: str) -> str:
    cached_result = query_cache.get(query)
    if cached_result:
        print("Using cached result")
        return cached_result
    
    try:
        print(f"Processing query: '{query}'")
        
        query_type, required_agents = classify_query_intent(query)
        
        if query_type in ["simple", "trending", "search"]:
            print(f"Classified as simple query: {query_type}")
            
            crew = Crew(
                agents=[manager_agent],
                tasks=[create_simple_manager_task(query)],
                process=Process.sequential,
                verbose=True
            )
            
            result = str(crew.kickoff())
            query_cache.set(query, result)
            return result
        
        print(f"Classified as complex query, requires agents: {', '.join(required_agents)}")
        
        manager_crew = Crew(
            agents=[manager_agent],
            tasks=[create_delegation_manager_task(query, required_agents)],
            process=Process.sequential,
            verbose=True
        )
        
        delegation_plan = str(manager_crew.kickoff())
        
        tasks = []
        for agent_name in required_agents:
            pattern = re.compile(f"{agent_name.replace('_agent', '').upper()} AGENT:(.*?)(?:RESEARCH AGENT:|DETAILS AGENT:|RECOMMENDATION AGENT:|PEOPLE AGENT:|$)", re.DOTALL | re.IGNORECASE)
            match = pattern.search(delegation_plan)
            
            if match:
                instructions = match.group(1).strip()
                tasks.append(create_optimized_tasks(agent_name, query, instructions))
        
        if not tasks:
            print("Using fallback tasks based on query type")
            for agent_name in required_agents:
                default_instruction = f"Provide information about '{query}'"
                tasks.append(create_optimized_tasks(agent_name, query, default_instruction))
        
        specialist_crew = Crew(
            agents=[research_agent, details_agent, recommendation_agent, people_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        specialist_results = str(specialist_crew.kickoff())
        
        synthesis_crew = Crew(
            agents=[manager_agent],
            tasks=[create_synthesis_task(query, specialist_results)],
            process=Process.sequential,
            verbose=True
        )
        
        final_result = str(synthesis_crew.kickoff())
        
        query_cache.set(query, final_result)
        
        return final_result
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"""# Error processing query

A problem occurred while processing: "{query}"
Error: {str(e)}

Please try rephrasing your question.
"""

if __name__ == "__main__":
    print("Testing optimized hierarchical system for token savings...")
    
    print("\n\n==== TEST 1: SIMPLE QUERY ====")
    simple_query = "What movies are trending this week?"
    simple_result = process_optimized_query(simple_query)
    print("\nSimple query result:")
    print(simple_result)
    
    print("\n\n==== TEST 2: COMPLEX QUERY ====")
    complex_query = "I want complete details about Star Wars: The Empire Strikes Back"
    complex_result = process_optimized_query(complex_query)
    print("\nComplex query result:")
    print(complex_result)