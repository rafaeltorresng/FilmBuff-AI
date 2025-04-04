import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent

from Tools import (
    IntentClassifierTool,
    FetchMovieInfoTool,
    FetchMovieReviewsTool,
    SearchSimilarMoviesTool,
    RecommendByGenreTool,
    FetchTrendingMoviesTool
)

load_dotenv()

# Configuração do modelo de linguagem
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Definição dos agentes
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

# Para importação conveniente
all_agents = {
    'manager': manager_agent,
    'information': information_agent,
    'recommendation': recommendation_agent,
    'trends': trends_agent
}

# Exportar todos os agentes
__all__ = [
    'llm',
    'manager_agent',
    'information_agent',
    'recommendation_agent',
    'trends_agent',
    'all_agents'
]