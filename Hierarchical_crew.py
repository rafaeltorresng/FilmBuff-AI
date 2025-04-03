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

# Carregar variáveis de ambiente
load_dotenv()

# ========== CACHE SYSTEM ==========

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
                print(f"Erro ao carregar cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Erro ao salvar cache: {e}")
    
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

# Instanciando o cache
query_cache = QueryCache()

# ========== QUERY CLASSIFIER ==========

def classify_query_intent(query: str) -> Tuple[str, List[str]]:
    """
    Classifica a consulta para determinar seu tipo e agentes necessários.
    Retorna: (tipo de consulta, lista de agentes necessários)
    """
    query_lower = query.lower()
    
    # Padrões simples que podem ser identificados com regras
    trending_patterns = ["em alta", "tendência", "trending", "popular", "melhores filmes", "novidades"]
    search_patterns = ["encontre", "busque", "procure", "pesquise", "lista de"]
    detail_patterns = ["detalhes", "informações", "sobre o filme", "sobre a série", "sinopse", "elenco", "quando foi lançado"]
    recommendation_patterns = ["recomende", "similar", "parecido", "como", "do mesmo estilo", "do mesmo gênero"]
    person_patterns = ["ator", "atriz", "diretor", "quem", "personagem", "artista", "elenco"]
    
    # Identificadores de filmes/séries específicos
    specific_title = any(title in query_lower for title in ["star wars", "vingadores", "harry potter", "senhor dos anéis", "game of thrones"])
    
    # Classificação do tipo de consulta
    query_type = "simple"  # padrão
    required_agents = []
    
    # Detecção de consulta de tendências
    if any(pattern in query_lower for pattern in trending_patterns):
        query_type = "trending"
        return query_type, []  # Não precisa de agentes, o manager pode responder
    
    # Detecção de busca simples
    elif any(pattern in query_lower for pattern in search_patterns) and not specific_title:
        query_type = "search"
        return query_type, []  # Não precisa de agentes, o manager pode responder
    
    # Se pede detalhes sobre um título específico
    elif any(pattern in query_lower for pattern in detail_patterns) or specific_title:
        query_type = "complex"
        required_agents.append("details_agent")
        
        if any(pattern in query_lower for pattern in recommendation_patterns):
            required_agents.append("recommendation_agent")
            
    # Se pede recomendações
    elif any(pattern in query_lower for pattern in recommendation_patterns):
        query_type = "complex"
        required_agents.append("recommendation_agent")
        
    # Se pergunta sobre pessoas
    elif any(pattern in query_lower for pattern in person_patterns):
        query_type = "complex"
        required_agents.append("people_agent")
    
    # Consulta complexa ou ambígua
    else:
        query_type = "complex"
        required_agents.append("research_agent")
    
    # Se não tiver nenhum agente identificado em uma consulta complexa, use o research_agent como padrão
    if query_type == "complex" and not required_agents:
        required_agents.append("research_agent")
    
    return query_type, required_agents

# ========== MANAGER AGENT (SIMPLIFIED) ==========

# Manager Agent otimizado com instruções reduzidas
manager_agent = Agent(
    role="Entertainment Concierge",
    goal="Analisar consultas sobre filmes e programas de TV e fornecer ou coordenar respostas adequadas",
    tools=[
        SearchMoviesTool(),
        SearchTVShowsTool(),
        GetTrendingContentTool(),
        SearchPersonTool(),
    ],
    backstory="Especialista em entretenimento que fornece informações sobre filmes e programas de TV, delegando consultas complexas a especialistas quando necessário.",
    verbose=True,
    llm=llm
)

# ========== FUNÇÕES DE TAREFAS OTIMIZADAS ==========

def create_simple_manager_task(query: str) -> Task:
    """Cria uma task simples para o manager responder diretamente"""
    return Task(
        description=f"""
        Responda diretamente à consulta do usuário: "{query}"
        
        Use suas ferramentas disponíveis para fornecer uma resposta direta e completa.
        Formate sua resposta de forma clara e organizada para apresentação ao usuário.
        """,
        expected_output="Uma resposta direta e informativa para a consulta do usuário",
        agent=manager_agent
    )

def create_delegation_manager_task(query: str, required_agents: List[str]) -> Task:
    """Cria uma task para o manager delegar para agentes específicos"""
    # Determina quais agentes devem ser incluídos nas instruções
    agents_info = ""
    
    if "research_agent" in required_agents:
        agents_info += """
        RESEARCH AGENT: [Instruções para pesquisar filmes ou programas de TV relevantes para a consulta]
        """
    
    if "details_agent" in required_agents:
        agents_info += """
        DETAILS AGENT: [Instruções para fornecer informações detalhadas sobre o conteúdo identificado]
        """
    
    if "recommendation_agent" in required_agents:
        agents_info += """
        RECOMMENDATION AGENT: [Instruções para recomendar conteúdo similar ou relacionado]
        """
    
    if "people_agent" in required_agents:
        agents_info += """
        PEOPLE AGENT: [Instruções para fornecer informações sobre pessoas relacionadas]
        """
    
    return Task(
        description=f"""
        Analise esta consulta do usuário: "{query}"
        
        Você deve criar um plano de delegação para os agentes especializados.
        
        Sua resposta DEVE seguir este formato:
        
        ```
        PLANO DE DELEGAÇÃO:
        [Breve descrição do seu plano para responder à consulta]
        
        {agents_info}
        ```
        
        Forneça instruções específicas e claras para cada agente listado acima.
        """,
        expected_output="Um plano de delegação com instruções claras para os agentes especializados",
        agent=manager_agent
    )

def create_optimized_tasks(agent_name: str, query: str, instructions: str) -> Task:
    """Cria tarefas otimizadas para agentes especializados"""
    
    # Limita o tamanho das instruções para economizar tokens
    max_instructions_length = 300
    if len(instructions) > max_instructions_length:
        instructions = instructions[:max_instructions_length] + "..."
    
    task_description = f"""
    Consulta do usuário: "{query}"
    
    Instruções: {instructions}
    
    Forneça uma resposta concisa e informativa.
    """
    
    if agent_name == "research_agent":
        return Task(
            description=task_description,
            expected_output="Lista de conteúdo relevante",
            agent=research_agent
        )
    elif agent_name == "details_agent":
        return Task(
            description=task_description,
            expected_output="Informações detalhadas sobre o conteúdo",
            agent=details_agent
        )
    elif agent_name == "recommendation_agent":
        return Task(
            description=task_description,
            expected_output="Recomendações personalizadas",
            agent=recommendation_agent
        )
    elif agent_name == "people_agent":
        return Task(
            description=task_description,
            expected_output="Informações sobre pessoas da indústria",
            agent=people_agent
        )
    else:
        raise ValueError(f"Agente desconhecido: {agent_name}")

def create_synthesis_task(query: str, results: str) -> Task:
    """Cria tarefa para sintetizar resultados dos agentes"""
    
    # Trunca os resultados se forem muito longos para economizar tokens
    max_results_length = 2000
    if len(results) > max_results_length:
        results = results[:max_results_length] + "\n...[resultados truncados para economizar tokens]..."
    
    return Task(
        description=f"""
        Consulta original do usuário: "{query}"
        
        Resultados dos agentes especializados:
        {results}
        
        Sintetize essas informações em uma resposta final bem formatada.
        Elimine redundâncias e organize as informações de forma lógica.
        """,
        expected_output="Resposta final sintetizada",
        agent=manager_agent
    )

# ========== PROCESSAMENTO OTIMIZADO ==========

def process_optimized_query(query: str) -> str:
    """Processa consultas de forma otimizada para economizar tokens"""
    
    # Verifica cache primeiro
    cached_result = query_cache.get(query)
    if cached_result:
        print("Usando resultado em cache")
        return cached_result
    
    try:
        print(f"Processando consulta: '{query}'")
        
        # Classifica a consulta
        query_type, required_agents = classify_query_intent(query)
        
        # Para consultas simples, use apenas o manager
        if query_type in ["simple", "trending", "search"]:
            print(f"Classificada como consulta simples: {query_type}")
            
            crew = Crew(
                agents=[manager_agent],
                tasks=[create_simple_manager_task(query)],
                process=Process.sequential,
                verbose=True
            )
            
            result = str(crew.kickoff())
            query_cache.set(query, result)
            return result
        
        # Para consultas complexas, use o sistema hierárquico
        print(f"Classificada como consulta complexa, requer agentes: {', '.join(required_agents)}")
        
        # 1. Manager cria plano de delegação
        manager_crew = Crew(
            agents=[manager_agent],
            tasks=[create_delegation_manager_task(query, required_agents)],
            process=Process.sequential,
            verbose=True
        )
        
        delegation_plan = str(manager_crew.kickoff())
        
        # 2. Extrai instruções para agentes
        tasks = []
        for agent_name in required_agents:
            pattern = re.compile(f"{agent_name.replace('_agent', '').upper()} AGENT:(.*?)(?:RESEARCH AGENT:|DETAILS AGENT:|RECOMMENDATION AGENT:|PEOPLE AGENT:|$)", re.DOTALL | re.IGNORECASE)
            match = pattern.search(delegation_plan)
            
            if match:
                instructions = match.group(1).strip()
                tasks.append(create_optimized_tasks(agent_name, query, instructions))
        
        # Se não conseguir extrair tarefas, use fallback
        if not tasks:
            print("Usando tarefas fallback baseadas no tipo de consulta")
            for agent_name in required_agents:
                default_instruction = f"Forneça informações sobre '{query}'"
                tasks.append(create_optimized_tasks(agent_name, query, default_instruction))
        
        # 3. Executa tarefas dos agentes especializados
        specialist_crew = Crew(
            agents=[research_agent, details_agent, recommendation_agent, people_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        specialist_results = str(specialist_crew.kickoff())
        
        # 4. Manager sintetiza os resultados
        synthesis_crew = Crew(
            agents=[manager_agent],
            tasks=[create_synthesis_task(query, specialist_results)],
            process=Process.sequential,
            verbose=True
        )
        
        final_result = str(synthesis_crew.kickoff())
        
        # Salva no cache
        query_cache.set(query, final_result)
        
        return final_result
        
    except Exception as e:
        print(f"Erro ao processar consulta: {str(e)}")
        return f"""# Erro ao processar consulta

Ocorreu um problema ao processar: "{query}"
Erro: {str(e)}

Por favor, tente reformular sua pergunta.
"""

# ========== FUNÇÃO PRINCIPAL ==========

if __name__ == "__main__":
    print("Testando sistema hierárquico otimizado para economia de tokens...")
    
    # Teste com uma consulta simples
    print("\n\n==== TESTE 1: CONSULTA SIMPLES ====")
    simple_query = "Quais filmes estão em alta esta semana?"
    simple_result = process_optimized_query(simple_query)
    print("\nResultado da consulta simples:")
    print(simple_result)
    
    # Teste com uma consulta complexa 
    print("\n\n==== TESTE 2: CONSULTA COMPLEXA ====")
    complex_query = "Quero detalhes completos sobre Star Wars: O Império Contra-Ataca"
    complex_result = process_optimized_query(complex_query)
    print("\nResultado da consulta complexa:")
    print(complex_result)