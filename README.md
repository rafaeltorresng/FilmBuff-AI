# 🎬 Film Buff - Movie & TV Show AI Assistant

Film Buff is an intelligent assistant for movie and TV show information, built with a hierarchical multi-agent architecture using CrewAI.

## 🌟 Features

- **Smart Query Analysis**: Optimized query classification to handle requests efficiently
- **Specialized Agents**: Multiple expert CrewAI agents that work together to answer complex questions
- **TMDB API Integration**: Real-time data from The Movie Database API
- **Result Caching**: Improved performance with intelligent caching
- **Modern UI**: Clean, responsive interface built with Gradio

## 🖥️  Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Set up your API keys in the .env file:
   ```bash
   TMDB_API_KEY = "your_tmdb_api_key"
   OPENAI_API_KEY = "your_openai_api_key"
4. Run the application:
   ```bash
   python Gradio.py
## 📝 Example Queries

- "What movies are trending this week?"
- "Tell me about Star Wars: The Empire Strikes Back"
- "Recommend sci-fi movies similar to Interstellar"
- "Who directed The Godfather and what other films did they make?"

## 🛠️ Technology Stack
- CrewAI for multi-agent architecture
- LangChain for AI integration
- TMDB API for movie/TV data
- Gradio for the web interface

