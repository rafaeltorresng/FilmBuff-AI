import os
import gradio as gr
import time
import json
import re
import threading
from datetime import datetime, timedelta
from Crew import process_film_buff_query, query_cache

VERSION = "1.0.0"  
LAST_UPDATED = "April 2025"
MAX_TOKENS = 50 

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray
).set(
    body_background_fill="linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)",
    block_background_fill="#ffffff",
    block_label_background_fill="#f8fafc",
    block_title_text_color="#ffffff",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_secondary_background_fill="#f3f4f6",
    button_secondary_background_fill_hover="#e5e7eb",
    button_secondary_text_color="#111827",
    block_radius="16px",
    button_small_radius="8px",
    input_background_fill="#f9fafb",
    input_border_width="1px",
    input_shadow="0px 2px 4px rgba(0,0,0,0.05)",
    shadow_spread="8px",
)

SYSTEM_NAME = "Film Buff"
SYSTEM_AVATAR = "https://api.dicebear.com/9.x/pixel-art/svg?backgroundType=gradientLinear,solid"
USER_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=FilmBuff&backgroundColor=b6e3f4"
HISTORY_FILE = "chat_history.json"

EXAMPLES = [
    "What movies are trending this week?",
    "Recommend me psychological horror movies with good ratings",
    "I want detailed information about Star Wars: The Empire Strikes Back",
    "Who directed Pulp Fiction and what other movies did they make?", 
    "Movies similar to Interstellar"
]

def count_tokens(text):
    try:
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoder.encode(text)
        return len(tokens)
    except ImportError:
        if not text:
            return 0
        words = text.strip().split()
        return len(words) * 4 // 3  
    except Exception as e:
        return len(text) // 4 

class RateLimiter:
    def __init__(self, max_calls=5, period=60):
        self.max_calls = max_calls  
        self.period = period 
        self.calls = []  
        self.lock = threading.Lock() 
    
    def can_proceed(self) -> bool:
        now = time.time()
        with self.lock:
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            else:
                return False
    
    def time_until_available(self) -> int:
        if self.can_proceed():
            return 0
            
        with self.lock:
            now = time.time()
            oldest_call = min(self.calls)
            return int(self.period - (now - oldest_call)) + 1

rate_limiter = RateLimiter(max_calls=5, period=60)

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

def enhance_content(text):
    text = re.sub(r"\]\s*-\s*⭐", " - ⭐", text)
    text = re.sub(r"(\([0-9]{4}\))\]", r"\1", text)
    text = re.sub(r"\"([^\"]+)\"", r"**\1**", text)
    text = re.sub(r"(\d\.\d\/10)", r"**\1**", text)
    text = text.replace("https://www.themoviedb.org", "[TMDb](https://www.themoviedb.org")
    text = re.sub(r"(\S)\](\s|$|:)", r"\1\2", text)
    return text

def handle_api_error(error):
    error_str = str(error).lower()
    
    if "429" in error_str or "too many requests" in error_str:
        return "The movie database API is currently rate limited. Please try again in a minute."
    elif "connection" in error_str or "timeout" in error_str:
        return "Could not connect to the movie database. Please check your internet connection and try again."
    elif "authentication" in error_str or "api key" in error_str:
        return "API authentication error. Please contact the administrator."
    else:
        return f"An error occurred while processing your request: {str(error)}. Please try rephrasing your question."

def validate_input(message):
    if not message or not message.strip():
        return False, "Please enter a question about movies or TV shows."
    
    token_count = count_tokens(message)
    if token_count > MAX_TOKENS:
        return False, f"Your message exceeds the {MAX_TOKENS} token limit (exact count: {token_count} tokens). Please shorten your request."
    
    return True, ""

def process_message(message, history):
    valid, error_msg = validate_input(message)
    if not valid:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ {error_msg}"}
        ], get_cache_stats(), get_cache_timestamp(), ""
    
    if not rate_limiter.can_proceed():
        wait_time = rate_limiter.time_until_available()
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ **Rate limit exceeded**. Please wait {wait_time} seconds before sending another query to protect our API usage."}
        ], get_cache_stats(), get_cache_timestamp(), ""
    
    yield history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Processing your query... ⌛"}
    ], get_cache_stats(), get_cache_timestamp(), ""
    
    try:
        response = process_film_buff_query(message)
        
        is_cached = query_cache.get(message) is not None
        
        enhanced_response = enhance_content(response)
        
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": enhanced_response}
        ]
        
        cache_stats_text = get_cache_stats()
        if is_cached:
            cache_stats_text = f"{cache_stats_text} (last response from cache)"
        
        yield updated_history, cache_stats_text, get_cache_timestamp(), ""
        
        save_history(updated_history)
        
    except Exception as e:
        error_message = handle_api_error(e)
        
        error_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ {error_message}"}
        ]
        
        yield error_history, get_cache_stats(), get_cache_timestamp(), ""
        
        save_history(error_history)

def load_example(example):
    return example

def clear_history_and_cache():
    query_cache.cache = {}
    query_cache.save_cache()
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], [{"role": "assistant", "content": "History and cache cleared successfully!"}], get_cache_stats(), get_cache_timestamp()

def clear_chat_only():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], [{"role": "assistant", "content": "Chat history cleared. Cache remains intact."}], get_cache_stats(), get_cache_timestamp()

def get_cache_stats():
    if query_cache.cache:
        num_entries = len(query_cache.cache)
        return f"Current cache: {num_entries} stored queries"
    else:
        return "Current cache: empty"

def get_cache_timestamp():
    if os.path.exists("query_cache.pkl"):
        cache_timestamp = datetime.fromtimestamp(os.path.getmtime("query_cache.pkl"))
        return f"Last update: {cache_timestamp.strftime('%m/%d/%Y %H:%M:%S')}"
    return "Cache not yet created"

def get_rate_limit_status():
    calls_used = len(rate_limiter.calls)
    calls_left = rate_limiter.max_calls - calls_used
    reset_time = datetime.now() + timedelta(seconds=rate_limiter.period)
    
    if calls_left <= 1:
        return f"⚠️ Rate limit: {calls_left}/{rate_limiter.max_calls} queries left"
    else:
        return f"Rate limit: {calls_left}/{rate_limiter.max_calls} queries available"

with gr.Blocks(theme=THEME, title=SYSTEM_NAME) as demo:
    gr.HTML("""
    <style>
        .header {
            margin-bottom: 25px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .chatbot-container {
            border-radius: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        }
        
        .accordion {
            margin-bottom: 10px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: none !important;
            background-color: transparent !important;
            border: none !important;
        }
        
        .accordion > div:first-child {
            background-color: rgba(44, 83, 100, 0.7) !important;
            padding: 10px 15px !important;
            font-weight: 500 !important;
            border-bottom: none !important;
            color: white !important;
        }
        
        .accordion > div:nth-child(2) {
            padding: 12px !important;
            background-color: rgba(32, 58, 67, 0.7) !important;
            color: white !important;
        }
        
        .message-bubble {
            padding: 12px 18px !important;
            border-radius: 18px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar-pattern {
            background-color: #0f2027;
            background-image: none;
            border-radius: 16px;
            margin-left: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }
        
        .hollywood-footer {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
            border-radius: 16px;
            margin-top: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chatbot-container > div > div > div {
            animation: fadeIn 0.3s ease-out;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c5c5c5;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        .modern-input input {
            border-radius: 12px !important;
            padding: 12px 18px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s ease !important;
        }
        
        .modern-input input:focus {
            box-shadow: 0 3px 15px rgba(79, 70, 229, 0.15) !important;
            border: 1px solid rgba(79, 70, 229, 0.3) !important;
        }
        
        .send-button {
            border-radius: 12px !important;
            padding: 12px 20px !important;
            transition: all 0.2s ease !important;
            transform: translateY(0);
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        }
        
        .send-button:active {
            transform: translateY(1px);
        }
        
        .stats-container {
            background: transparent;
            border-radius: 0;
            padding: 5px;
            margin-bottom: 10px;
        }
        
        .stat-item {
            margin: 5px 0;
            padding: 8px 12px;
            background: transparent;
            border-radius: 4px;
            border-left: 2px solid rgba(255, 255, 255, 0.2);
            color: #e2e8f0;
        }
        
        .section-title {
            margin-top: 10px !important;
            margin-bottom: 8px !important;
            padding-left: 0 !important;
            border-left: none !important;
            font-weight: 500 !important;
            color: #e2e8f0 !important;
        }
        
        .action-btn {
            transition: all 0.2s ease !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: none !important;
            color: white !important;
        }
        
        .action-btn:hover {
            transform: translateY(-2px);
            background-color: rgba(255, 255, 255, 0.2) !important;
        }
    </style>
    """)
    
    with gr.Row(elem_classes="header"):
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 5px">
            <div style="display: flex; justify-content: center; align-items: center;">
                <div style="margin-right: 20px; font-size: 3rem; animation: pulse 2s infinite ease-in-out;">
                    <span style="display: inline-block; transform-origin: center;">🎬</span>
                </div>
                <div>
                    <h1 style="margin-bottom: 8px; color: white; font-size: 2.6rem; font-weight: 700; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">Film Buff</h1>
                    <p style="margin: 0; color: #e2e8f0; font-weight: 400; font-size: 1.2rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        Your AI-powered movie and TV assistant
                        <span style="display: inline-block; margin-left: 8px; font-size: 1.4rem; animation: wave 2.5s infinite; transform-origin: 70% 70%;">🎥</span>
                    </p>
                </div>
            </div>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            @keyframes wave {
                0% { transform: rotate(0deg); }
                10% { transform: rotate(14deg); }
                20% { transform: rotate(-8deg); }
                30% { transform: rotate(14deg); }
                40% { transform: rotate(-4deg); }
                50% { transform: rotate(10deg); }
                60% { transform: rotate(0deg); }
                100% { transform: rotate(0deg); }
            }
        </style>
        """)
    
    with gr.Row():
        with gr.Column(scale=7):
            initial_history = load_history()
            if not initial_history:
                initial_history = [
                    {"role": "assistant", "content": "Hello! I'm Film Buff, your movies and TV shows assistant. How can I help you today?"}
                ]
            
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=[SYSTEM_AVATAR, USER_AVATAR],
                height=500,
                type="messages",
                render_markdown=True,
                value=initial_history,
                elem_classes="chatbot-container"
            )
            
            with gr.Row(elem_classes="message-input-container"):
                msg = gr.Textbox(
                    placeholder=f"Ask something about movies or TV shows (max {MAX_TOKENS} tokens)...",
                    show_label=False,
                    container=False,
                    scale=9,
                    min_width=100,
                    elem_classes="modern-input"
                )
                submit_btn = gr.Button(
                    "Send", 
                    variant="primary", 
                    scale=1,
                    elem_classes="send-button"
                )
            
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Question suggestions",
                fn=load_example,
                outputs=msg,
                examples_per_page=5
            )
        
        with gr.Column(scale=3, elem_classes="sidebar-pattern"):
            with gr.Accordion("📊 Statistics & Controls", open=False, elem_classes="accordion"):
                with gr.Group(elem_classes="stats-container"):
                    rate_limit = gr.Markdown(get_rate_limit_status, every=5, elem_classes="stat-item")
                    cache_stats = gr.Markdown(get_cache_stats(), elem_classes="stat-item")
                    cache_time = gr.Markdown(get_cache_timestamp(), elem_classes="stat-item")
                
                gr.Markdown("### Actions", elem_classes="section-title")
                with gr.Row():
                    clear_chat_btn = gr.Button(
                        "🗑️ Clear Chat", 
                        variant="secondary", 
                        scale=1,
                        elem_classes="action-btn"
                    )
                    clear_all_btn = gr.Button(
                        "🧹 Clear All", 
                        variant="secondary", 
                        scale=1,
                        elem_classes="action-btn"
                    )
            
            with gr.Accordion("🔄 Status", open=False, elem_classes="accordion"):
                gr.Markdown("""
                Statistics are updated automatically when:
                - A new query is processed
                - The cache is cleared
                - A response is retrieved from cache
                """)
            
            with gr.Accordion("ℹ️ About the System", open=False, elem_classes="accordion"):
                gr.Markdown("""
                ### How It Works
                This system uses a hierarchical architecture of specialized agents:
                
                - **Manager**: Analyzes user intent and delegates to specialized agents
                - **Information**: Provides detailed information about movies and TV shows
                - **Recommendation**: Suggests content based on preferences or similarities
                - **Trends**: Shows what's popular and trending in entertainment
                
                The system optimizes queries through intelligent delegation and caching.
                """)
            
            with gr.Accordion("💡 Tips", open=False, elem_classes="accordion"):
                gr.Markdown(f"""
                ### Getting the Best Results
                
                - **Be specific** when asking about movies or shows
                - Include **year of release** when titles might be ambiguous
                - For recommendations, mention **what you liked** about similar content
                - Try **combining questions** (e.g., "Action movies with Tom Cruise")
                - Keep queries concise (max {MAX_TOKENS} tokens)
                """)
    
    with gr.Row(elem_classes="hollywood-footer"):
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; color: #e2e8f0; font-size: 0.9rem;">
            <div style="display: inline-block; padding: 0 30px; position: relative;">
                <span style="font-weight: 500;">Film Buff</span> <span style="opacity: 0.8;">v1.0.0</span> 
                <span style="margin: 0 8px;">•</span> 
                Last Updated: <span style="font-weight: 500;">April 2025</span>
                <span style="margin: 0 8px;">•</span>
                Built with <span style="color: #ff6b6b;">❤️</span> using CrewAI and LLM technology
            </div>
            <br>
            <div style="margin-top: 8px; opacity: 0.7;">
                Data powered by <a href="https://www.themoviedb.org" target="_blank" style="color: #e2e8f0; text-decoration: underline; transition: all 0.2s ease;">The Movie Database (TMDb)</a>
            </div>
        </div>
        """)
    
    msg.submit(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    submit_btn.click(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    
    clear_chat_btn.click(clear_chat_only, None, [chatbot, chatbot, cache_stats, cache_time])
    clear_all_btn.click(clear_history_and_cache, None, [chatbot, chatbot, cache_stats, cache_time])

if __name__ == "__main__":
    import subprocess
    import sys
    
    print("Checking required dependencies...")
    try:
        import gradio
        version = gradio.__version__
        print(f"Gradio version {version} found")
        
        try:
            import tiktoken
            print("tiktoken module available - accurate token counting enabled")
        except ImportError:
            print("tiktoken not found. Installing for accurate token counting...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            print("tiktoken installed successfully!")
        
        import json
        print("JSON module available")
        import re
        print("Regex module available")
        import threading
        print("Threading module available")
        
    except ImportError as e:
        print(f"Installing missing dependency: {e}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        print("Dependencies installed successfully!")
    
    print(f"Starting {SYSTEM_NAME}...")
    demo.launch(share=True, inbrowser=True)