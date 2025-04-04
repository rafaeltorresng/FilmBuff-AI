import os
import gradio as gr
import time
import json
import re
import threading
from datetime import datetime, timedelta
# Updated import path to use processor instead of Uptadet_crew
from Crew import process_film_buff_query, query_cache

VERSION = "1.0.0"  
LAST_UPDATED = "April 2025"

MAX_TOKENS = 50 

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="linear-gradient(to right, #0f2027, #203a43, #2c5364)",
    block_background_fill="#ffffff",
    block_label_background_fill="#f7f7f8",
    block_title_text_color="#ffffff",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_secondary_background_fill="#f3f4f6",
    button_secondary_background_fill_hover="#e5e7eb",
    button_secondary_text_color="#111827",
    block_radius="15px",
    button_small_radius="8px"
)

SYSTEM_NAME = "Film Buff"
SYSTEM_AVATAR = "https://api.dicebear.com/9.x/micah/svg?flip=True"
USER_AVATAR = "https://api.dicebear.com/9.x/micah/svg?flip=True"
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
        print("tiktoken not installed, falling back to estimation")
        if not text:
            return 0
        words = text.strip().split()
        return len(words) * 4 // 3  
    except Exception as e:
        print(f"Error counting tokens: {e}")
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
    # Remove square brackets that appear after titles
    text = re.sub(r"\]\s*-\s*⭐", " - ⭐", text)
    
    # Remove standalone square brackets at the end of titles
    text = re.sub(r"(\([0-9]{4}\))\]", r"\1", text)
    
    # Bold quoted text
    text = re.sub(r"\"([^\"]+)\"", r"**\1**", text)
    
    # Bold ratings
    text = re.sub(r"(\d\.\d\/10)", r"**\1**", text)
    
    # Fix TMDb links
    text = text.replace("https://www.themoviedb.org", "[TMDb](https://www.themoviedb.org")
    
    # Remove any remaining single square brackets
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
        # Using process_film_buff_query from processor.py
        response = process_film_buff_query(message)
        
        is_cached = query_cache.get(message) is not None
        cache_indicator = " (response from cache)" if is_cached else ""
        
        enhanced_response = enhance_content(response)
        
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": enhanced_response + cache_indicator}
        ]
        
        yield updated_history, get_cache_stats(), get_cache_timestamp(), ""
        
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

# Initialize Gradio interface
with gr.Blocks(theme=THEME, title=SYSTEM_NAME) as demo:
    # CSS styling (unchanged)
    gr.HTML("""
    <style>
    /* CSS code remains unchanged */
    </style>
    """)
    
    # Header section
    with gr.Row(elem_classes="header"):
        gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 5px">
            <div style="display: flex; justify-content: center; align-items: center;">
                <div style="margin-right: 15px; font-size: 2.5rem;">🎬</div>
                <div>
                    <h1 style="margin-bottom: 5px; color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{SYSTEM_NAME}</h1>
                    <h3 style="margin: 0; color: #e2e8f0; font-weight: 400;">Your AI-powered movie and TV assistant</h3>
                </div>
            </div>
        </div>
        """)
    
    # Main section layout
    with gr.Row(elem_classes="mobile-stack"):
        # Chatbot column
        with gr.Column(scale=7, elem_classes="mobile-full"):
            initial_history = load_history()
            if not initial_history:
                initial_history = [
                    {"role": "assistant", "content": f"Hello! I'm {SYSTEM_NAME}, your movies and TV shows assistant. How can I help you today?"}
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
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder=f"Ask something about movies or TV shows (max {MAX_TOKENS} tokens)...",
                    show_label=False,
                    container=False,
                    scale=9,
                    min_width=100
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Question suggestions",
                fn=load_example,
                outputs=msg,
                examples_per_page=5
            )
        
        # Sidebar column
        with gr.Column(scale=3, elem_classes="mobile-full sidebar-pattern"):
            with gr.Accordion("📊 Statistics & Controls", open=False, elem_classes="accordion"):
                rate_limit = gr.Markdown(get_rate_limit_status, every=5)
                cache_stats = gr.Markdown(get_cache_stats())
                cache_time = gr.Markdown(get_cache_timestamp())
                
                gr.Markdown("### Actions")
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                    clear_all_btn = gr.Button("🧹 Clear All", variant="secondary", scale=1)
            
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
    
    # Footer
    with gr.Row(elem_classes="hollywood-footer"):
        gr.HTML(f"""
        <div style="text-align: center; margin-top: 20px; padding: 10px; color: #a0aec0; font-size: 0.8rem;">
            <div style="display: inline-block; padding: 0 30px; position: relative;">
                {SYSTEM_NAME} v{VERSION} | Last Updated: {LAST_UPDATED} | Built with ❤️ using CrewAI and LLM technology
            </div>
            <br>Data powered by <a href="https://www.themoviedb.org" target="_blank" style="color: #a0aec0; text-decoration: underline;">The Movie Database (TMDb)</a>
        </div>
        """)
    
    # Button actions
    msg.submit(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    submit_btn.click(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    
    clear_chat_btn.click(clear_chat_only, None, [chatbot, chatbot, cache_stats, cache_time])
    clear_all_btn.click(clear_history_and_cache, None, [chatbot, chatbot, cache_stats, cache_time])

# Launch with dependency check
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