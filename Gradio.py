import os
import gradio as gr
import time
import json
import re
import threading
from datetime import datetime, timedelta
from Hierarchical_crew import process_optimized_query, query_cache

# Version information
VERSION = "1.1.0"
LAST_UPDATED = "April 2025"

# System configuration
MAX_TOKENS = 50  # Token limit for user input

# Theme and style settings
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

# System settings
SYSTEM_NAME = "Film Buff"
SYSTEM_AVATAR = "https://api.dicebear.com/9.x/micah/svg?flip=True"
USER_AVATAR = "https://api.dicebear.com/9.x/micah/svg?flip=True"
HISTORY_FILE = "chat_history.json"

# Examples of quick suggestion questions
EXAMPLES = [
    "What movies are trending this week?",
    "Recommend me psychological horror movies with good ratings",
    "I want detailed information about Star Wars: The Empire Strikes Back",
    "Who directed Pulp Fiction and what other movies did they make?", 
    "Movies similar to Interstellar"
]

# ========== TOKEN COUNTING FUNCTION USING OPENAI'S TOKENIZER ==========

def count_tokens(text):
    """Count tokens using OpenAI's tiktoken tokenizer"""
    try:
        import tiktoken
        # Initialize the tokenizer for the appropriate model (e.g., gpt-3.5-turbo)
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # Encode and count tokens
        tokens = encoder.encode(text)
        return len(tokens)
    except ImportError:
        # Fallback if tiktoken is not installed
        print("tiktoken not installed, falling back to estimation")
        # Simple fallback estimation
        if not text:
            return 0
        words = text.strip().split()
        return len(words) * 4 // 3  # Rough estimation
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Even simpler fallback
        return len(text) // 4  # Very rough estimation

# ========== RATE LIMITING ==========

class RateLimiter:
    def __init__(self, max_calls=5, period=60):
        self.max_calls = max_calls  # Maximum calls in period
        self.period = period  # Period in seconds
        self.calls = []  # List of timestamps
        self.lock = threading.Lock()  # Thread safety
    
    def can_proceed(self) -> bool:
        """Check if a new call can proceed under rate limits"""
        now = time.time()
        with self.lock:
            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.period]
            
            # Check if under limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            else:
                return False
    
    def time_until_available(self) -> int:
        """Returns seconds until a new call can proceed"""
        if self.can_proceed():
            return 0
            
        with self.lock:
            now = time.time()
            oldest_call = min(self.calls)
            return int(self.period - (now - oldest_call)) + 1

# Initialize rate limiter (5 queries per minute)
rate_limiter = RateLimiter(max_calls=5, period=60)

# ========== ENHANCED FUNCTIONS ==========

# Function to save chat history
def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Function to load chat history
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

# Function for content formatting and enhancement
def enhance_content(text):
    # Format movie/show titles with bold
    text = re.sub(r"\"([^\"]+)\"", r"**\1**", text)
    
    # Format ratings for better visibility
    text = re.sub(r"(\d\.\d\/10)", r"**\1**", text)
    
    # Format TMDb links
    text = text.replace("https://www.themoviedb.org", "[TMDb](https://www.themoviedb.org")
    text = text.replace(")", ")]")
    
    return text

# Function to handle API errors specifically
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

# Function to validate user input - UPDATED with proper tokenizer
def validate_input(message):
    """Validate user input against token limits and other constraints"""
    # Check if empty
    if not message or not message.strip():
        return False, "Please enter a question about movies or TV shows."
    
    # Check token count using OpenAI's tokenizer
    token_count = count_tokens(message)
    if token_count > MAX_TOKENS:
        return False, f"Your message exceeds the {MAX_TOKENS} token limit (exact count: {token_count} tokens). Please shorten your request."
    
    # All checks passed
    return True, ""

# Function to process messages - UPDATED with token limit validation
def process_message(message, history):
    # Validate the user input
    valid, error_msg = validate_input(message)
    if not valid:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ {error_msg}"}
        ], get_cache_stats(), get_cache_timestamp(), ""
    
    # Check rate limiting
    if not rate_limiter.can_proceed():
        wait_time = rate_limiter.time_until_available()
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ **Rate limit exceeded**. Please wait {wait_time} seconds before sending another query to protect our API usage."}
        ], get_cache_stats(), get_cache_timestamp(), ""
    
    # Spinner during processing
    yield history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Processing your query... ⌛"}
    ], get_cache_stats(), get_cache_timestamp(), ""
    
    # Process the query using the optimized hierarchical system
    try:
        response = process_optimized_query(message)
        
        # Check if response is in cache
        is_cached = query_cache.get(message) is not None
        cache_indicator = " (response from cache)" if is_cached else ""
        
        # Apply content enhancement
        enhanced_response = enhance_content(response)
        
        # Build updated history
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": enhanced_response + cache_indicator}
        ]
        
        # Return updated history, statistics, and empty input
        yield updated_history, get_cache_stats(), get_cache_timestamp(), ""
        
        # Save chat history
        save_history(updated_history)
        
    except Exception as e:
        # Handle API errors specifically
        error_message = handle_api_error(e)
        
        # Build error history
        error_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ {error_message}"}
        ]
        
        # Return error history and clear input
        yield error_history, get_cache_stats(), get_cache_timestamp(), ""
        
        # Save chat history with error
        save_history(error_history)

# Function to load an example question
def load_example(example):
    return example

# Function to clear history and cache
def clear_history_and_cache():
    query_cache.cache = {}
    query_cache.save_cache()
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], [{"role": "assistant", "content": "History and cache cleared successfully!"}], get_cache_stats(), get_cache_timestamp()

# Function to clear only chat history
def clear_chat_only():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return [], [{"role": "assistant", "content": "Chat history cleared. Cache remains intact."}], get_cache_stats(), get_cache_timestamp()

# Function to display cache statistics
def get_cache_stats():
    if query_cache.cache:
        num_entries = len(query_cache.cache)
        return f"Current cache: {num_entries} stored queries"
    else:
        return "Current cache: empty"

# Function to get cache timestamp
def get_cache_timestamp():
    if os.path.exists("query_cache.pkl"):
        cache_timestamp = datetime.fromtimestamp(os.path.getmtime("query_cache.pkl"))
        return f"Last update: {cache_timestamp.strftime('%m/%d/%Y %H:%M:%S')}"
    return "Cache not yet created"


# Function to get rate limit status
def get_rate_limit_status():
    calls_used = len(rate_limiter.calls)
    calls_left = rate_limiter.max_calls - calls_used
    reset_time = datetime.now() + timedelta(seconds=rate_limiter.period)
    
    if calls_left <= 1:
        return f"⚠️ Rate limit: {calls_left}/{rate_limiter.max_calls} queries left"
    else:
        return f"Rate limit: {calls_left}/{rate_limiter.max_calls} queries available"

# ========== MAIN INTERFACE ==========

with gr.Blocks(theme=THEME, title=SYSTEM_NAME) as demo:
    # Add enhanced CSS styling with modern fonts
    gr.HTML("""
    <style>
    /* Import modern fonts from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap');
    
    /* Set base font for the entire application */
    body, .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-feature-settings: 'liga' 1, 'calt' 1; /* enable font ligatures */
    }
    
    /* Apply Montserrat to headers */
    h1, h2, h3, h4, h5, h6, .header {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Apply DM Sans to interface elements */
    button, .label-wrap span, .panel-header span, .accordion {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500;
    }
    
    /* Chatbot message styling */
    .message {
        font-family: 'Inter', sans-serif !important;
        line-height: 1.5;
        font-size: 15px !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
      .mobile-stack { flex-direction: column !important; }
      .mobile-full { width: 100% !important; }
    }
    
    /* Enhanced UI styling */
    .container { 
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Film-themed styling */
    .header {
        background: rgba(0,0,0,0.2);
        border-radius: 15px;
        padding: 10px;
        border-bottom: 3px solid rgba(79, 70, 229, 0.6);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chatbot-container {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    .chatbot-container:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
    }
    
    /* Better message styling */
    .message-bot, .message-user {
        padding: 12px !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
        position: relative !important;
    }
    
    /* Pulse animation for processing message */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .processing {
        animation: pulse 1.5s infinite;
    }
    
    /* Better accordions */
    .accordion {
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }
    .accordion:hover {
        border-left: 3px solid #4f46e5;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 70, 229, 0.6);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 70, 229, 0.8);
    }
    
    /* Improve buttons */
    button {
        transition: all 0.2s ease !important;
        transform: translateY(0) !important;
        letter-spacing: 0.01em;
    }
    button:active {
        transform: translateY(2px) !important;
    }
    
    /* Improved input styling */
    input, textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
    }
    
    /* Hollywood style footer */
    .hollywood-footer {
        background: linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(255,255,255,0.1) 50%, rgba(0,0,0,0) 100%);
        padding-top: 5px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Add subtle clapper board pattern to sidebar */
    .sidebar-pattern {
        background-image: repeating-linear-gradient(
            -45deg, 
            rgba(255,255,255,0.03) 0px, 
            rgba(255,255,255,0.03) 10px, 
            transparent 10px, 
            transparent 20px
        );
    }
    
    /* Enhanced markdown styling in messages */
    .message-wrap p, .message-wrap li {
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }
    
    .message-wrap strong, .message-wrap b {
        font-weight: 600 !important;
    }
    
    /* Make code sections more cinematic */
    .message-wrap code {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(0,0,0,0.05) !important;
        border-radius: 4px !important;
        padding: 2px 4px !important;
    }
    </style>
    """)
    
    # Header with film reel design
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
    
    # Main chat and statistics panel
    with gr.Row(elem_classes="mobile-stack"):
        # Chat panel (70% width)
        with gr.Column(scale=7, elem_classes="mobile-full"):
            # Initialize chatbot with history or welcome message
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
            
            # Input area - Styled better
            with gr.Row():
                msg = gr.Textbox(
                    placeholder=f"Ask something about movies or TV shows (max {MAX_TOKENS} tokens)...",
                    show_label=False,
                    container=False,
                    scale=9,
                    min_width=100
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Examples with better styling
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Question suggestions",
                fn=load_example,
                outputs=msg,
                examples_per_page=5
            )
        
        # Side panel with film pattern (30% width)
        with gr.Column(scale=3, elem_classes="mobile-full sidebar-pattern"):
            # All accordions closed by default
            with gr.Accordion("📊 Statistics & Controls", open=False, elem_classes="accordion"):
                # Rate limit status
                rate_limit = gr.Markdown(get_rate_limit_status, every=5)
                
                # Cache statistics with automatic updates
                cache_stats = gr.Markdown(get_cache_stats())
                cache_time = gr.Markdown(get_cache_timestamp())
                
                # Control buttons with better layout and styling
                gr.Markdown("### Actions")
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                    clear_all_btn = gr.Button("🧹 Clear All", variant="secondary", scale=1)
            
            # Visual feedback accordion - unchanged
            with gr.Accordion("🔄 Status", open=False, elem_classes="accordion"):
                gr.Markdown("""
                Statistics are updated automatically when:
                - A new query is processed
                - The cache is cleared
                - A response is retrieved from cache
                """)
            
            # Information section - unchanged
            with gr.Accordion("ℹ️ About the System", open=False, elem_classes="accordion"):
                gr.Markdown("""
                ### How It Works
                This system uses a hierarchical architecture of specialized agents:
                
                - **Manager**: Coordinates all other agents
                - **Research**: Finds movies based on specific criteria
                - **Details**: Provides detailed information
                - **Recommendation**: Suggests similar content
                - **People**: Information about actors, directors, etc.
                
                The system optimizes resource usage by responding directly when possible
                or delegating to specialists when necessary.
                """)
            
            # Help & tips section - unchanged
            with gr.Accordion("💡 Tips", open=False, elem_classes="accordion"):
                gr.Markdown(f"""
                ### Getting the Best Results
                
                - **Be specific** when asking about movies or shows
                - Include **year of release** when titles might be ambiguous
                - For recommendations, mention **what you liked** about similar content
                - Try **combining questions** (e.g., "Action movies with Tom Cruise")
                - Keep queries concise (max {MAX_TOKENS} tokens)
                """)
    
    # Hollywood-style footer
    with gr.Row(elem_classes="hollywood-footer"):
        gr.HTML(f"""
        <div style="text-align: center; margin-top: 20px; padding: 10px; color: #a0aec0; font-size: 0.8rem;">
            <div style="display: inline-block; padding: 0 30px; position: relative;">
                {SYSTEM_NAME} v{VERSION} | Last Updated: {LAST_UPDATED} | Built with ❤️ using CrewAI and LLM technology
            </div>
            <br>Data powered by <a href="https://www.themoviedb.org" target="_blank" style="color: #a0aec0; text-decoration: underline;">The Movie Database (TMDb)</a>
        </div>
        """)
    
    # Event logic - Simplified
    # Submit events
    msg.submit(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    submit_btn.click(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time, msg])
    
    # Clear buttons
    clear_chat_btn.click(clear_chat_only, None, [chatbot, chatbot, cache_stats, cache_time])
    clear_all_btn.click(clear_history_and_cache, None, [chatbot, chatbot, cache_stats, cache_time])

# Start the interface
if __name__ == "__main__":
    # Automatically install required dependencies
    import subprocess
    import sys
    
    print("Checking required dependencies...")
    try:
        import gradio
        version = gradio.__version__
        print(f"Gradio version {version} found")
        
        # Check for tiktoken (OpenAI's tokenizer)
        try:
            import tiktoken
            print("tiktoken module available - accurate token counting enabled")
        except ImportError:
            print("tiktoken not found. Installing for accurate token counting...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            print("tiktoken installed successfully!")
        
        # Check for other dependencies
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
    # Add inbrowser=True to automatically open in browser
    demo.launch(share=True, inbrowser=True)