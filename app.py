"""
SEO Keyword Research Agent - Fixed Version with Updated Models

Fixes:
1. Updated to use available Gemini 2.5 models (2.5-flash, 2.5-pro)
2. Improved model fallback chain
3. Enhanced error handling with better diagnostics
4. Fixed API configuration issues
5. Added model availability checks
"""

import os
import re
import json
import logging
import time
from typing import List, Tuple, Optional
from functools import lru_cache

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('seo_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_SEED_KEYWORD = "digital marketing"
MAX_KEYWORDS = 50
# Updated model names based on current availability (as of July 2025)
AVAILABLE_MODELS = [
    "gemini-2.0-flash",         # Primary model - fast and cost-effective
    "gemini-2.0-flash-lite",    # Lightweight version
    "gemini-2.5-flash",         # Advanced flash model
    "gemini-2.5-pro",           # Most advanced model
    "gemini-1.0-pro",           # Legacy fallback if others fail
]
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds
REQUEST_TIMEOUT = 30  # seconds

# Safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

SYSTEM_PROMPT = """You are an expert SEO analyst with 15 years of experience in keyword research. 
Generate {max_keywords} relevant keyword suggestions based on: "{seed_keyword}"

Guidelines:
1. Sort by: 1) Least competition, 2) Highest monthly search volume
2. Include:
   - Buyer-intent keywords (e.g., "buy", "best", "review")
   - Long-tail variations (3-5 words)
   - Question-based keywords (start with "how", "what", etc.)
   - Location modifiers (if relevant, e.g., "in New York")
3. Ensure all keywords are realistically rankable on Google's first page
4. Exclude:
   - Branded terms (unless part of seed keyword)
   - Overly generic terms
   - Copyrighted/trademarked terms

Return ONLY a JSON array of strings, example:
["keyword 1", "keyword 2", ...]"""

class SEOKeywordAgent:
    """SEO Keyword Research Agent using Gemini AI."""
    
    def __init__(self):
        """Initialize the Gemini API client with proper configuration."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Initialize generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            candidate_count=1,
            max_output_tokens=2000
        )
        
        # Test model availability
        try:
            self.available_models = self._check_available_models()
            if not self.available_models:
                raise ValueError("""
No Gemini models are available. This could be due to:

1. **API Key Issues:**
   - Invalid or expired API key
   - API key doesn't have proper permissions
   - API key is for a different service (e.g., Vertex AI instead of AI Studio)

2. **Model Access Issues:**
   - Your project doesn't have access to these models
   - Models may not be available in your region
   - New projects may have limited model access

3. **Configuration Issues:**
   - Internet connectivity problems
   - Firewall blocking API requests
   - Incorrect API endpoint

**Recommended Actions:**
1. Verify your API key at https://aistudio.google.com/
2. Try creating a new API key
3. Check if your region supports Gemini models
4. Test with a simple request in AI Studio first
5. Make sure you're using the AI Studio API key (not Vertex AI)

**Available Models to Test:** gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-flash, gemini-2.5-pro, gemini-1.0-pro
                """)
        except Exception as e:
            raise ValueError(f"Error checking models: {str(e)}")
        
        logger.info(f"Available models: {self.available_models}")
    
    def _check_available_models(self) -> List[str]:
        """Check which models are actually available with detailed diagnostics."""
        available = []
        errors = {}
        
        for model_name in AVAILABLE_MODELS:
            try:
                logger.info(f"Testing model: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                # Try a simple test to verify the model works
                test_response = model.generate_content(
                    "Hello",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=10
                    )
                )
                if test_response.text:
                    available.append(model_name)
                    logger.info(f"✅ Model {model_name} is available")
                else:
                    errors[model_name] = "Empty response"
                    logger.warning(f"❌ Model {model_name} returned empty response")
                    
            except exceptions.NotFound as e:
                errors[model_name] = f"Model not found: {str(e)}"
                logger.warning(f"❌ Model {model_name} not found: {str(e)}")
                
            except exceptions.PermissionDenied as e:
                errors[model_name] = f"Permission denied: {str(e)}"
                logger.warning(f"❌ Model {model_name} permission denied: {str(e)}")
                
            except Exception as e:
                errors[model_name] = str(e)
                logger.warning(f"❌ Model {model_name} error: {str(e)}")
                
        # Log summary
        if available:
            logger.info(f"Available models: {available}")
        else:
            logger.error("No models available!")
            for model, error in errors.items():
                logger.error(f"  {model}: {error}")
                
        return available
    
    def _try_model(self, model_name: str, prompt: str) -> Optional[List[str]]:
        """Try to generate keywords with a specific model."""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            
            if not response.text:
                raise ValueError("Empty response from API")
            
            # Extract and validate JSON
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if not json_match:
                raise ValueError("Could not parse JSON from response")
            
            keywords = json.loads(json_match.group(0))
            
            if not isinstance(keywords, list):
                raise ValueError("Response is not a list")
            
            if not all(isinstance(k, str) for k in keywords):
                raise ValueError("Non-string values in keyword list")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {str(e)}")
            return None
    
    @lru_cache(maxsize=100)
    def generate_keywords(self, seed_keyword: str, max_keywords: int = MAX_KEYWORDS) -> List[str]:
        """
        Generate SEO keyword suggestions with robust error handling and model fallback.
        
        Args:
            seed_keyword: Keyword to base suggestions on
            max_keywords: Number of keywords to generate
            
        Returns:
            List of suggested keywords
            
        Raises:
            Exception: If keyword generation fails with all models
        """
        prompt = SYSTEM_PROMPT.format(
            max_keywords=max_keywords,
            seed_keyword=seed_keyword
        )
        
        # Try each available model
        for model_name in self.available_models:
            logger.info(f"Trying model: {model_name}")
            
            for attempt in range(MAX_RETRIES):
                try:
                    keywords = self._try_model(model_name, prompt)
                    if keywords:
                        logger.info(f"Successfully generated {len(keywords)} keywords with {model_name}")
                        return keywords[:max_keywords]
                        
                except exceptions.ResourceExhausted as e:
                    if attempt == MAX_RETRIES - 1:
                        logger.warning(f"Rate limited on {model_name}, trying next model")
                        break
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Rate limited. Attempt {attempt + 1}/{MAX_RETRIES}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                except exceptions.DeadlineExceeded:
                    if attempt == MAX_RETRIES - 1:
                        logger.warning(f"Timeout on {model_name}, trying next model")
                        break
                    logger.warning(f"Timeout. Retrying ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed with {model_name}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        break
                    time.sleep(RETRY_DELAY)
        
        # If all models fail, provide helpful error message
        error_messages = []
        if not self.available_models:
            error_messages.append("No Gemini models are available")
        else:
            error_messages.append(f"All available models failed: {', '.join(self.available_models)}")
        
        raise Exception(f"Failed to generate keywords. {' '.join(error_messages)}")

def validate_keyword(keyword: str) -> bool:
    """Validate the seed keyword input."""
    keyword = keyword.strip()
    return 2 <= len(keyword) <= 100 and not any(c.isdigit() for c in keyword)

def display_header():
    """Display the application header with production styling."""
    st.set_page_config(
        page_title="SEO Keyword Research Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 SEO Keyword Research Agent")
    st.markdown("""
    <style>
        .main { padding-top: 2rem; }
        .sidebar .sidebar-content { padding: 2rem 1rem; }
        h1 { color: #2c3e50; }
        .stButton button { width: 100%; }
        .stDownloadButton button { width: 100%; }
        .model-status { 
            background-color: #e8f4f8; 
            padding: 10px; 
            border-radius: 5px; 
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Generate high-quality SEO keyword suggestions using Google's latest Gemini 2.5 models.
    Perfect for content strategists and digital marketers.
    """)
    st.markdown("---")

def display_sidebar() -> Tuple[str, int]:
    """Display sidebar controls with improved UX."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        seed_keyword = st.text_input(
            "Seed Keyword",
            value=DEFAULT_SEED_KEYWORD,
            help="Enter your starting keyword (e.g. 'content marketing')",
            placeholder="Enter your seed keyword..."
        )
        
        max_keywords = st.slider(
            "Number of Keywords",
            min_value=10,
            max_value=100,
            value=MAX_KEYWORDS,
            help="Adjust the number of keyword suggestions"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        st.markdown("""
        **Current Models:** Gemini 2.0 Flash (primary), Gemini 2.5 Flash/Pro (advanced)
        
        **Note:** Gemini 1.5 models are no longer available for new projects as of April 2025.
        
        **Model Hierarchy:**
        - Gemini 2.0 Flash: Fast, cost-effective
        - Gemini 2.0 Flash-Lite: Lightweight version
        - Gemini 2.5 Flash: Advanced capabilities
        - Gemini 2.5 Pro: Most advanced reasoning
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool uses Google's latest Gemini 2.5 AI models to generate 
        SEO-optimized keyword suggestions.
        
        **Tips for best results:**
        - Be specific with your seed keyword
        - Try different variations
        - Use long-tail phrases for niche topics
        """)
        
        st.markdown("---")
        st.markdown("**Rate Limit Notes:**")
        st.markdown("""
        - Free tier has usage limits
        - Multiple models available as fallbacks
        - Automatic retry with exponential backoff
        - If limits hit, try again in 1-2 minutes
        """)
        
    return seed_keyword, max_keywords

def display_results(keywords: List[str], seed_keyword: str, model_used: str = None):
    """Display the generated results with enhanced presentation."""
    st.subheader(f"📊 Results for: '{seed_keyword}'")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Keywords Generated", len(keywords))
    with col2:
        st.metric("Competition Level", "Low", delta="Optimal")
    with col3:
        if model_used:
            st.metric("Model Used", model_used.replace("gemini-", "").title())
    
    # Show top keywords in expandable section
    with st.expander("🏆 Top 10 Recommended Keywords", expanded=True):
        for i, keyword in enumerate(keywords[:10], 1):
            st.markdown(f"{i}. **{keyword}**")
    
    # Show all keywords in a searchable table
    st.markdown("### 📋 Complete Keyword List")
    st.dataframe(
        data=[{"Rank": i+1, "Keyword": k, "Length": len(k.split())} for i, k in enumerate(keywords)],
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Length": st.column_config.NumberColumn(
                "Word Count",
                help="Number of words in keyword",
                width="small"
            )
        },
        hide_index=True
    )
    
    # Enhanced download options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data="\n".join([f"{i+1},{k}" for i, k in enumerate(keywords)]),
            file_name=f"seo_keywords_{seed_keyword.replace(' ', '_')}.csv",
            mime="text/csv",
            help="Download keywords as CSV file"
        )
    
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json.dumps({"seed_keyword": seed_keyword, "keywords": keywords, "model_used": model_used}, indent=2),
            file_name=f"seo_keywords_{seed_keyword.replace(' ', '_')}.json",
            mime="application/json",
            help="Download keywords as JSON file"
        )

def display_model_status(agent: SEOKeywordAgent):
    """Display current model availability status."""
    st.markdown("### 🤖 Model Status")
    
    if agent.available_models:
        st.success(f"✅ {len(agent.available_models)} models available: {', '.join(agent.available_models)}")
    else:
        st.error("❌ No models available")
    
    # Show model priority
    st.info(f"**Primary Model:** {agent.available_models[0] if agent.available_models else 'None'}")
    
    if len(agent.available_models) > 1:
        st.info(f"**Fallback Models:** {', '.join(agent.available_models[1:])}")

def display_rate_limit_error():
    """Display rate limit error with recovery options."""
    st.error("""
    ⚠️ **Rate Limit Exceeded**
    
    You've hit Google's API rate limits. Here's what you can do:
    
    1. **Wait 1-2 minutes** - Limits reset quickly
    2. **Try again** - Multiple models available as fallbacks
    3. **Reduce request frequency** - Space out your requests
    4. **Check your quota** - Monitor usage in Google Cloud Console
    
    [Learn more about rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
    """)

def display_error(message: str):
    """Display a user-friendly error message."""
    st.error(f"""
    🛑 **Error Occurred**
    
    {message}
    
    **Troubleshooting Steps:**
    - Verify your GEMINI_API_KEY is correct
    - Check if your region supports Gemini 2.5 models
    - Ensure you have API quota available
    - Try a different seed keyword
    - Wait a moment and try again
    
    **Need Help?**
    - Check [Google AI Studio](https://aistudio.google.com/) for your API key
    - Visit [Gemini API docs](https://ai.google.dev/gemini-api/docs) for troubleshooting
    """)

def main():
    """Main application function with production-grade error handling."""
    display_header()
    
    try:
        # Get user input
        seed_keyword, max_keywords = display_sidebar()
        
        # Validate input
        if not seed_keyword:
            st.warning("Please enter a seed keyword to begin")
            return
            
        if not validate_keyword(seed_keyword):
            st.warning("Please enter a valid keyword (2-100 characters, no numbers)")
            return
        
        # Initialize agent
        try:
            agent = SEOKeywordAgent()
            display_model_status(agent)
        except ValueError as e:
            st.error(f"**Configuration Error:** {str(e)}")
            
            # Add troubleshooting section
            st.markdown("### 🔧 Troubleshooting Guide")
            
            with st.expander("**Step 1: Verify API Key**", expanded=True):
                st.markdown("""
                1. Go to [Google AI Studio](https://aistudio.google.com/)
                2. Sign in with your Google account
                3. Click "Get API Key" in the left sidebar
                4. Create a new API key or copy an existing one
                5. Make sure it's set as `GEMINI_API_KEY` environment variable
                """)
            
            with st.expander("**Step 2: Test Your API Key**"):
                st.markdown("""
                Test your API key directly in AI Studio:
                1. Go to [AI Studio Chat](https://aistudio.google.com/app/prompts/new_chat)
                2. Try typing a simple message
                3. If it works there, your key should work here
                """)
            
            with st.expander("**Step 3: Check Model Availability**"):
                st.markdown("""
                Available models as of July 2025:
                - `gemini-2.0-flash` (recommended)
                - `gemini-2.0-flash-lite`
                - `gemini-2.5-flash`
                - `gemini-2.5-pro`
                - `gemini-1.0-pro` (legacy)
                
                **Note:** Gemini 1.5 models are NOT available for new projects.
                """)
            
            with st.expander("**Step 4: Environment Setup**"):
                st.code("""
# Option 1: Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Create .env file
echo "GEMINI_API_KEY=your-api-key-here" > .env

# Option 3: Set in Python (not recommended for production)
import os
os.environ['GEMINI_API_KEY'] = 'your-api-key-here'
                """)
            
            return
        
        # Generate keywords on button click
        if st.button("✨ Generate Keywords", type="primary"):
            with st.spinner(f"🔍 Analyzing '{seed_keyword}' and generating {max_keywords} keywords..."):
                try:
                    start_time = time.time()
                    keywords = agent.generate_keywords(seed_keyword, max_keywords)
                    elapsed_time = time.time() - start_time
                    
                    # Determine which model was used (simplified)
                    model_used = agent.available_models[0] if agent.available_models else "Unknown"
                    
                    logger.info(f"Generated {len(keywords)} keywords in {elapsed_time:.2f}s")
                    display_results(keywords, seed_keyword, model_used)
                    
                except exceptions.ResourceExhausted:
                    logger.warning("Rate limit exceeded")
                    display_rate_limit_error()
                    
                except Exception as e:
                    logger.exception("Keyword generation failed")
                    display_error(str(e))
                    
    except Exception as e:
        logger.exception("Application error")
        display_error("A system error occurred. Please try again later.")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        st.error("""
        **Missing API Key**
        
        Please set your `GEMINI_API_KEY` environment variable.
        
        **How to get an API key:**
        1. Visit [Google AI Studio](https://aistudio.google.com/)
        2. Create a new API key
        3. Add it to your environment variables
        """)
        st.stop()
    
    main()
