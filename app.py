# 🔍 SEO Keyword Research Agent

A powerful AI-powered tool that generates high-quality SEO keyword suggestions using Google's latest Gemini 2.5 models. Perfect for content strategists, digital marketers, and SEO professionals.

## ✨ Features

- **Advanced AI Models**: Uses Google's latest Gemini 2.0/2.5 models for intelligent keyword generation
- **Smart Fallback System**: Automatically tries multiple models if one fails
- **Comprehensive Keyword Analysis**: Generates buyer-intent, long-tail, and question-based keywords
- **Export Options**: Download results as CSV or JSON
- **Rate Limit Handling**: Intelligent retry mechanism with exponential backoff
- **Production-Ready**: Comprehensive error handling and logging
- **Modern UI**: Clean, responsive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google AI Studio API key
- Internet connection

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd seo-keyword-research-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   Get your API key from [Google AI Studio](https://aistudio.google.com/) and set it as an environment variable:
   
   ```bash
   # Option 1: Environment variable
   export GEMINI_API_KEY="your-api-key-here"
   
   # Option 2: Create .env file
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google AI Studio API key | Yes |

### Supported Models

The application automatically detects and uses available models in this priority order:

1. **gemini-2.0-flash** - Fast and cost-effective (primary)
2. **gemini-2.0-flash-lite** - Lightweight version
3. **gemini-2.5-flash** - Advanced capabilities
4. **gemini-2.5-pro** - Most advanced reasoning
5. **gemini-1.0-pro** - Legacy fallback

> **Note**: Gemini 1.5 models are no longer available for new projects as of April 2025.

## 📖 Usage

### Basic Usage

1. Enter your seed keyword (e.g., "digital marketing")
2. Adjust the number of keywords (10-100)
3. Click "Generate Keywords"
4. Review results and export if needed

### Advanced Tips

- **Be specific**: Use specific terms like "content marketing strategy" instead of "marketing"
- **Try variations**: Test different keyword variations to find the best opportunities
- **Use long-tail phrases**: For niche topics, use 3-5 word phrases
- **Consider user intent**: Think about what your audience is searching for

### Generated Keyword Types

The tool generates:
- **Buyer-intent keywords**: Terms with commercial intent (e.g., "buy", "best", "review")
- **Long-tail variations**: 3-5 word phrases with lower competition
- **Question-based keywords**: "How to", "What is", "Where to find" variations
- **Location modifiers**: Location-specific terms when relevant

## 🛠️ API Reference

### Core Methods

```python
from seo_agent import SEOKeywordAgent

# Initialize agent
agent = SEOKeywordAgent()

# Generate keywords
keywords = agent.generate_keywords(
    seed_keyword="digital marketing",
    max_keywords=50
)
```

### Configuration Options

```python
# Custom configuration
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 2000
}

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds
```

## 📁 Project Structure

```
seo-keyword-research-agent/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── README.md             # This file
├── seo_agent.log         # Application logs
└── tests/                # Test files (if any)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "No models available" error
- **Check your API key**: Verify it's valid in [Google AI Studio](https://aistudio.google.com/)
- **Check model access**: Ensure your account has access to Gemini 2.0/2.5 models
- **Try different regions**: Some models may not be available in all regions

#### 2. Rate limit errors
- **Wait 1-2 minutes**: Rate limits reset quickly
- **Space out requests**: Don't make rapid successive requests
- **Check quota**: Monitor usage in Google Cloud Console

#### 3. Empty responses
- **Try different keywords**: Some keywords may not generate results
- **Check input validation**: Ensure keywords are 2-100 characters
- **Verify model status**: Check if models are available in your region

### Getting Help

1. **Check the logs**: Review `seo_agent.log` for detailed error information
2. **Test API key**: Use Google AI Studio to verify your API key works
3. **Check documentation**: Visit [Gemini API docs](https://ai.google.dev/gemini-api/docs)
4. **Model availability**: Verify model access in your Google Cloud Console

## 📊 Performance

- **Generation time**: ~2-5 seconds per request
- **Keywords per request**: 10-100 (configurable)
- **Rate limits**: Varies by API tier (Free/Paid)
- **Cache**: Results are cached for repeated requests

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Streamlit Cloud
1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GEMINI_API_KEY` in secrets

#### Heroku
```bash
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your-api-key
git push heroku main
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- [ ] Integration with Google Search Console
- [ ] Search volume estimation
- [ ] Competition analysis
- [ ] Export to Google Sheets
- [ ] Bulk keyword processing
- [ ] Custom model fine-tuning
- [ ] Real-time keyword tracking

## 🙏 Acknowledgments

- Google AI Team for the Gemini API
- Streamlit team for the amazing framework
- Open source community for inspiration

## 📞 Support

For support, please:
1. Check the troubleshooting section above
2. Review the [Gemini API documentation](https://ai.google.com/dev/api/rest)
3. Open an issue on GitHub
4. Contact the maintainers

---

Made with ❤️ by developers, for SEO professionals.
