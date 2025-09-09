# arXiv Paper Fetcher

A comprehensive Python library for fetching and analyzing research papers from arXiv using LangChain and AutoGen. This project provides multiple approaches to interact with arXiv, from simple paper fetching to advanced AI-powered analysis.

## Features

### 🔍 **Multiple Implementation Approaches**
- **LangChain Integration**: Use LangChain tools and agents for arXiv operations
- **AutoGen Multi-Agent System**: Collaborative AI agents for comprehensive paper analysis
- **Simple Fetcher**: Basic arXiv functionality without external API dependencies

### 📚 **Core Functionality**
- Search papers by keywords, authors, or categories
- Get detailed paper information including abstracts, authors, and metadata
- Download paper PDFs
- Export results to JSON
- Search recent papers within specified time ranges
- Advanced paper analysis and summarization (with AI agents)

### 🤖 **AI-Powered Features** (Requires Google Gemini API Key)
- Intelligent paper search and filtering
- Automated paper analysis and summarization
- Multi-agent collaborative research
- Natural language queries for paper discovery

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Google Gemini API Key (for AI features):**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## Quick Start

### Simple Usage (No API Key Required)

```python
from simple_arxiv_fetcher import SimpleArxivFetcher

# Initialize fetcher
fetcher = SimpleArxivFetcher()

# Search for papers
papers = fetcher.search_papers("machine learning", max_results=5)

# Print results
for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"arXiv ID: {paper['arxiv_id']}")
    print(f"Abstract: {paper['abstract'][:200]}...")
    print("-" * 50)
```

### LangChain Integration

```python
from arxiv_langchain import ArxivPaperFetcher

# Initialize with Google Gemini API key
fetcher = ArxivPaperFetcher()

# Search papers
papers = fetcher.search_papers("transformer models", max_results=3)

# Ask the AI agent
response = fetcher.ask_agent("Find recent papers on transformer models and summarize their main contributions")
print(response)
```

### AutoGen Multi-Agent System

```python
from arxiv_autogen import ArxivMultiAgentSystem

# Initialize multi-agent system
multi_agent = ArxivMultiAgentSystem()

# Collaborative analysis
result = multi_agent.collaborative_analysis(
    "Find and analyze recent papers on federated learning, focusing on privacy-preserving techniques."
)
print(result)
```

## File Structure

```
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── simple_arxiv_fetcher.py      # Basic arXiv fetcher (no API key needed)
├── arxiv_langchain.py          # LangChain implementation
├── arxiv_autogen.py            # AutoGen implementation
├── example_usage.py            # Comprehensive usage examples
└── .env                        # Environment variables (create this)
```

## Detailed Usage Examples

### 1. Basic Paper Search

```python
from simple_arxiv_fetcher import SimpleArxivFetcher

fetcher = SimpleArxivFetcher()

# Search by keyword
papers = fetcher.search_papers("deep learning", max_results=10)

# Search by author
author_papers = fetcher.search_by_author("Geoffrey Hinton", max_results=5)

# Search by category
ai_papers = fetcher.search_by_category("cs.AI", max_results=5)

# Search recent papers
recent_papers = fetcher.search_recent_papers("transformer", days=30, max_results=5)
```

### 2. Paper Analysis

```python
# Get detailed paper information
paper_details = fetcher.get_paper_by_id("2301.00001")

# Print formatted summary
fetcher.print_paper_summary(paper_details)

# Download PDF
fetcher.download_paper("2301.00001", "paper.pdf")

# Export to JSON
fetcher.export_to_json(papers, "papers.json")
```

### 3. LangChain Agent Usage

```python
from arxiv_langchain import ArxivPaperFetcher

fetcher = ArxivPaperFetcher()

# Natural language queries
response = fetcher.ask_agent("Find papers on computer vision published in 2023")
print(response)

# Use tools directly
papers = fetcher.search_papers("neural networks", max_results=5)
details = fetcher.get_paper_by_id(papers[0]['arxiv_id'])
```

### 4. AutoGen Multi-Agent Analysis

```python
from arxiv_autogen import ArxivMultiAgentSystem

multi_agent = ArxivMultiAgentSystem()

# Collaborative research
result = multi_agent.collaborative_analysis(
    "Research the latest developments in large language models, "
    "focusing on efficiency improvements and novel architectures."
)
```

## API Reference

### SimpleArxivFetcher

#### Methods

- `search_papers(query, max_results=10, sort_by="relevance")` - Search for papers
- `get_paper_by_id(arxiv_id)` - Get detailed paper information
- `download_paper(arxiv_id, save_path=None)` - Download paper PDF
- `search_by_author(author_name, max_results=10)` - Search by author
- `search_by_category(category, max_results=10)` - Search by category
- `search_recent_papers(query="", days=30, max_results=10)` - Search recent papers
- `export_to_json(papers, filename)` - Export papers to JSON
- `print_paper_summary(paper)` - Print formatted paper summary

### ArxivPaperFetcher (LangChain)

#### Methods

- `search_papers(query, max_results=5)` - Search for papers
- `get_paper_by_id(arxiv_id)` - Get detailed paper information
- `download_paper(arxiv_id, save_path=None)` - Download paper PDF
- `ask_agent(question)` - Ask the AI agent a question

### ArxivMultiAgentSystem (AutoGen)

#### Methods

- `collaborative_analysis(query)` - Perform collaborative analysis using multiple agents

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required for LangChain and AutoGen features

### arXiv Query Parameters

- **Sort Options**: `relevance`, `lastUpdatedDate`, `submittedDate`
- **Categories**: `cs.AI`, `cs.LG`, `cs.CV`, `math.NA`, etc.
- **Date Ranges**: Use `submittedDate:[YYYYMMDD TO YYYYMMDD]` format

## Examples

Run the example scripts to see the library in action:

```bash
# Run all examples
python example_usage.py

# Run simple fetcher only
python simple_arxiv_fetcher.py

# Run LangChain example
python arxiv_langchain.py

# Run AutoGen example
python arxiv_autogen.py
```

## Advanced Features

### Custom Search Queries

```python
# Complex queries
query = "cat:cs.AI AND (transformer OR attention) AND submittedDate:[20230101 TO 20231231]"
papers = fetcher.search_papers(query, max_results=20)
```

### Batch Operations

```python
# Process multiple papers
arxiv_ids = ["2301.00001", "2301.00002", "2301.00003"]
for arxiv_id in arxiv_ids:
    paper = fetcher.get_paper_by_id(arxiv_id)
    if paper:
        fetcher.print_paper_summary(paper)
```

### Integration with Other Tools

```python
# Export for further analysis
papers = fetcher.search_papers("machine learning", max_results=100)
fetcher.export_to_json(papers, "ml_papers.json")

# Use with pandas for analysis
import pandas as pd
df = pd.DataFrame(papers)
print(df.groupby('categories').size())
```

## Troubleshooting

### Common Issues

1. **Google Gemini API Key Error**: Make sure your API key is set correctly
2. **arXiv Rate Limiting**: The library includes built-in rate limiting, but you may need to add delays for large requests
3. **PDF Download Issues**: Check your internet connection and arXiv server status

### Error Handling

The library includes comprehensive error handling. Check the console output for specific error messages and suggestions.

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving error handling
- Adding more AI agent capabilities
- Creating additional examples

## License

This project is open source and available under the MIT License.

## Acknowledgments

- arXiv for providing the research paper database
- LangChain for the agent framework
- AutoGen for the multi-agent system
- The Python community for the excellent libraries used in this project
