"""
arXiv Paper Fetcher using LangChain
This module provides functionality to search and fetch research papers from arXiv using LangChain.
"""

import os
from typing import List, Dict, Optional
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import arxiv
import requests
from bs4 import BeautifulSoup
import json


class ArxivPaperFetcher:
    """A class to fetch research papers from arXiv using LangChain tools."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the ArxivPaperFetcher.
        
        Args:
            gemini_api_key: Google Gemini API key for LangChain. If None, will try to get from environment.
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.gemini_api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        self._setup_tools()
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def _setup_tools(self):
        """Setup LangChain tools for arXiv operations."""
        
        def search_arxiv_papers(query: str, max_results: int = 5) -> str:
            """
            Search for papers on arXiv.
            
            Args:
                query: Search query
                max_results: Maximum number of results to return
                
            Returns:
                JSON string containing paper information
            """
            try:
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                papers = []
                for result in client.results(search):
                    paper_info = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "published": result.published.strftime("%Y-%m-%d"),
                        "arxiv_id": result.entry_id.split('/')[-1],
                        "pdf_url": result.pdf_url,
                        "categories": result.categories,
                        "doi": result.doi
                    }
                    papers.append(paper_info)
                
                return json.dumps(papers, indent=2)
            except Exception as e:
                return f"Error searching arXiv: {str(e)}"
        
        def get_paper_details(arxiv_id: str) -> str:
            """
            Get detailed information about a specific paper.
            
            Args:
                arxiv_id: arXiv ID of the paper
                
            Returns:
                JSON string containing detailed paper information
            """
            try:
                client = arxiv.Client()
                paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
                
                paper_info = {
                    "title": paper.title,
                    "authors": [{"name": author.name, "affiliation": getattr(author, 'affiliation', None)} for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "arxiv_id": paper.entry_id.split('/')[-1],
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories,
                    "doi": paper.doi,
                    "comment": getattr(paper, 'comment', None),
                    "journal_ref": getattr(paper, 'journal_ref', None)
                }
                
                return json.dumps(paper_info, indent=2)
            except Exception as e:
                return f"Error fetching paper details: {str(e)}"
        
        def download_paper_pdf(arxiv_id: str, save_path: str = None) -> str:
            """
            Download PDF of a paper.
            
            Args:
                arxiv_id: arXiv ID of the paper
                save_path: Path to save the PDF (optional)
                
            Returns:
                Status message
            """
            try:
                client = arxiv.Client()
                paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
                
                if not save_path:
                    save_path = f"{arxiv_id}.pdf"
                
                # Download the PDF
                response = requests.get(paper.pdf_url)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                return f"PDF downloaded successfully to {save_path}"
            except Exception as e:
                return f"Error downloading PDF: {str(e)}"
        
        # Create LangChain tools
        self.tools = [
            Tool(
                name="search_arxiv",
                description="Search for research papers on arXiv. Input should be a search query string.",
                func=lambda query: search_arxiv_papers(query)
            ),
            Tool(
                name="get_paper_details",
                description="Get detailed information about a specific arXiv paper. Input should be an arXiv ID.",
                func=get_paper_details
            ),
            Tool(
                name="download_paper_pdf",
                description="Download PDF of a specific arXiv paper. Input should be an arXiv ID.",
                func=download_paper_pdf
            )
        ]
    
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                paper_info = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "doi": result.doi
                }
                papers.append(paper_info)
            
            return papers
        except Exception as e:
            print(f"Error searching arXiv: {str(e)}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific paper.
        
        Args:
            arxiv_id: arXiv ID of the paper
            
        Returns:
            Paper information dictionary or None if not found
        """
        try:
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
            
            paper_info = {
                "title": paper.title,
                "authors": [{"name": author.name, "affiliation": getattr(author, 'affiliation', None)} for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d"),
                "arxiv_id": paper.entry_id.split('/')[-1],
                "pdf_url": paper.pdf_url,
                "categories": paper.categories,
                "doi": paper.doi,
                "comment": getattr(paper, 'comment', None),
                "journal_ref": getattr(paper, 'journal_ref', None)
            }
            
            return paper_info
        except Exception as e:
            print(f"Error fetching paper details: {str(e)}")
            return None
    
    def download_paper(self, arxiv_id: str, save_path: str = None) -> bool:
        """
        Download PDF of a paper.
        
        Args:
            arxiv_id: arXiv ID of the paper
            save_path: Path to save the PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
            
            if not save_path:
                save_path = f"{arxiv_id}.pdf"
            
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF downloaded successfully to {save_path}")
            return True
        except Exception as e:
            print(f"Error downloading PDF: {str(e)}")
            return False
    
    def ask_agent(self, question: str) -> str:
        """
        Ask a question to the LangChain agent about arXiv papers.
        
        Args:
            question: Question about arXiv papers
            
        Returns:
            Agent's response
        """
        return self.agent.run(question)


def main():
    """Example usage of the ArxivPaperFetcher."""
    # Set your Google Gemini API key
    # os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
    
    try:
        # Initialize the fetcher
        fetcher = ArxivPaperFetcher()
        
        # Example 1: Search for papers
        print("=== Searching for papers on 'machine learning' ===")
        papers = fetcher.search_papers("machine learning", max_results=3)
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'])}")
            print(f"   Published: {paper['published']}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            print(f"   Abstract: {paper['abstract'][:200]}...")
        
        # Example 2: Get specific paper details
        if papers:
            print(f"\n=== Getting details for paper: {papers[0]['arxiv_id']} ===")
            details = fetcher.get_paper_by_id(papers[0]['arxiv_id'])
            if details:
                print(f"Title: {details['title']}")
                print(f"Authors: {[author['name'] for author in details['authors']]}")
                print(f"Abstract: {details['abstract']}")
        
        # Example 3: Ask the agent
        print("\n=== Asking the agent ===")
        response = fetcher.ask_agent("Find recent papers on transformer models and summarize their main contributions")
        print(f"Agent response: {response}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your Google Gemini API key in the environment variable GOOGLE_API_KEY")


if __name__ == "__main__":
    main()
