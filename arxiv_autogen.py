"""
arXiv Paper Fetcher using AutoGen
This module provides functionality to search and fetch research papers from arXiv using AutoGen agents.
"""

import os
import json
from typing import List, Dict, Optional, Any
import arxiv
import requests
from autogen import ConversableAgent, GroupChat, GroupChatManager
import tempfile
import subprocess
import google.generativeai as genai


class ArxivAutoGenAgent:
    """An AutoGen agent specialized in arXiv paper operations."""
    
    def __init__(self, name: str = "arxiv_agent", system_message: str = None, gemini_api_key: Optional[str] = None):
        """
        Initialize the ArxivAutoGenAgent.
        
        Args:
            name: Name of the agent
            system_message: Custom system message for the agent
            gemini_api_key: Google Gemini API key. If None, will try to get from environment.
        """
        self.name = name
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        if system_message is None:
            system_message = """You are an arXiv research paper assistant. You can:
1. Search for papers on arXiv using various criteria
2. Get detailed information about specific papers
3. Download paper PDFs
4. Analyze and summarize paper content
5. Find related papers and authors

When searching for papers, provide clear, specific queries. When analyzing papers, focus on:
- Main contributions and innovations
- Methodology and approach
- Key findings and results
- Limitations and future work
- Relevance to the research field

Always provide accurate arXiv IDs and verify information before making claims about papers."""
        
        # Create a custom LLM config for Gemini
        def gemini_llm_config(messages, **kwargs):
            model = genai.GenerativeModel('gemini-pro')
            # Convert messages to Gemini format
            prompt = ""
            for message in messages:
                if message.get("role") == "system":
                    prompt += f"System: {message['content']}\n\n"
                elif message.get("role") == "user":
                    prompt += f"User: {message['content']}\n\n"
                elif message.get("role") == "assistant":
                    prompt += f"Assistant: {message['content']}\n\n"
            
            response = model.generate_content(prompt)
            return response.text
        
        self.agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "functions": [gemini_llm_config],
                "temperature": 0.1
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "arxiv_workspace",
                "use_docker": False
            }
        )
        
        # Add custom functions to the agent
        self._setup_arxiv_functions()
    
    def _setup_arxiv_functions(self):
        """Setup arXiv-specific functions for the agent."""
        
        def search_arxiv_papers(query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
            """
            Search for papers on arXiv.
            
            Args:
                query: Search query
                max_results: Maximum number of results
                sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)
                
            Returns:
                JSON string containing paper information
            """
            try:
                sort_criterion = {
                    "relevance": arxiv.SortCriterion.Relevance,
                    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                    "submittedDate": arxiv.SortCriterion.SubmittedDate
                }.get(sort_by, arxiv.SortCriterion.Relevance)
                
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=sort_criterion
                )
                
                papers = []
                for result in client.results(search):
                    paper_info = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "published": result.published.strftime("%Y-%m-%d"),
                        "updated": result.updated.strftime("%Y-%m-%d"),
                        "arxiv_id": result.entry_id.split('/')[-1],
                        "pdf_url": result.pdf_url,
                        "categories": result.categories,
                        "doi": result.doi,
                        "comment": getattr(result, 'comment', None),
                        "journal_ref": getattr(result, 'journal_ref', None)
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
                    save_path = f"arxiv_workspace/{arxiv_id}.pdf"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Download the PDF
                response = requests.get(paper.pdf_url)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                return f"PDF downloaded successfully to {save_path}"
            except Exception as e:
                return f"Error downloading PDF: {str(e)}"
        
        def analyze_paper_content(arxiv_id: str) -> str:
            """
            Analyze and summarize paper content.
            
            Args:
                arxiv_id: arXiv ID of the paper
                
            Returns:
                Analysis summary
            """
            try:
                paper_details = get_paper_details(arxiv_id)
                paper_data = json.loads(paper_details)
                
                analysis = {
                    "title": paper_data["title"],
                    "arxiv_id": paper_data["arxiv_id"],
                    "key_points": [
                        "This is a placeholder for key points analysis",
                        "In a real implementation, you would use NLP to extract key points",
                        "Consider using libraries like spaCy or transformers for content analysis"
                    ],
                    "methodology": "Methodology analysis would go here",
                    "contributions": "Main contributions would be extracted here",
                    "limitations": "Limitations and future work would be identified here",
                    "relevance_score": "Relevance to query would be calculated here"
                }
                
                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"Error analyzing paper: {str(e)}"
        
        # Add functions to the agent's code execution context
        self.agent._code_execution_config["functions"] = {
            "search_arxiv_papers": search_arxiv_papers,
            "get_paper_details": get_paper_details,
            "download_paper_pdf": download_paper_pdf,
            "analyze_paper_content": analyze_paper_content
        }
    
    def search_papers(self, query: str, max_results: int = 5, sort_by: str = "relevance") -> List[Dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criterion
            
        Returns:
            List of paper dictionaries
        """
        try:
            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }.get(sort_by, arxiv.SortCriterion.Relevance)
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )
            
            papers = []
            for result in client.results(search):
                paper_info = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "updated": result.updated.strftime("%Y-%m-%d"),
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "doi": result.doi,
                    "comment": getattr(result, 'comment', None),
                    "journal_ref": getattr(result, 'journal_ref', None)
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
                save_path = f"arxiv_workspace/{arxiv_id}.pdf"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF downloaded successfully to {save_path}")
            return True
        except Exception as e:
            print(f"Error downloading PDF: {str(e)}")
            return False
    
    def chat_with_agent(self, message: str) -> str:
        """
        Send a message to the AutoGen agent.
        
        Args:
            message: Message to send to the agent
            
        Returns:
            Agent's response
        """
        try:
            response = self.agent.generate_reply(
                messages=[{"role": "user", "content": message}],
                sender=None
            )
            return response
        except Exception as e:
            return f"Error in agent communication: {str(e)}"


class ArxivMultiAgentSystem:
    """A multi-agent system for comprehensive arXiv paper analysis."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize the multi-agent system."""
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        self.researcher_agent = ArxivAutoGenAgent(
            name="researcher",
            system_message="""You are a research assistant specialized in finding and analyzing academic papers. 
            Your role is to search for relevant papers, gather information, and provide comprehensive summaries.""",
            gemini_api_key=self.gemini_api_key
        )
        
        self.analyst_agent = ArxivAutoGenAgent(
            name="analyst",
            system_message="""You are a research analyst who specializes in analyzing paper content, 
            identifying key contributions, methodologies, and research gaps. Focus on providing 
            critical analysis and insights.""",
            gemini_api_key=self.gemini_api_key
        )
        
        self.summarizer_agent = ArxivAutoGenAgent(
            name="summarizer",
            system_message="""You are a technical writer who creates clear, concise summaries of research papers. 
            Your role is to distill complex academic content into accessible summaries while maintaining 
            technical accuracy.""",
            gemini_api_key=self.gemini_api_key
        )
        
        # Setup group chat
        self.group_chat = GroupChat(
            agents=[self.researcher_agent.agent, self.analyst_agent.agent, self.summarizer_agent.agent],
            messages=[],
            max_round=10
        )
        
        # Create Gemini-based manager
        def gemini_manager_config(messages, **kwargs):
            model = genai.GenerativeModel('gemini-pro')
            prompt = "You are managing a group chat of research agents. Coordinate their responses and ensure they work together effectively.\n\n"
            for message in messages:
                if message.get("role") == "system":
                    prompt += f"System: {message['content']}\n\n"
                elif message.get("role") == "user":
                    prompt += f"User: {message['content']}\n\n"
                elif message.get("role") == "assistant":
                    prompt += f"Assistant: {message['content']}\n\n"
            
            response = model.generate_content(prompt)
            return response.text
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "functions": [gemini_manager_config],
                "temperature": 0.1
            }
        )
    
    def collaborative_analysis(self, query: str) -> str:
        """
        Perform collaborative analysis using multiple agents.
        
        Args:
            query: Research query or paper analysis request
            
        Returns:
            Collaborative analysis result
        """
        try:
            # Start the group chat
            result = self.manager.run_chat(
                messages=[{"role": "user", "content": query}],
                sender=self.researcher_agent.agent
            )
            return result
        except Exception as e:
            return f"Error in collaborative analysis: {str(e)}"


def main():
    """Example usage of the ArxivAutoGenAgent."""
    # Set your Google Gemini API key
    # os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
    
    try:
        # Example 1: Single agent usage
        print("=== Single Agent Example ===")
        agent = ArxivAutoGenAgent()
        
        # Search for papers
        print("Searching for papers on 'transformer models'...")
        papers = agent.search_papers("transformer models", max_results=3)
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'])}")
            print(f"   Published: {paper['published']}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
        
        # Chat with agent
        if papers:
            print(f"\n=== Chatting with Agent ===")
            response = agent.chat_with_agent(
                f"Analyze the paper {papers[0]['arxiv_id']} and tell me about its main contributions."
            )
            print(f"Agent response: {response}")
        
        # Example 2: Multi-agent system
        print("\n=== Multi-Agent System Example ===")
        multi_agent = ArxivMultiAgentSystem()
        
        collaborative_result = multi_agent.collaborative_analysis(
            "Find and analyze recent papers on large language models, focusing on their training methodologies and performance improvements."
        )
        print(f"Collaborative analysis result: {collaborative_result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please set your Google Gemini API key in the environment variable GOOGLE_API_KEY")


if __name__ == "__main__":
    main()
