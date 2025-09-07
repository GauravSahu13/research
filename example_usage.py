"""
Example usage scripts for arXiv paper fetching with LangChain and AutoGen
"""

import os
from arxiv_langchain import ArxivPaperFetcher
from arxiv_autogen import ArxivAutoGenAgent, ArxivMultiAgentSystem


def langchain_example():
    """Example using LangChain implementation."""
    print("=" * 60)
    print("LANGCHAIN EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize the LangChain fetcher
        fetcher = ArxivPaperFetcher()
        
        # Search for papers
        print("\n1. Searching for papers on 'deep learning'...")
        papers = fetcher.search_papers("deep learning", max_results=3)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'])}")
            print(f"   Published: {paper['published']}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            print(f"   Abstract: {paper['abstract'][:150]}...")
        
        # Get detailed information about a specific paper
        if papers:
            print(f"\n2. Getting detailed information for paper: {papers[0]['arxiv_id']}")
            details = fetcher.get_paper_by_id(papers[0]['arxiv_id'])
            if details:
                print(f"   Title: {details['title']}")
                print(f"   Authors: {[author['name'] for author in details['authors']]}")
                print(f"   Categories: {details['categories']}")
                print(f"   DOI: {details['doi']}")
        
        # Ask the agent a question
        print("\n3. Asking the agent about recent transformer papers...")
        response = fetcher.ask_agent("Find recent papers on transformer models and summarize their main contributions")
        print(f"   Agent response: {response[:200]}...")
        
        # Download a paper (optional)
        if papers:
            print(f"\n4. Downloading PDF for paper: {papers[0]['arxiv_id']}")
            success = fetcher.download_paper(papers[0]['arxiv_id'], f"downloaded_{papers[0]['arxiv_id']}.pdf")
            if success:
                print("   PDF downloaded successfully!")
            else:
                print("   PDF download failed.")
        
    except Exception as e:
        print(f"Error in LangChain example: {e}")


def autogen_single_agent_example():
    """Example using AutoGen single agent."""
    print("\n" + "=" * 60)
    print("AUTOGEN SINGLE AGENT EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize the AutoGen agent
        agent = ArxivAutoGenAgent()
        
        # Search for papers
        print("\n1. Searching for papers on 'computer vision'...")
        papers = agent.search_papers("computer vision", max_results=3)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'])}")
            print(f"   Published: {paper['published']}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            print(f"   Categories: {paper['categories']}")
        
        # Chat with the agent
        if papers:
            print(f"\n2. Chatting with agent about paper analysis...")
            response = agent.chat_with_agent(
                f"Please analyze the paper '{papers[0]['title']}' and tell me about its main contributions and methodology."
            )
            print(f"   Agent response: {response[:300]}...")
        
        # Get specific paper details
        if papers:
            print(f"\n3. Getting detailed information for paper: {papers[0]['arxiv_id']}")
            details = agent.get_paper_by_id(papers[0]['arxiv_id'])
            if details:
                print(f"   Title: {details['title']}")
                print(f"   Authors: {[author['name'] for author in details['authors']]}")
                print(f"   Abstract: {details['abstract'][:200]}...")
                print(f"   Journal Reference: {details['journal_ref']}")
        
    except Exception as e:
        print(f"Error in AutoGen single agent example: {e}")


def autogen_multi_agent_example():
    """Example using AutoGen multi-agent system."""
    print("\n" + "=" * 60)
    print("AUTOGEN MULTI-AGENT SYSTEM EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize the multi-agent system
        multi_agent = ArxivMultiAgentSystem()
        
        # Collaborative analysis
        print("\n1. Performing collaborative analysis on 'machine learning interpretability'...")
        result = multi_agent.collaborative_analysis(
            "Find and analyze recent papers on machine learning interpretability. "
            "Focus on different approaches, their advantages and limitations, "
            "and provide a comprehensive summary of the current state of the field."
        )
        print(f"   Collaborative analysis result: {result[:400]}...")
        
        # Another collaborative analysis
        print("\n2. Performing collaborative analysis on 'federated learning'...")
        result2 = multi_agent.collaborative_analysis(
            "Research papers on federated learning, particularly focusing on privacy-preserving techniques "
            "and communication efficiency. Provide insights on recent advances and challenges."
        )
        print(f"   Collaborative analysis result: {result2[:400]}...")
        
    except Exception as e:
        print(f"Error in AutoGen multi-agent example: {e}")


def interactive_demo():
    """Interactive demo for users to try different queries."""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    
    print("\nChoose an option:")
    print("1. LangChain Agent")
    print("2. AutoGen Single Agent")
    print("3. AutoGen Multi-Agent System")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                query = input("Enter your search query: ").strip()
                if query:
                    fetcher = ArxivPaperFetcher()
                    papers = fetcher.search_papers(query, max_results=3)
                    for i, paper in enumerate(papers, 1):
                        print(f"\n{i}. {paper['title']}")
                        print(f"   Authors: {', '.join(paper['authors'])}")
                        print(f"   arXiv ID: {paper['arxiv_id']}")
                        print(f"   Abstract: {paper['abstract'][:150]}...")
            
            elif choice == "2":
                query = input("Enter your search query: ").strip()
                if query:
                    agent = ArxivAutoGenAgent()
                    papers = agent.search_papers(query, max_results=3)
                    for i, paper in enumerate(papers, 1):
                        print(f"\n{i}. {paper['title']}")
                        print(f"   Authors: {', '.join(paper['authors'])}")
                        print(f"   arXiv ID: {paper['arxiv_id']}")
                        print(f"   Abstract: {paper['abstract'][:150]}...")
            
            elif choice == "3":
                query = input("Enter your research question: ").strip()
                if query:
                    multi_agent = ArxivMultiAgentSystem()
                    result = multi_agent.collaborative_analysis(query)
                    print(f"\nCollaborative analysis result: {result}")
            
            elif choice == "4":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run all examples."""
    print("arXiv Paper Fetcher Examples")
    print("=" * 60)
    
    # Check if Google Gemini API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("Please set your Google Gemini API key to use the LangChain and AutoGen features.")
        print("You can still use the basic arXiv functionality without the API key.")
        print("\nTo set the API key, run:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        print("\nOr add it to your .env file.")
        return
    
    # Run examples
    langchain_example()
    autogen_single_agent_example()
    autogen_multi_agent_example()
    
    # Ask if user wants to try interactive demo
    try:
        run_interactive = input("\nWould you like to try the interactive demo? (y/n): ").strip().lower()
        if run_interactive in ['y', 'yes']:
            interactive_demo()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
