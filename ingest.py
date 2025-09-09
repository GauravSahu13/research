"""
arXiv Paper Processing Pipeline Activator
Simple interface to activate the LangGraph pipeline based on topic name
"""

import asyncio
import argparse
import sys
from arxiv_langgraph import ArxivLangGraphAgent


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 arXiv Paper Processing Pipeline")
    print("=" * 60)
    print("Multi-agent system for fetching, extracting, and viewing research papers")
    print("Powered by LangGraph + arXiv API")
    print("=" * 60)


def print_usage():
    """Print usage instructions"""
    print("\n📖 USAGE:")
    print("python ingest.py <topic> [options]")
    print("\n🔍 EXAMPLES:")
    print("  python ingest.py 'machine learning'")
    print("  python ingest.py 'natural language processing' --max-papers 3")
    print("  python ingest.py 'deep learning' --no-images")
    print("  python ingest.py 'computer vision' --max-papers 5 --no-images")
    print("\n⚙️  OPTIONS:")
    print("  --max-papers N    Number of papers to process (default: 5)")
    print("  --no-images       Skip image display")
    print("  --help            Show this help message")


async def process_topic(topic: str, max_papers: int = 5, show_images: bool = True):
    """Process papers for a given topic"""
    try:
        print(f"\n🎯 Processing topic: '{topic}'")
        print(f"📊 Max papers: {max_papers}")
        print(f"🖼️  Show images: {'Yes' if show_images else 'No'}")
        print("-" * 60)
        
        # Initialize the agent system
        agent = ArxivLangGraphAgent(
            max_results=max_papers,
            show_images=show_images
        )
        
        # Process papers
        results = await agent.process_papers(topic, show_images=show_images)
        
        # Display results summary
        if results.get("results"):
            print("\n📊 PROCESSING SUMMARY:")
            print("-" * 30)
            results_data = results["results"]
            print(f"✅ Total papers processed: {results_data['total_papers']}")
            print(f"⏰ Processing time: {results_data['processing_time']}")
            print(f"🔍 Query: {results_data['query']}")
            
            print("\n📄 PAPERS:")
            for i, paper in enumerate(results_data['papers'], 1):
                print(f"  {i}. {paper['id']}")
                print(f"     Title: {paper['title']}")
                print(f"     Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                print(f"     Images: {paper['num_images']}, Tables: {paper['num_tables']}")
                print()
        
        print("🎉 Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing topic '{topic}': {e}")
        return False


def main():
    """Main function to handle command line arguments and run the pipeline"""
    parser = argparse.ArgumentParser(
        description="arXiv Paper Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py 'machine learning'
  python ingest.py 'natural language processing' --max-papers 3
  python ingest.py 'deep learning' --no-images
  python ingest.py 'computer vision' --max-papers 5 --no-images
        """
    )
    
    parser.add_argument(
        'topic',
        help='Research topic to search for (e.g., "machine learning", "NLP", "computer vision")'
    )
    
    parser.add_argument(
        '--max-papers',
        type=int,
        default=5,
        help='Maximum number of papers to process (default: 5)'
    )
    
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image display during processing'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate arguments
    if not args.topic.strip():
        print("❌ Error: Topic cannot be empty")
        print_usage()
        sys.exit(1)
    
    if args.max_papers < 1 or args.max_papers > 20:
        print("❌ Error: max-papers must be between 1 and 20")
        sys.exit(1)
    
    # Run the pipeline
    show_images = not args.no_images
    success = asyncio.run(process_topic(args.topic, args.max_papers, show_images))
    
    if success:
        print("\n✅ Pipeline execution completed!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
