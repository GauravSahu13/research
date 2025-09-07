"""
Simple arXiv Paper Fetcher (No API Key Required)
This is a basic implementation that doesn't require OpenAI API keys.
"""

import arxiv
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime


class SimpleArxivFetcher:
    """A simple arXiv paper fetcher without external API dependencies."""
    
    def __init__(self):
        """Initialize the simple fetcher."""
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)
            
        Returns:
            List of paper dictionaries
        """
        try:
            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }.get(sort_by, arxiv.SortCriterion.Relevance)
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )
            
            papers = []
            for result in self.client.results(search):
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
            paper = next(self.client.results(arxiv.Search(id_list=[arxiv_id])))
            
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
            paper = next(self.client.results(arxiv.Search(id_list=[arxiv_id])))
            
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
    
    def search_by_author(self, author_name: str, max_results: int = 10) -> List[Dict]:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Name of the author
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"au:{author_name}"
        return self.search_papers(query, max_results)
    
    def search_by_category(self, category: str, max_results: int = 10) -> List[Dict]:
        """
        Search for papers in a specific category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG', 'math.NA')
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"cat:{category}"
        return self.search_papers(query, max_results)
    
    def search_recent_papers(self, query: str = "", days: int = 30, max_results: int = 10) -> List[Dict]:
        """
        Search for recent papers.
        
        Args:
            query: Search query (optional)
            days: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for arXiv query
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # Build query
        if query:
            full_query = f"{query} AND submittedDate:[{start_str} TO {end_str}]"
        else:
            full_query = f"submittedDate:[{start_str} TO {end_str}]"
        
        return self.search_papers(full_query, max_results, "submittedDate")
    
    def get_paper_citations(self, arxiv_id: str) -> List[Dict]:
        """
        Get papers that cite the given paper (if available).
        Note: This is a placeholder implementation as arXiv doesn't provide citation data directly.
        
        Args:
            arxiv_id: arXiv ID of the paper
            
        Returns:
            List of citing papers (placeholder)
        """
        print("Note: arXiv doesn't provide direct citation data. This would require integration with other services like Semantic Scholar or Google Scholar.")
        return []
    
    def export_to_json(self, papers: List[Dict], filename: str) -> bool:
        """
        Export papers to JSON file.
        
        Args:
            papers: List of paper dictionaries
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            print(f"Papers exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
            return False
    
    def print_paper_summary(self, paper: Dict):
        """
        Print a formatted summary of a paper.
        
        Args:
            paper: Paper dictionary
        """
        print(f"\n{'='*80}")
        print(f"Title: {paper['title']}")
        print(f"{'='*80}")
        # Handle both string and dict author formats
        if paper['authors'] and isinstance(paper['authors'][0], dict):
            authors = [author['name'] for author in paper['authors']]
        else:
            authors = paper['authors']
        print(f"Authors: {', '.join(authors)}")
        print(f"Published: {paper['published']}")
        print(f"Updated: {paper['updated']}")
        print(f"arXiv ID: {paper['arxiv_id']}")
        print(f"Categories: {', '.join(paper['categories'])}")
        if paper.get('doi'):
            print(f"DOI: {paper['doi']}")
        if paper.get('journal_ref'):
            print(f"Journal Reference: {paper['journal_ref']}")
        print(f"\nAbstract:")
        print(f"{paper['abstract']}")
        print(f"\nPDF URL: {paper['pdf_url']}")
        print(f"{'='*80}")


def main():
    """Example usage of the SimpleArxivFetcher."""
    print("Simple arXiv Paper Fetcher")
    print("=" * 50)
    
    # Initialize the fetcher
    fetcher = SimpleArxivFetcher()
    
    # Example 1: Search for papers
    print("\n1. Searching for papers on 'machine learning'...")
    papers = fetcher.search_papers("machine learning", max_results=3)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Published: {paper['published']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")
    
    # Example 2: Search by author
    print("\n2. Searching for papers by 'Yann LeCun'...")
    author_papers = fetcher.search_by_author("Yann LeCun", max_results=2)
    
    for i, paper in enumerate(author_papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Published: {paper['published']}")
    
    # Example 3: Search by category
    print("\n3. Searching for papers in 'cs.AI' category...")
    category_papers = fetcher.search_by_category("cs.AI", max_results=2)
    
    for i, paper in enumerate(category_papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Published: {paper['published']}")
    
    # Example 4: Search recent papers
    print("\n4. Searching for recent papers (last 7 days)...")
    recent_papers = fetcher.search_recent_papers("deep learning", days=7, max_results=2)
    
    for i, paper in enumerate(recent_papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Published: {paper['published']}")
    
    # Example 5: Get detailed paper information
    if papers:
        print(f"\n5. Getting detailed information for paper: {papers[0]['arxiv_id']}")
        details = fetcher.get_paper_by_id(papers[0]['arxiv_id'])
        if details:
            fetcher.print_paper_summary(details)
    
    # Example 6: Export to JSON
    if papers:
        print("\n6. Exporting papers to JSON...")
        fetcher.export_to_json(papers, "papers_export.json")
    
    # Example 7: Download a paper (optional)
    if papers:
        print(f"\n7. Downloading PDF for paper: {papers[0]['arxiv_id']}")
        success = fetcher.download_paper(papers[0]['arxiv_id'], f"downloaded_{papers[0]['arxiv_id']}.pdf")
        if success:
            print("PDF downloaded successfully!")
        else:
            print("PDF download failed.")


if __name__ == "__main__":
    main()
