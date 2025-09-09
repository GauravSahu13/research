"""
Complete arXiv Paper Processing System using LangGraph
Multi-agent workflow for fetching, extracting, and viewing research papers
"""

import os
import json
import asyncio
import glob
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime

import arxiv
import fitz  # PyMuPDF
import camelot
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class PaperState(TypedDict):
    """State for the paper processing workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    papers: List[Dict]
    downloaded_papers: List[Dict]
    extracted_content: Dict
    results: Dict
    error: Optional[str]
    show_images: bool


class ArxivFetcherAgent:
    """Agent responsible for fetching papers from arXiv"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
    
    def fetch_papers(self, state: PaperState) -> PaperState:
        """Fetch papers from arXiv based on query"""
        try:
            query = state["query"]
            print(f"🔍 Fetching papers for query: {query}")
            
            # Search arXiv
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                paper_info = {
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "entry_id": result.entry_id
                }
                papers.append(paper_info)
            
            state["papers"] = papers
            state["messages"].append(AIMessage(
                content=f"Found {len(papers)} papers for query: {query}"
            ))
            
            print(f"✅ Found {len(papers)} papers")
            return state
            
        except Exception as e:
            error_msg = f"Error fetching papers: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            print(f"❌ {error_msg}")
            return state


class PaperDownloaderAgent:
    """Agent responsible for downloading PDF files"""
    
    def __init__(self, download_dir: str = "downloads/ai_ml_nlp"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
    
    def download_papers(self, state: PaperState) -> PaperState:
        """Download PDF files for papers"""
        try:
            papers = state["papers"]
            downloaded = []
            
            print(f"📥 Downloading {len(papers)} papers...")
            
            for paper in papers:
                try:
                    # Download PDF
                    paper_id = paper["id"]
                    pdf_path = os.path.join(self.download_dir, f"{paper_id}.pdf")
                    
                    if not os.path.exists(pdf_path):
                        # Use arxiv client to download
                        client = arxiv.Client()
                        search = arxiv.Search(id_list=[paper_id])
                        result = next(client.results(search))
                        
                        # Download the PDF
                        result.download_pdf(dirpath=self.download_dir, filename=f"{paper_id}.pdf")
                        print(f"  ✅ Downloaded: {paper_id}")
                    else:
                        print(f"  ⏭️  Already exists: {paper_id}")
                    
                    paper["pdf_path"] = pdf_path
                    downloaded.append(paper)
                    
                except Exception as e:
                    print(f"  ❌ Failed to download {paper_id}: {e}")
                    continue
            
            state["downloaded_papers"] = downloaded
            state["messages"].append(AIMessage(
                content=f"Downloaded {len(downloaded)} papers successfully"
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error downloading papers: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            return state


class ContentExtractorAgent:
    """Agent responsible for extracting text, images, and tables from PDFs"""
    
    def __init__(self, output_dir: str = "downloads/ai_ml_nlp_extracted"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_content(self, state: PaperState) -> PaperState:
        """Extract text, images, and tables from downloaded papers"""
        try:
            papers = state["downloaded_papers"]
            extracted_content = {}
            
            print(f"🔧 Extracting content from {len(papers)} papers...")
            
            for paper in papers:
                paper_id = paper["id"]
                pdf_path = paper["pdf_path"]
                
                print(f"  Processing: {paper_id}")
                
                # Create output directories
                paper_dir = os.path.join(self.output_dir, paper_id)
                text_dir = os.path.join(paper_dir, "text")
                images_dir = os.path.join(paper_dir, "images")
                tables_dir = os.path.join(paper_dir, "tables")
                
                os.makedirs(text_dir, exist_ok=True)
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(tables_dir, exist_ok=True)
                
                # Extract text
                text_content = self._extract_text(pdf_path)
                text_path = os.path.join(text_dir, f"{paper_id}.md")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                
                # Extract images
                num_images = self._extract_images(pdf_path, images_dir)
                
                # Extract tables
                num_tables = self._extract_tables(pdf_path, tables_dir)
                
                extracted_content[paper_id] = {
                    "text_path": text_path,
                    "images_dir": images_dir,
                    "tables_dir": tables_dir,
                    "num_images": num_images,
                    "num_tables": num_tables,
                    "paper_info": paper
                }
                
                print(f"    ✅ Text: {text_path}")
                print(f"    ✅ Images: {num_images} saved")
                print(f"    ✅ Tables: {num_tables} saved")
            
            state["extracted_content"] = extracted_content
            state["messages"].append(AIMessage(
                content=f"Extracted content from {len(extracted_content)} papers"
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error extracting content: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            return state
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"## Page {page_num + 1}\n\n{text}")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def _extract_images(self, pdf_path: str, output_dir: str) -> int:
        """Extract images from PDF using PyMuPDF with captions"""
        try:
            doc = fitz.open(pdf_path)
            saved_count = 0
            captions_data = []
            
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                page_text = page.get_text()
                
                # Extract images
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    
                    img_filename = os.path.join(
                        output_dir, 
                        f"page{page_index+1}_img{img_index+1}.{ext}"
                    )
                    
                    with open(img_filename, "wb") as f:
                        f.write(image_bytes)
                    
                    # Extract caption for this image
                    caption = self._extract_image_caption(page_text, img_index + 1, page_index + 1)
                    
                    # Store caption data
                    caption_data = {
                        "image_file": os.path.basename(img_filename),
                        "page": page_index + 1,
                        "image_index": img_index + 1,
                        "caption": caption
                    }
                    captions_data.append(caption_data)
                    
                    saved_count += 1
            
            # Save captions to JSON file
            captions_file = os.path.join(output_dir, "image_captions.json")
            with open(captions_file, "w", encoding="utf-8") as f:
                json.dump(captions_data, f, indent=2)
            
            doc.close()
            return saved_count
            
        except Exception as e:
            print(f"Error extracting images: {e}")
            return 0
    
    def _extract_image_caption(self, page_text: str, img_index: int, page_num: int) -> str:
        """Extract caption for an image from page text"""
        try:
            # Common caption patterns
            caption_patterns = [
                r'Figure\s+\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Fig\.\s+\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Figure\s+\d+\.\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Fig\.\s+\d+\.\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Image\s+\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Table\s+\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
                r'Algorithm\s+\d+[\.:]?\s*(.+?)(?=\n\n|\n[A-Z]|\n\d+\.|\Z)',
            ]
            
            # Look for captions in the page text
            for pattern in caption_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Try to find the most relevant caption
                    for match in matches:
                        caption_text = match.strip()
                        if len(caption_text) > 10 and len(caption_text) < 500:  # Reasonable caption length
                            return caption_text
            
            # If no specific caption found, look for text near image references
            image_ref_patterns = [
                r'see\s+figure\s+\d+[\.:]?\s*(.+?)(?=\n|\.)',
                r'as\s+shown\s+in\s+figure\s+\d+[\.:]?\s*(.+?)(?=\n|\.)',
                r'figure\s+\d+\s+shows\s*(.+?)(?=\n|\.)',
                r'fig\.\s+\d+\s+illustrates\s*(.+?)(?=\n|\.)',
            ]
            
            for pattern in image_ref_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    for match in matches:
                        caption_text = match.strip()
                        if len(caption_text) > 10 and len(caption_text) < 500:
                            return caption_text
            
            # Fallback: return a generic description
            return f"Image {img_index} from page {page_num} - No caption found"
            
        except Exception as e:
            return f"Image {img_index} from page {page_num} - Caption extraction failed: {str(e)}"
    
    def _extract_tables(self, pdf_path: str, output_dir: str) -> int:
        """Extract tables from PDF using Camelot"""
        try:
            # Try lattice first, then stream
            for flavor in ['lattice', 'stream']:
                try:
                    tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
                    if tables and tables.n > 0:
                        count = 0
                        for i, table in enumerate(tables):
                            df = table.df
                            # Clean DataFrame
                            df = df.map(
                                lambda x: re.sub(r"\s+", " ", str(x).strip()) 
                                if pd.notna(x) else x
                            )
                            df = df.replace("", pd.NA)
                            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
                            
                            if not df.empty:
                                csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
                                df.to_csv(csv_path, index=False)
                                count += 1
                        
                        return count
                        
                except Exception:
                    continue
            
            return 0
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return 0


class ImageViewerAgent:
    """Agent responsible for displaying extracted images"""
    
    def __init__(self, output_dir: str = "downloads/ai_ml_nlp_extracted"):
        self.output_dir = output_dir
    
    def display_images(self, state: PaperState) -> PaperState:
        """Display images from extracted papers"""
        try:
            if not state.get("show_images", False):
                print("⏭️  Skipping image display (show_images=False)")
                return state
            
            extracted_content = state["extracted_content"]
            
            print(f"🖼️  Displaying images from {len(extracted_content)} papers...")
            
            for paper_id, content in extracted_content.items():
                if content["num_images"] > 0:
                    print(f"\n📄 Displaying images from: {paper_id}")
                    print(f"   Title: {content['paper_info']['title']}")
                    print(f"   Images: {content['num_images']}")
                    
                    # Display images for this paper
                    self._display_images_from_paper(paper_id, content["images_dir"])
                    
                    # Ask user if they want to continue
                    continue_display = input("\nContinue to next paper? (y/n): ").lower().strip()
                    if continue_display != 'y':
                        break
                else:
                    print(f"⏭️  No images found in {paper_id}")
            
            state["messages"].append(AIMessage(
                content="Image display completed"
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error displaying images: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            return state
    
    def _display_images_from_paper(self, paper_id: str, images_dir: str, max_images: int = 9):
        """Display all images from a specific paper with captions"""
        try:
            # Get all image files
            image_files = glob.glob(os.path.join(images_dir, "*.png")) + \
                         glob.glob(os.path.join(images_dir, "*.jpg")) + \
                         glob.glob(os.path.join(images_dir, "*.jpeg"))
            
            if not image_files:
                print(f"No images found in {images_dir}")
                return
            
            # Load captions if available
            captions_data = {}
            captions_file = os.path.join(images_dir, "image_captions.json")
            if os.path.exists(captions_file):
                try:
                    with open(captions_file, "r", encoding="utf-8") as f:
                        captions_list = json.load(f)
                        for caption_info in captions_list:
                            captions_data[caption_info["image_file"]] = caption_info["caption"]
                except Exception as e:
                    print(f"Warning: Could not load captions: {e}")
            
            # Limit number of images to display
            image_files = sorted(image_files)[:max_images]
            
            # Calculate grid size
            n_images = len(image_files)
            cols = min(3, n_images)
            rows = (n_images + cols - 1) // cols
            
            # Create figure with more space for captions
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
            fig.suptitle(f"Images from Paper: {paper_id}", fontsize=16, fontweight='bold')
            
            # Handle single image case
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            # Display each image
            for i, img_path in enumerate(image_files):
                try:
                    # Load and display image
                    img = mpimg.imread(img_path)
                    axes[i].imshow(img)
                    
                    # Get caption for this image
                    img_filename = os.path.basename(img_path)
                    caption = captions_data.get(img_filename, f"Page {self._extract_page_number(img_path)}")
                    
                    # Truncate long captions for display
                    if len(caption) > 100:
                        caption = caption[:97] + "..."
                    
                    axes[i].set_title(caption, fontsize=9, pad=10, wrap=True)
                    axes[i].axis('off')
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print captions summary
            print(f"✅ Displayed {n_images} images from {paper_id}")
            if captions_data:
                print(f"📝 Captions loaded for {len(captions_data)} images")
                print("\n📋 Image Captions:")
                for img_file in sorted(image_files):
                    img_filename = os.path.basename(img_file)
                    if img_filename in captions_data:
                        caption = captions_data[img_filename]
                        print(f"  • {img_filename}: {caption[:100]}{'...' if len(caption) > 100 else ''}")
            
            # Ask if user wants to download displayed images
            download_choice = input(f"\n📥 Download the {n_images} displayed images? (y/n): ").lower().strip()
            if download_choice == 'y':
                self._download_displayed_images(paper_id, image_files, captions_data)
            
        except Exception as e:
            print(f"❌ Error displaying images: {e}")
    
    def _download_displayed_images(self, paper_id: str, image_files: list, captions_data: dict):
        """Download the displayed images to a downloads folder"""
        try:
            # Create downloads folder
            downloads_folder = "downloaded_images"
            os.makedirs(downloads_folder, exist_ok=True)
            
            # Create paper-specific folder
            paper_folder = os.path.join(downloads_folder, paper_id)
            os.makedirs(paper_folder, exist_ok=True)
            
            downloaded_count = 0
            
            print(f"\n📥 Downloading {len(image_files)} images to {paper_folder}...")
            
            for img_file in image_files:
                img_filename = os.path.basename(img_file)
                dest_path = os.path.join(paper_folder, img_filename)
                
                # Copy image file
                import shutil
                shutil.copy2(img_file, dest_path)
                downloaded_count += 1
                
                print(f"  ✅ Downloaded: {img_filename}")
            
            # Save captions file if available
            if captions_data:
                captions_file = os.path.join(paper_folder, "image_captions.json")
                captions_list = []
                for img_file in image_files:
                    img_filename = os.path.basename(img_file)
                    if img_filename in captions_data:
                        captions_list.append({
                            "image_file": img_filename,
                            "caption": captions_data[img_filename]
                        })
                
                with open(captions_file, "w", encoding="utf-8") as f:
                    json.dump(captions_list, f, indent=2)
                print(f"  ✅ Downloaded: image_captions.json")
            
            print(f"\n🎉 Download complete!")
            print(f"📁 Location: {paper_folder}")
            print(f"🖼️  Images downloaded: {downloaded_count}")
            
        except Exception as e:
            print(f"❌ Error downloading images: {e}")
    
    def _extract_page_number(self, image_path: str) -> str:
        """Extract page number from image filename"""
        filename = os.path.basename(image_path)
        # Extract page number from filename like "page12_img1.png"
        if "page" in filename:
            try:
                page_part = filename.split("page")[1].split("_")[0]
                return f"Page {page_part}"
            except:
                pass
        return "Unknown Page"


class SummaryAgent:
    """Agent responsible for generating summaries and organizing results"""
    
    def __init__(self, output_dir: str = "downloads/ai_ml_nlp_extracted"):
        self.output_dir = output_dir
    
    def generate_summary(self, state: PaperState) -> PaperState:
        """Generate summary of the processing results"""
        try:
            extracted_content = state["extracted_content"]
            results = {
                "total_papers": len(extracted_content),
                "processing_time": datetime.now().isoformat(),
                "query": state["query"],
                "papers": []
            }
            
            for paper_id, content in extracted_content.items():
                paper_summary = {
                    "id": paper_id,
                    "title": content["paper_info"]["title"],
                    "authors": content["paper_info"]["authors"],
                    "num_images": content["num_images"],
                    "num_tables": content["num_tables"],
                    "text_path": content["text_path"],
                    "images_dir": content["images_dir"],
                    "tables_dir": content["tables_dir"]
                }
                results["papers"].append(paper_summary)
            
            state["results"] = results
            
            # Save results to file
            results_path = os.path.join(self.output_dir, "processing_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            state["messages"].append(AIMessage(
                content=f"Processing complete! Results saved to {results_path}"
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            return state


class ArxivLangGraphAgent:
    """Complete LangGraph agent system for arXiv paper processing"""
    
    def __init__(self, 
                 max_results: int = 5,
                 download_dir: str = "downloads/ai_ml_nlp",
                 output_dir: str = "downloads/ai_ml_nlp_extracted",
                 show_images: bool = True):
        
        # Initialize agents
        self.fetcher = ArxivFetcherAgent(max_results)
        self.downloader = PaperDownloaderAgent(download_dir)
        self.extractor = ContentExtractorAgent(output_dir)
        self.image_viewer = ImageViewerAgent(output_dir)
        self.summarizer = SummaryAgent(output_dir)
        self.output_dir = output_dir
        self.show_images = show_images
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(PaperState)
        
        # Add nodes
        workflow.add_node("fetch", self.fetcher.fetch_papers)
        workflow.add_node("download", self.downloader.download_papers)
        workflow.add_node("extract", self.extractor.extract_content)
        workflow.add_node("view_images", self.image_viewer.display_images)
        workflow.add_node("summarize", self.summarizer.generate_summary)
        
        # Define the flow
        workflow.set_entry_point("fetch")
        workflow.add_edge("fetch", "download")
        workflow.add_edge("download", "extract")
        workflow.add_edge("extract", "view_images")
        workflow.add_edge("view_images", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    async def process_papers(self, query: str, show_images: Optional[bool] = None) -> Dict:
        """Process papers using the LangGraph workflow"""
        show_images = show_images if show_images is not None else self.show_images
        
        initial_state = PaperState(
            messages=[HumanMessage(content=f"Process papers for query: {query}")],
            query=query,
            papers=[],
            downloaded_papers=[],
            extracted_content={},
            results={},
            error=None,
            show_images=show_images
        )
        
        print(f"🚀 Starting paper processing workflow for: {query}")
        print(f"🖼️  Image display: {'Enabled' if show_images else 'Disabled'}")
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        if final_state.get("error"):
            print(f"❌ Workflow failed: {final_state['error']}")
        else:
            print("✅ Workflow completed successfully!")
        
        return final_state


# Export the main class for use in ingest.py
__all__ = ['ArxivLangGraphAgent']
