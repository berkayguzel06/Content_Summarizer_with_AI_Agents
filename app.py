import os
import sys
import argparse
import json
import urllib.request
import tempfile
from pathlib import Path
import yaml
import textwrap
from typing import Union, Dict, Any, Optional, List
import colorama
from colorama import Fore, Style, Back
import time
from dotenv import load_dotenv

# SmoLAgents imports
from smolagents import CodeAgent, LiteLLMModel, tool

# Define ASCII art for the application banner
BANNER = f"""
{Fore.CYAN}
   ██████╗ ██████╗ ███╗   ██╗████████╗███████╗███╗   ██╗████████╗
  ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝
  ██║     ██║   ██║██╔██╗ ██║   ██║   █████╗  ██╔██╗ ██║   ██║   
  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝  ██║╚██╗██║   ██║   
  ╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██║ ╚████║   ██║   
   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   
███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗███████╗██████╗ 
██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
█████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   █████╗  ██████╔╝
██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██╔══╝  ██╔══██╗
███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ███████╗██║  ██║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
{Style.RESET_ALL}
{Fore.GREEN}A powerful AI agent for extracting and summarizing content from PDFs and web pages 🚀{Style.RESET_ALL}
{Fore.YELLOW}Version 1.0.0{Style.RESET_ALL}
"""

# Initialize colorama
colorama.init()

# Define custom tools for PDF processing
@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        # Using PyPDF2 for PDF extraction
        from PyPDF2 import PdfReader
        
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text from each page
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
            
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

@tool
def extract_text_from_webpage(url: str) -> str:
    """
    Extract text content from a webpage.
    
    Args:
        url: URL of the webpage
        
    Returns:
        Extracted text content from the webpage
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from newspaper import Article
        
        # First try using newspaper3k for article extraction
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Check if we got meaningful content
            if article.text and len(article.text) > 100:
                result = f"Title: {article.title}\n\n{article.text}"
                return result
        except Exception:
            # If newspaper3k fails, fallback to BeautifulSoup
            pass
            
        # Fallback to BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
        
        # Get title
        title = soup.title.string if soup.title else "No title found"
        
        # Try to find the main content
        main_content = None
        
        # Look for article tags first
        if soup.find('article'):
            main_content = soup.find('article')
        # Then look for main tag
        elif soup.find('main'):
            main_content = soup.find('main')
        # Then look for common content div IDs
        elif soup.find(id=['content', 'main-content', 'article-content', 'post-content']):
            main_content = soup.find(id=['content', 'main-content', 'article-content', 'post-content'])
        
        # If we found main content, extract text from it
        if main_content:
            text = main_content.get_text(separator='\n')
        else:
            # Otherwise get text from entire body
            text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return f"Title: {title}\n\n{text}"
    except Exception as e:
        return f"Error extracting text from webpage: {str(e)}"

@tool
def extract_text_from_multiple_webpages(urls: List[str]) -> Dict[str, str]:
    """
    Extract text content from multiple webpages.
    
    Args:
        urls: List of webpage URLs
        
    Returns:
        Dictionary mapping URLs to their extracted text content
    """
    results = {}
    
    for url in urls:
        try:
            results[url] = extract_text_from_webpage(url)
        except Exception as e:
            results[url] = f"Error extracting text from {url}: {str(e)}"
    
    return results

@tool
def save_summary_to_file(summary: str, output_path: str) -> str:
    """
    Save the summary to a file.
    
    Args:
        summary: The summary text to save
        output_path: Path to save the summary
        
    Returns:
        Confirmation message
    """
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        return f"Summary saved to {output_path}"
    except Exception as e:
        return f"Error saving summary: {str(e)}"

@tool
def download_pdf_from_url(url: str, save_path: Optional[str] = None) -> str:
    """
    Download a PDF from a URL and save it to a temporary file.
    
    Args:
        url: URL of the PDF file
        save_path: Optional path to save the downloaded PDF
        
    Returns:
        Path to the downloaded PDF file
    """
    try:
        if save_path is None:
            # Create a temporary file with .pdf extension
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                save_path = temp_file.name
                
        # Download the PDF file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
            
        return save_path
    except Exception as e:
        return f"Error downloading PDF: {str(e)}"

class ContentExtractor:
    """Main class for the Content Extractor application"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Content Extractor application.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_llm_model()
        self._setup_agents()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        return yaml.safe_load(file)
                    elif config_path.endswith('.json'):
                        return json.load(file)
            
            # If config file doesn't exist, create a default one
            default_config = {
                "llm": {
                    "model_id": "openai/gpt-3.5-turbo",
                    "api_key": ""
                },
                "output": {
                    "default_folder": "summaries",
                    "format": "markdown"
                },
                "summary": {
                    "max_length": 1000,
                    "style": "concise"
                },
                "web": {
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "timeout": 30
                }
            }
            
            # Create the config file
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(default_config, file)
                elif config_path.endswith('.json'):
                    json.dump(default_config, file, indent=2)
                else:
                    with open(config_path, 'w') as file:
                        yaml.dump(default_config, file)
            
            print(f"{Fore.YELLOW}Created default configuration file at {config_path}{Style.RESET_ALL}")
            
            # Also try to load from environment
            load_dotenv()
            default_config["llm"]["api_key"] = os.getenv("API_KEY", "")
            
            return default_config
            
        except Exception as e:
            print(f"{Fore.RED}Error loading configuration: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    def _setup_llm_model(self):
        """Set up the LLM model based on configuration"""
        model_id = self.config["llm"]["model_id"]
        api_key = self.config["llm"]["api_key"]
        
        # Use environment variable if not in config
        if not api_key:
            api_key = os.getenv("API_KEY")
            
        if not api_key:
            print(f"{Fore.RED}API key not found. Please set it in config.yaml or as API_KEY environment variable.{Style.RESET_ALL}")
            sys.exit(1)
            
        # Set the appropriate environment variable based on the model
        if "openai" in model_id:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "gemini" in model_id:
            os.environ["GEMINI_API_KEY"] = api_key
        elif "anthropic" in model_id:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["API_KEY"] = api_key
            
        # Initialize the model
        self.model = LiteLLMModel(model_id=model_id)
        
    def _setup_agents(self):
        """Set up the AI agents for different tasks"""
        # Content processing agent
        self.content_agent = CodeAgent(
            tools=[extract_text_from_pdf, extract_text_from_webpage, 
                  extract_text_from_multiple_webpages, download_pdf_from_url],
            additional_authorized_imports=["PyPDF2", "tempfile", "urllib.request", 
                                         "requests", "bs4", "newspaper", "subprocess"],
            name="Content Processing Agent",
            description="This agent can extract text from PDF documents and web pages.",
            model=self.model
        )
        
        # Summary agent
        self.summary_agent = CodeAgent(
            tools=[save_summary_to_file],
            name="Summary Agent",
            description="This agent can summarize text and save the summary to a file.",
            model=self.model
        )
        
    def process_content(self, source: str, output_path: Optional[str] = None, 
                      custom_prompt: Optional[str] = None, source_type: Optional[str] = None) -> str:
        """
        Process content from a PDF file or webpage and generate a summary.
        
        Args:
            source: Path or URL to the PDF file or webpage
            output_path: Path to save the summary
            custom_prompt: Custom prompt for the summary
            source_type: Type of source ('pdf', 'web', or None for auto-detection)
            
        Returns:
            The generated summary
        """
        print(f"{Fore.CYAN}Processing content from: {source}{Style.RESET_ALL}")
        
        # Determine the source type if not specified
        if source_type is None:
            if source.lower().endswith('.pdf') or '/pdf/' in source.lower():
                source_type = 'pdf'
            elif source.startswith('http://') or source.startswith('https://'):
                source_type = 'web'
            else:
                # Assume it's a local file, check the extension
                if os.path.exists(source) and source.lower().endswith('.pdf'):
                    source_type = 'pdf'
                else:
                    print(f"{Fore.RED}Unable to determine source type. Please specify with --type.{Style.RESET_ALL}")
                    return "Error: Unable to determine source type"
        
        # Extract text based on source type
        if source_type == 'pdf':
            # Handle PDF file (local or URL)
            is_url = source.startswith('http://') or source.startswith('https://')
            
            if is_url:
                print(f"{Fore.YELLOW}Downloading PDF...{Style.RESET_ALL}")
                pdf_path_response = self.content_agent.run(f"Download the PDF from the URL: {source}")
                
                # Extract the path from the response
                pdf_path = pdf_path_response.strip().split('\n')[-1]
                if "Error" in pdf_path:
                    print(f"{Fore.RED}{pdf_path}{Style.RESET_ALL}")
                    return pdf_path
            else:
                # Use local path
                pdf_path = source
                if not os.path.exists(pdf_path):
                    error_msg = f"Error: PDF file not found at {pdf_path}"
                    print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                    return error_msg
            
            # Extract text from PDF
            print(f"{Fore.YELLOW}Extracting text from PDF...{Style.RESET_ALL}")
            extracted_text_response = self.content_agent.run(f"Extract text from the PDF file: {pdf_path}")
            
        elif source_type == 'web':
            # Handle webpage
            print(f"{Fore.YELLOW}Extracting content from webpage...{Style.RESET_ALL}")
            extracted_text_response = self.content_agent.run(f"Extract text from the webpage: {source}")
        else:
            return f"Error: Unsupported source type '{source_type}'"
        
        # Check for extraction errors
        if "Error" in extracted_text_response:
            print(f"{Fore.RED}{extracted_text_response}{Style.RESET_ALL}")
            return extracted_text_response
            
        # Generate summary
        summary_style = self.config["summary"]["style"]
        max_length = self.config["summary"]["max_length"]
        
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"""
            Summarize the following text from a {'PDF document' if source_type == 'pdf' else 'webpage'}. 
            Style: {summary_style}
            Maximum length: {max_length} words
            
            Text:
            {extracted_text_response}
            """
        
        print(f"{Fore.YELLOW}Generating summary...{Style.RESET_ALL}")
        summary = self.summary_agent.run(prompt)
        
        # Save summary if output path is provided
        if output_path:
            print(f"{Fore.YELLOW}Saving summary to file...{Style.RESET_ALL}")
            self.summary_agent.run(f"Save the following summary to {output_path}:\n\n{summary}")
            print(f"{Fore.GREEN}Summary saved to: {output_path}{Style.RESET_ALL}")
        
        return summary
    
    def process_multiple(self, sources: List[str], output_folder: str = None, 
                       custom_prompt: Optional[str] = None) -> Dict[str, str]:
        """
        Process multiple sources (PDFs or webpages) and generate summaries.
        
        Args:
            sources: List of paths or URLs to PDFs or webpages
            output_folder: Folder to save the summaries
            custom_prompt: Custom prompt for the summaries
            
        Returns:
            Dictionary mapping sources to their summaries
        """
        if output_folder is None:
            output_folder = self.config["output"]["default_folder"]
            
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        for i, source in enumerate(sources):
            print(f"{Fore.CYAN}Processing {i+1}/{len(sources)}: {source}{Style.RESET_ALL}")
            
            # Determine output path
            filename = f"summary_{i+1}.md"
            if source.startswith('http'):
                # Create a filename from the URL
                from urllib.parse import urlparse
                parsed_url = urlparse(source)
                domain = parsed_url.netloc.replace("www.", "")
                path = parsed_url.path.strip('/').replace('/', '_')
                if path:
                    filename = f"{domain}_{path}.md"
                else:
                    filename = f"{domain}.md"
            else:
                # Create a filename from the local path
                filename = f"{os.path.splitext(os.path.basename(source))[0]}_summary.md"
                
            output_path = os.path.join(output_folder, filename)
            
            # Process the source
            summary = self.process_content(source, output_path, custom_prompt)
            results[source] = summary
            
            # Add a separator between sources
            if i < len(sources) - 1:
                print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
        
        return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Content Extractor - An AI tool for extracting and summarizing content from PDFs and web pages")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single source processing
    single_parser = subparsers.add_parser("extract", help="Extract and summarize content from a single source")
    single_parser.add_argument("source", help="Path or URL to the PDF file or webpage")
    single_parser.add_argument("-o", "--output", help="Path to save the summary")
    single_parser.add_argument("-p", "--prompt", help="Custom prompt for the summary")
    single_parser.add_argument("-t", "--type", choices=["pdf", "web"], help="Type of source (pdf or web)")
    
    # Multiple sources processing
    multi_parser = subparsers.add_parser("batch", help="Extract and summarize content from multiple sources")
    multi_parser.add_argument("-f", "--file", help="Path to a text file containing sources (one per line)")
    multi_parser.add_argument("-s", "--sources", nargs="+", help="List of sources (paths or URLs)")
    multi_parser.add_argument("-o", "--output-folder", help="Folder to save the summaries")
    multi_parser.add_argument("-p", "--prompt", help="Custom prompt for the summaries")
    
    # Configuration
    config_parser = subparsers.add_parser("config", help="Configure the application")
    config_parser.add_argument("-m", "--model", help="Set the LLM model identifier")
    config_parser.add_argument("-k", "--api-key", help="Set the API key")
    config_parser.add_argument("-s", "--style", choices=["concise", "detailed", "bullet-points", "executive"], 
                             help="Set the summary style")
    config_parser.add_argument("-l", "--max-length", type=int, help="Set the maximum summary length")
    config_parser.add_argument("-c", "--config", default="config.yaml", help="Path to configuration file")
    
    # Add config option to all commands
    for p in [single_parser, multi_parser]:
        p.add_argument("-c", "--config", default="config.yaml", help="Path to configuration file")
    
    # Default behavior (for backward compatibility)
    parser.add_argument("-o", "--output", help="Path to save the summary")
    parser.add_argument("-p", "--prompt", help="Custom prompt for the summary")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to configuration file")
    
    return parser.parse_args()

def update_config(config_path, updates):
    """Update configuration file with new values"""
    # Load existing config
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(file) or {}
            elif config_path.endswith('.json'):
                config = json.load(file)
            
    # Initialize nested dictionaries if they don't exist
    if "llm" not in config:
        config["llm"] = {}
    if "summary" not in config:
        config["summary"] = {}
    
    # Update config with new values
    if updates.get("model"):
        config["llm"]["model_id"] = updates["model"]
    if updates.get("api_key"):
        config["llm"]["api_key"] = updates["api_key"]
    if updates.get("style"):
        config["summary"]["style"] = updates["style"]
    if updates.get("max_length"):
        config["summary"]["max_length"] = updates["max_length"]
    
    # Save updated config
    with open(config_path, 'w') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, file)
        elif config_path.endswith('.json'):
            json.dump(config, file, indent=2)
        else:
            yaml.dump(config, file)
    
    return config

def main():
    """Main entry point for the application"""
    # Print banner
    print(BANNER)
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle configuration command
    if args.command == "config":
        updates = {
            "model": args.model,
            "api_key": args.api_key,
            "style": args.style,
            "max_length": args.max_length
        }
        config = update_config(args.config, updates)
        print(f"{Fore.GREEN}Configuration updated!{Style.RESET_ALL}")
        
        # Display current configuration
        print(f"\n{Fore.CYAN}Current Configuration:{Style.RESET_ALL}")
        print(f"  LLM Model: {config['llm'].get('model_id', 'Not set')}")
        print(f"  API Key: {'Set' if config['llm'].get('api_key') else 'Not set'}")
        print(f"  Summary Style: {config['summary'].get('style', 'Not set')}")
        print(f"  Max Length: {config['summary'].get('max_length', 'Not set')}")
        
        return
    
    # Initialize Content Extractor
    extractor = ContentExtractor(args.config)
    
    # Handle batch processing command
    if args.command == "batch":
        sources = []
        
        # Load sources from file if provided
        if args.file:
            try:
                with open(args.file, 'r') as file:
                    sources.extend([line.strip() for line in file if line.strip()])
            except Exception as e:
                print(f"{Fore.RED}Error reading sources file: {str(e)}{Style.RESET_ALL}")
                sys.exit(1)
        
        # Add sources from command line if provided
        if args.sources:
            sources.extend(args.sources)
        
        if not sources:
            print(f"{Fore.RED}No sources provided. Use --file or --sources to specify sources.{Style.RESET_ALL}")
            sys.exit(1)
        
        try:
            # Process multiple sources
            results = extractor.process_multiple(sources, args.output_folder, args.prompt)
            
            print(f"\n{Fore.GREEN}Processed {len(results)} sources!{Style.RESET_ALL}")
            return
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    # Handle single source processing (either with "extract" command or default behavior)
    source = args.source
    if not source:
        print(f"{Fore.RED}No source provided. Please provide a path or URL to a PDF file or webpage.{Style.RESET_ALL}")
        print(f"DEBUG: args.source = {args.source}")
        sys.exit(1)
    
    # Set default output path if not provided
    if not args.output:
        default_folder = extractor.config["output"]["default_folder"]
        os.makedirs(default_folder, exist_ok=True)
        
        # Generate output filename based on input
        if source.startswith('http'):
            from urllib.parse import urlparse
            parsed_url = urlparse(source)
            domain = parsed_url.netloc.replace("www.", "")
            path = parsed_url.path.strip('/').replace('/', '_')
            if path:
                filename = f"{domain}_{path}.md"
            else:
                filename = f"{domain}.md"
        else:
            filename = f"{os.path.splitext(os.path.basename(source))[0]}_summary.md"
            
        args.output = os.path.join(default_folder, filename)
    
    try:
        # Determine source type if provided
        source_type = args.type if hasattr(args, 'type') else None
        
        # Process content
        summary = extractor.process_content(source, args.output, args.prompt, source_type)
        
        # Display summary
        print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{textwrap.fill(summary, width=80)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}Done!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()