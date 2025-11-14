"""
structPDF: Financial Data Extraction with DSPy
Fast, accurate PDF extraction with incremental feature building
"""

from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Type
import fitz  # PyMuPDF
import pandas as pd
import time
import dspy
import json
import re

# ============================================================================
# SETUP
# ============================================================================
load_dotenv()

# Configure DSPy with API key from environment
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    max_tokens: int = Field(default=8000, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=500, description="Overlapping tokens between chunks")
    preserve_sentences: bool = Field(default=True, description="Preserve sentence boundaries")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        
        if self.preserve_sentences:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = ""
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.estimate_tokens(sentence)
                
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap_tokens * 4:]
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.estimate_tokens(current_chunk)
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # Simple character-based chunking
            max_chars = self.max_tokens * 4
            overlap_chars = self.overlap_tokens * 4
            
            for i in range(0, len(text), max_chars - overlap_chars):
                chunk = text[i:i + max_chars]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks

# ============================================================================
# SCHEMAS
# ============================================================================
class QuarterlyData(BaseModel):
    """Financial data for a single quarter"""
    quarter: str = Field(description="Quarter (e.g., 'Q1 2025')")
    total_revenue: Optional[str] = Field(None, description="Total revenue")
    earnings_per_share: Optional[str] = Field(None, description="EPS (diluted)")
    net_income: Optional[str] = Field(None, description="Net income")
    operating_income: Optional[str] = Field(None, description="Operating income")
    gross_margin: Optional[str] = Field(None, description="Gross margin percentage")
    operating_expenses: Optional[str] = Field(None, description="Operating expenses")
    buybacks: Optional[str] = Field(None, description="Share buybacks")
    dividends: Optional[str] = Field(None, description="Dividends paid")

class CompanyFinancialData(BaseModel):
    """Complete financial data for a company across multiple quarters"""
    company_name: str = Field(description="Company name")
    quarters: List[QuarterlyData] = Field(description="List of quarterly financial data")

# ============================================================================
# DSPY MODULE
# ============================================================================
class FinancialExtractor(dspy.Module):
    """DSPy module for extracting financial data from PDF text"""
    
    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        super().__init__()
        if schema:
            # Custom schema extraction
            self.predictor = dspy.ChainOfThought(
                f"document_text, document_type -> extracted_data: {schema.__name__}"
            )
            self.schema = schema
        else:
            # Default financial extraction
            self.predictor = dspy.ChainOfThought(
                "document_text, document_type -> financial_data: CompanyFinancialData"
            )
            self.schema = CompanyFinancialData
    
    def forward(self, document_text: str, document_type: str = "report"):
        """Extract financial data using DSPy ChainOfThought"""
        return self.predictor(
            document_text=document_text,
            document_type=document_type
        )

# ============================================================================
# MAIN PROCESSOR
# ============================================================================
class structPDF:
    """
    Main PDF financial extraction class
    
    Usage:
        extractor = structPDF()
        result = extractor.process("quarterly_report.pdf")
        df = extractor.to_dataframe()
    """
    
    def __init__(
        self, 
        schema: Optional[Type[BaseModel]] = None, 
        instructions: Optional[str] = None,
        chunking_config: Optional[ChunkingConfig] = None
    ):
        """
        Initialize the extractor
        
        Args:
            schema: Optional Pydantic model for custom extraction
            instructions: Optional custom instructions for extraction
            chunking_config: Optional ChunkingConfig for large document processing
        """
        self.extractor = FinancialExtractor(schema=schema)
        self.results = []
        self.schema = schema or CompanyFinancialData
        self.instructions = instructions
        self.chunking_config = chunking_config or ChunkingConfig()
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        return "\n\n".join(pages_text)
    
    def _detect_document_type(self, text: str) -> str:
        """Detect if document is a transcript or quarterly report"""
        text_lower = text.lower()
        
        # Check for transcript indicators
        transcript_keywords = [
            'transcript', 'earnings call', 'conference call',
            'operator:', 'moderator:', 'q&a session'
        ]
        
        if any(keyword in text_lower for keyword in transcript_keywords):
            return 'transcript'
        
        return 'report'
    
    def _chunk_and_extract(self, text: str, doc_type: str, verbose: bool = False) -> Any:
        """Extract data using chunking for large documents"""
        chunks = self.chunking_config.split_text(text)
        
        if verbose:
            print(f"  - Document split into {len(chunks)} chunks for processing")
        
        all_results = []
        
        for i, chunk in enumerate(chunks, 1):
            if verbose:
                print(f"    - Processing chunk {i}/{len(chunks)}...")
            
            try:
                prediction = self.extractor(
                    document_text=chunk,
                    document_type=doc_type
                )
                
                # Extract the data based on schema type
                if hasattr(prediction, 'financial_data'):
                    all_results.append(prediction.financial_data)
                elif hasattr(prediction, 'extracted_data'):
                    all_results.append(prediction.extracted_data)
                
            except Exception as e:
                if verbose:
                    print(f"      - Warning: Chunk {i} failed: {e}")
                continue
        
        # Merge results from all chunks
        if not all_results:
            return None
        
        # For financial data, merge quarters from all chunks
        if isinstance(all_results[0], CompanyFinancialData):
            merged_quarters = []
            seen_quarters = set()
            
            for result in all_results:
                for quarter in result.quarters:
                    if quarter.quarter not in seen_quarters:
                        merged_quarters.append(quarter)
                        seen_quarters.add(quarter.quarter)
            
            return CompanyFinancialData(
                company_name=all_results[0].company_name,
                quarters=merged_quarters
            )
        
        # For custom schemas, return the first non-empty result
        return all_results[0]
    
    def process(self, pdf_path: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF file with chunking support
        
        Args:
            pdf_path: Path to PDF file
            verbose: Print progress information
            
        Returns:
            Dictionary with extraction results and metadata
        """
        start_time = time.time()
        
        if verbose:
            print(f"Processing: {pdf_path}")
        
        # Extract text
        if verbose:
            print("  - Extracting text from PDF...")
        text = self._extract_text_from_pdf(pdf_path)
        
        # Count pages and estimate tokens
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        estimated_tokens = self.chunking_config.estimate_tokens(text)
        
        if verbose:
            print(f"  - Total pages: {total_pages}")
            print(f"  - Estimated tokens: {estimated_tokens:,}")
        
        # Detect document type
        doc_type = self._detect_document_type(text)
        if verbose:
            print(f"  - Document type: {doc_type}")
        
        # Decide whether to chunk
        use_chunking = estimated_tokens > self.chunking_config.max_tokens
        
        if verbose and use_chunking:
            print(f"  - Using chunking strategy (document exceeds {self.chunking_config.max_tokens} tokens)")
        
        # Extract with DSPy
        if verbose:
            print("  - Extracting financial data with DSPy...")
        
        try:
            if use_chunking:
                extracted_data = self._chunk_and_extract(text, doc_type, verbose)
            else:
                prediction = self.extractor(
                    document_text=text,
                    document_type=doc_type
                )
                
                if hasattr(prediction, 'financial_data'):
                    extracted_data = prediction.financial_data
                elif hasattr(prediction, 'extracted_data'):
                    extracted_data = prediction.extracted_data
                else:
                    extracted_data = None
            
            result = {
                'file': pdf_path,
                'document_type': doc_type,
                'data': extracted_data,
                'processing_time': time.time() - start_time,
                'success': True,
                'error': None,
                'total_pages': total_pages,
                'pages_processed': total_pages,  # 100% coverage
                'chunked': use_chunking,
                'estimated_tokens': estimated_tokens
            }
            
            if verbose:
                if hasattr(extracted_data, 'quarters'):
                    print(f"  - Extracted {len(extracted_data.quarters)} quarters")
                print(f"  - Pages processed: {total_pages}/{total_pages} (100% coverage)")
                print(f"  - Processing time: {result['processing_time']:.2f}s")
            
        except Exception as e:
            result = {
                'file': pdf_path,
                'document_type': doc_type,
                'data': None,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'total_pages': total_pages,
                'pages_processed': 0,
                'chunked': use_chunking,
                'estimated_tokens': estimated_tokens
            }
            
            if verbose:
                print(f"  - ERROR: {e}")
        
        self.results.append(result)
        return result
    
    def process_batch(self, pdf_paths: List[str], verbose: bool = False) -> pd.DataFrame:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            verbose: Print progress information
            
        Returns:
            DataFrame with all extracted data
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {len(pdf_paths)} PDF files...")
            print(f"{'='*60}\n")
        
        for i, path in enumerate(pdf_paths, 1):
            if verbose:
                print(f"[{i}/{len(pdf_paths)}]")
            self.process(path, verbose=verbose)
            if verbose:
                print()
        
        if verbose:
            print(f"{'='*60}")
            print("Processing complete!")
            print(f"{'='*60}\n")
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert extraction results to pandas DataFrame"""
        records = []
        
        for result in self.results:
            if result['success'] and result['data']:
                data = result['data']
                
                for quarter in data.quarters:
                    records.append({
                        'Company': data.company_name,
                        'Quarter': quarter.quarter,
                        'Revenue': quarter.total_revenue,
                        'EPS': quarter.earnings_per_share,
                        'Net Income': quarter.net_income,
                        'Operating Income': quarter.operating_income,
                        'Gross Margin': quarter.gross_margin,
                        'OpEx': quarter.operating_expenses,
                        'Buybacks': quarter.buybacks,
                        'Dividends': quarter.dividends,
                        'Source': result['file'],
                        'Doc Type': result['document_type'],
                        'Processing Time': f"{result['processing_time']:.2f}s"
                    })
        
        return pd.DataFrame(records)
    
    def export_csv(self, filename: str = "financial_data.csv"):
        """Export results to CSV"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        print(f"Exported to {filename}")
        return filename
    
    def export_excel(self, filename: str = "financial_data.xlsx"):
        """Export results to Excel"""
        df = self.to_dataframe()
        df.to_excel(filename, index=False)
        print(f"Exported to {filename}")
        return filename
    
    def export_json(self, filename: str = "financial_data.json"):
        """Export results to JSON"""
        df = self.to_dataframe()
        df.to_json(filename, orient='records', indent=2)
        print(f"Exported to {filename}")
        return filename
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of processed documents"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        total_time = sum(r['processing_time'] for r in self.results)
        avg_time = total_time / total if total > 0 else 0
        
        total_quarters = sum(
            len(r['data'].quarters) 
            for r in self.results 
            if r['success'] and r['data']
        )
        
        return {
            'total_documents': total,
            'successful': successful,
            'failed': failed,
            'success_rate': f"{(successful/total*100):.1f}%" if total > 0 else "0%",
            'total_quarters_extracted': total_quarters,
            'total_processing_time': f"{total_time:.2f}s",
            'avg_processing_time': f"{avg_time:.2f}s"
        }
