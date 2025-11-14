"""
Confidence Scoring System for structPDF
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import fitz  # PyMuPDF for coordinate extraction

class FieldConfidence(BaseModel):
    """Confidence score for a single field"""
    field_name: str
    value: Any
    confidence: float  # 0.0 to 1.0
    source_page: Optional[int] = None
    source_coordinates: Optional[Dict[str, float]] = None  # {x0, y0, x1, y1}
    source_text_block: Optional[str] = None

class ConfidenceScorer:
    """Calculate confidence scores for extracted data"""
    
    def __init__(self):
        self.field_scores: List[FieldConfidence] = []
    
    def calculate_self_consistency(self, values: List[Any]) -> float:
        """
        Calculate confidence based on consistency across chunks
        If same value appears in multiple chunks, higher confidence
        """
        if not values:
            return 0.0
        
        if len(values) == 1:
            return 0.5  # Medium confidence for single occurrence
        
        # Count unique values
        unique_values = set(str(v) for v in values if v is not None)
        
        if len(unique_values) == 1:
            # All values agree - high confidence
            return min(1.0, 0.5 + (len(values) * 0.1))
        else:
            # Disagreement - lower confidence
            majority_count = max(str(v) for v in values).count
            return majority_count / len(values)
    
    def extract_source_coordinates(
        self, 
        pdf_path: str, 
        search_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract coordinates of text in PDF for source grounding
        """
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Search for text instances
                text_instances = page.search_for(search_text)
                
                if text_instances:
                    rect = text_instances[0]  # First occurrence
                    
                    return {
                        'page': page_num + 1,
                        'coordinates': {
                            'x0': rect.x0,
                            'y0': rect.y0,
                            'x1': rect.x1,
                            'y1': rect.y1
                        },
                        'block_text': self._extract_block_text(page, rect)
                    }
            
            doc.close()
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
        
        return None
    
    def _extract_block_text(self, page, rect) -> str:
        """Extract surrounding text block for context"""
        # Expand rectangle for context
        expanded_rect = fitz.Rect(
            rect.x0 - 50,
            rect.y0 - 20,
            rect.x1 + 50,
            rect.y1 + 20
        )
        
        blocks = page.get_text("blocks", clip=expanded_rect)
        return " ".join([b[4] for b in blocks if len(b) > 4])
    
    def score_field(
        self,
        field_name: str,
        value: Any,
        source_values: List[Any],
        pdf_path: Optional[str] = None
    ) -> FieldConfidence:
        """
        Score a single field with confidence
        
        Args:
            field_name: Name of the field
            value: Extracted value
            source_values: All values found across chunks (for consistency)
            pdf_path: Optional path to PDF for coordinate extraction
        """
        # Calculate base confidence from consistency
        confidence = self.calculate_self_consistency(source_values)
        
        # Boost confidence if value is not None/empty
        if value is not None and str(value).strip():
            confidence += 0.2
        
        # Boost confidence if value matches expected format
        if self._matches_expected_format(field_name, value):
            confidence += 0.1
        
        # Cap at 1.0
        confidence = min(1.0, confidence)
        
        # Extract source coordinates if PDF path provided
        source_info = None
        if pdf_path and value:
            source_info = self.extract_source_coordinates(pdf_path, str(value)[:50])
        
        field_conf = FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=confidence,
            source_page=source_info['page'] if source_info else None,
            source_coordinates=source_info['coordinates'] if source_info else None,
            source_text_block=source_info['block_text'] if source_info else None
        )
        
        self.field_scores.append(field_conf)
        return field_conf
    
    def _matches_expected_format(self, field_name: str, value: Any) -> bool:
        """Check if value matches expected format for field"""
        if not value:
            return False
        
        value_str = str(value).lower()
        
        # Define format expectations
        format_checks = {
            'revenue': lambda v: any(c in v for c in ['$', 'b', 'm', 'billion', 'million']),
            'eps': lambda v: '$' in v or '.' in v,
            'margin': lambda v: '%' in v,
            'quarter': lambda v: 'q' in v and any(d in v for d in '1234'),
        }
        
        for key, check_fn in format_checks.items():
            if key in field_name.lower():
                return check_fn(value_str)
        
        return True  # Default to true if no specific check
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all fields"""
        if not self.field_scores:
            return 0.0
        
        return sum(f.confidence for f in self.field_scores) / len(self.field_scores)
    
    def get_low_confidence_fields(self, threshold: float = 0.5) -> List[FieldConfidence]:
        """Get fields with confidence below threshold"""
        return [f for f in self.field_scores if f.confidence < threshold]
