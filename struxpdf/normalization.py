"""
Regex and Heuristics for Table Cell Normalization
"""

import re
from typing import Optional

class DataNormalizer:
    """Normalize financial data using regex and heuristics"""
    
    @staticmethod
    def normalize_currency(value: Optional[str]) -> Optional[str]:
        """Normalize currency values"""
        if not value:
            return None
        
        # Remove common currency symbols and standardize
        value = str(value).strip()
        
        # Handle various formats
        patterns = {
            r'\$\s*': '$',  # Standardize $ spacing
            r'USD\s*': '$',  # USD → $
            r'€\s*': '€',
            r'£\s*': '£',
            r'\s+': '',  # Remove internal spaces
        }
        
        for pattern, replacement in patterns.items():
            value = re.sub(pattern, replacement, value)
        
        # Standardize number formats
        # Handle: 1,234.56M, 1.234,56B (European), etc.
        if re.match(r'.*[KMB]$', value, re.IGNORECASE):
            # Keep magnitude suffix
            return value
        
        return value
    
    @staticmethod
    def normalize_percentage(value: Optional[str]) -> Optional[str]:
        """Normalize percentage values"""
        if not value:
            return None
            
        value = str(value).strip()
        
        # Ensure % symbol
        if '%' not in value:
            # If it's a decimal like 0.18, convert to 18%
            try:
                num = float(value)
                if 0 <= num <= 1:
                    return f"{num * 100:.1f}%"
            except:
                pass
        
        # Clean up spacing
        value = re.sub(r'\s*%', '%', value)
        
        return value
    
    @staticmethod
    def normalize_number(value: Optional[str]) -> Optional[str]:
        """Normalize number formats"""
        if not value:
            return None
        
        value = str(value).strip()
        
        # Remove parentheses (often used for negative numbers)
        if value.startswith('(') and value.endswith(')'):
            value = '-' + value[1:-1]
        
        # Standardize thousand separators
        # European: 1.234,56 → American: 1,234.56
        if re.match(r'[\d.]+,\d{2}$', value):
            # Likely European format
            value = value.replace('.', '').replace(',', '.')
        
        return value
    
    @staticmethod
    def extract_magnitude(value: str) -> tuple[Optional[float], Optional[str]]:
        """Extract numeric value and magnitude (K, M, B, T)"""
        if not value:
            return None, None
        
        match = re.search(r'([\d.,]+)\s*([KMBT])?', value, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                num = float(num_str)
                magnitude = match.group(2).upper() if match.group(2) else None
                return num, magnitude
            except:
                return None, None
        
        return None, None
