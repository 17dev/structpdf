"""
Data Validation Module for structPDF
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import re

class ValidationResult(BaseModel):
    """Result of a validation check"""
    field_name: str
    is_valid: bool
    error_message: Optional[str] = None
    expected_type: Optional[str] = None
    actual_value: Any = None

class DataValidator:
    """Validate extracted data against expected formats and ranges"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def validate_type(self, field_name: str, value: Any, expected_type: str) -> ValidationResult:
        """Validate data type"""
        is_valid = True
        error_msg = None
        
        if value is None:
            is_valid = False
            error_msg = "Value is None"
        elif expected_type == "number":
            # Extract numeric value
            if isinstance(value, str):
                # Remove currency symbols, commas
                clean_val = re.sub(r'[^\d.-]', '', value)
                try:
                    float(clean_val)
                except ValueError:
                    is_valid = False
                    error_msg = f"Cannot convert '{value}' to number"
        elif expected_type == "percentage":
            if isinstance(value, str) and '%' not in value:
                is_valid = False
                error_msg = f"Expected percentage but got '{value}'"
        elif expected_type == "currency":
            if isinstance(value, str) and not any(c in value for c in ['$', '€', '£', 'M', 'B', 'K']):
                is_valid = False
                error_msg = f"Expected currency but got '{value}'"
        
        result = ValidationResult(
            field_name=field_name,
            is_valid=is_valid,
            error_message=error_msg,
            expected_type=expected_type,
            actual_value=value
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_range(
        self, 
        field_name: str, 
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> ValidationResult:
        """Validate value is within expected range"""
        is_valid = True
        error_msg = None
        
        if value is None:
            is_valid = False
            error_msg = "Value is None"
        else:
            try:
                # Extract numeric value
                if isinstance(value, str):
                    clean_val = re.sub(r'[^\d.-]', '', value)
                    num_val = float(clean_val)
                else:
                    num_val = float(value)
                
                if min_val is not None and num_val < min_val:
                    is_valid = False
                    error_msg = f"Value {num_val} is below minimum {min_val}"
                
                if max_val is not None and num_val > max_val:
                    is_valid = False
                    error_msg = f"Value {num_val} exceeds maximum {max_val}"
                    
            except (ValueError, TypeError) as e:
                is_valid = False
                error_msg = f"Cannot validate range: {e}"
        
        result = ValidationResult(
            field_name=field_name,
            is_valid=is_valid,
            error_message=error_msg,
            expected_type=f"range[{min_val}, {max_val}]",
            actual_value=value
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_cross_field_consistency(
        self,
        field_relationships: Dict[str, tuple[str, str]]  # {field1: (field2, relationship)}
    ) -> List[ValidationResult]:
        """
        Validate consistency between related fields
        
        Example:
            {'gross_profit': ('revenue', 'less_than')}  
            # gross_profit should be < revenue
        """
        results = []
        
        for field1, (field2, relationship) in field_relationships.items():
            # This would require access to all fields
            # Implementation depends on having all data available
            # Placeholder for now
            pass
        
        return results
    
    def validate_format(self, field_name: str, value: Any, pattern: str) -> ValidationResult:
        """Validate value matches expected regex pattern"""
        is_valid = True
        error_msg = None
        
        if value is None:
            is_valid = False
            error_msg = "Value is None"
        elif not re.match(pattern, str(value)):
            is_valid = False
            error_msg = f"Value '{value}' doesn't match pattern '{pattern}'"
        
        result = ValidationResult(
            field_name=field_name,
            is_valid=is_valid,
            error_message=error_msg,
            expected_type=f"pattern: {pattern}",
            actual_value=value
        )
        
        self.validation_results.append(result)
        return result
    
    def get_failed_validations(self) -> List[ValidationResult]:
        """Get all failed validation results"""
        return [r for r in self.validation_results if not r.is_valid]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.is_valid)
        
        return {
            'total_checks': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'failed_checks': self.get_failed_validations()
        }
