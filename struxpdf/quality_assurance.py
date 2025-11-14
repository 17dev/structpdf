"""
Quality Assurance Module for StruxPDF
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd

class QAResult(BaseModel):
    """Quality assurance check result"""
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    issues: List[str] = []

class QualityAssurance:
    """Quality assurance checks for extracted data"""
    
    def __init__(self, critical_fields: List[str] = None):
        """
        Initialize QA module
        
        Args:
            critical_fields: Fields that must be present (e.g., ['revenue', 'eps'])
        """
        self.critical_fields = critical_fields or ['revenue', 'eps', 'net_income']
        self.qa_results: List[QAResult] = []
    
    def validate_completeness(self, df: pd.DataFrame) -> QAResult:
        """Check data completeness"""
        issues = []
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Check critical fields
        for field in self.critical_fields:
            if field.replace('_', ' ').title() in df.columns:
                col_name = field.replace('_', ' ').title()
                missing = df[col_name].isna().sum()
                if missing > 0:
                    issues.append(f"{col_name}: {missing} missing values")
        
        result = QAResult(
            check_name="Data Completeness",
            passed=completeness >= 0.9,
            score=completeness,
            details=f"{completeness*100:.1f}% of cells have data",
            issues=issues
        )
        
        self.qa_results.append(result)
        return result
    
    def validate_critical_metrics(self, df: pd.DataFrame) -> QAResult:
        """Validate that critical metrics are present"""
        issues = []
        critical_cols = [f.replace('_', ' ').title() for f in self.critical_fields]
        
        missing_cols = [col for col in critical_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {', '.join(missing_cols)}")
        
        # Check if critical fields have values
        present_critical = [col for col in critical_cols if col in df.columns]
        for col in present_critical:
            if df[col].isna().all():
                issues.append(f"{col} has no values")
        
        score = (len(present_critical) / len(critical_cols)) if critical_cols else 1.0
        
        result = QAResult(
            check_name="Critical Metrics",
            passed=len(issues) == 0,
            score=score,
            details=f"{len(present_critical)}/{len(critical_cols)} critical metrics present",
            issues=issues
        )
        
        self.qa_results.append(result)
        return result
    
    def validate_page_coverage(self, results: List[Dict[str, Any]]) -> QAResult:
        """Verify 100% page coverage"""
        issues = []
        total_pages = 0
        processed_pages = 0
        
        for result in results:
            if result.get('success'):
                total_pages += result.get('total_pages', 0)
                processed_pages += result.get('pages_processed', 0)
        
        coverage = processed_pages / total_pages if total_pages > 0 else 0
        
        if coverage < 1.0:
            issues.append(f"Only {processed_pages}/{total_pages} pages processed ({coverage*100:.1f}%)")
        
        result = QAResult(
            check_name="Page Coverage",
            passed=coverage >= 1.0,
            score=coverage,
            details=f"{processed_pages}/{total_pages} pages processed",
            issues=issues
        )
        
        self.qa_results.append(result)
        return result
    
    def validate_accuracy(
        self, 
        extracted_df: pd.DataFrame, 
        ground_truth_df: Optional[pd.DataFrame] = None
    ) -> QAResult:
        """
        Validate accuracy against ground truth
        
        Args:
            extracted_df: DataFrame with extracted data
            ground_truth_df: Optional DataFrame with ground truth values
        """
        if ground_truth_df is None:
            return QAResult(
                check_name="Accuracy Validation",
                passed=True,
                score=1.0,
                details="No ground truth provided - skipped",
                issues=[]
            )
        
        issues = []
        matches = 0
        total = 0
        
        # Compare common columns
        common_cols = set(extracted_df.columns) & set(ground_truth_df.columns)
        
        for col in common_cols:
            for idx in range(min(len(extracted_df), len(ground_truth_df))):
                total += 1
                extracted_val = str(extracted_df.iloc[idx][col]).strip().lower()
                truth_val = str(ground_truth_df.iloc[idx][col]).strip().lower()
                
                # Fuzzy match (remove spaces, currency symbols)
                extracted_clean = extracted_val.replace(' ', '').replace('$', '').replace(',', '')
                truth_clean = truth_val.replace(' ', '').replace('$', '').replace(',', '')
                
                if extracted_clean == truth_clean:
                    matches += 1
                else:
                    issues.append(f"{col} row {idx}: '{extracted_val}' vs '{truth_val}'")
        
        accuracy = matches / total if total > 0 else 0
        
        result = QAResult(
            check_name="Accuracy Validation",
            passed=accuracy >= 0.9,
            score=accuracy,
            details=f"{matches}/{total} fields match ground truth ({accuracy*100:.1f}%)",
            issues=issues[:10]  # Limit to first 10 issues
        )
        
        self.qa_results.append(result)
        return result
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        results: List[Dict[str, Any]],
        ground_truth: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Run all QA checks and return summary"""
        self.qa_results = []
        
        # Run all checks
        self.validate_completeness(df)
        self.validate_critical_metrics(df)
        self.validate_page_coverage(results)
        if ground_truth is not None:
            self.validate_accuracy(df, ground_truth)
        
        # Calculate overall score
        total_score = sum(r.score for r in self.qa_results) / len(self.qa_results)
        all_passed = all(r.passed for r in self.qa_results)
        
        return {
            'overall_passed': all_passed,
            'overall_score': total_score,
            'total_checks': len(self.qa_results),
            'passed_checks': sum(1 for r in self.qa_results if r.passed),
            'results': self.qa_results
        }
