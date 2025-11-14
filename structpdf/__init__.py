"""structPDF: DSPy-powered PDF extraction wrapper"""
from .core import structPDF, ChunkingConfig, CompanyFinancialData, QuarterlyData
from .normalization import DataNormalizer
from .confidence import ConfidenceScorer, FieldConfidence
from .quality_assurance import QualityAssurance, QAResult
from .validation import DataValidator, ValidationResult
from .optimizers import (
    structPDFOptimizer,
    RefinementEngine,
    OptimizerConfig,
    financial_extraction_metric
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "structPDF",
    "ChunkingConfig",
    "CompanyFinancialData",
    "QuarterlyData",
    # Normalization
    "DataNormalizer",
    # Confidence
    "ConfidenceScorer",
    "FieldConfidence",
    # Quality Assurance
    "QualityAssurance",
    "QAResult",
    # Validation
    "DataValidator",
    "ValidationResult",
    # Optimizers
    "structPDFOptimizer",
    "RefinementEngine",
    "OptimizerConfig",
    "financial_extraction_metric"
]
