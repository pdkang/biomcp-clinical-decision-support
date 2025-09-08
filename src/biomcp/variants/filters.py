"""Variant filtering functionality."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class VariantFilter(BaseModel):
    """Base class for variant filters."""
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filter to variants."""
        return variants


class FrequencyFilter(VariantFilter):
    """Filter variants by frequency."""
    
    min_frequency: float = Field(default=0.0, description="Minimum frequency")
    max_frequency: float = Field(default=1.0, description="Maximum frequency")
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter variants by frequency range."""
        filtered = []
        for variant in variants:
            frequency = variant.get("frequency", 0.0)
            if self.min_frequency <= frequency <= self.max_frequency:
                filtered.append(variant)
        return filtered


class CADDFilter(VariantFilter):
    """Filter variants by CADD score."""
    
    min_cadd: float = Field(default=0.0, description="Minimum CADD score")
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter variants by minimum CADD score."""
        filtered = []
        for variant in variants:
            cadd_score = variant.get("cadd_score", 0.0)
            if cadd_score >= self.min_cadd:
                filtered.append(variant)
        return filtered


class SignificanceFilter(VariantFilter):
    """Filter variants by clinical significance."""
    
    significance: List[str] = Field(default_factory=list, description="Allowed significance values")
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter variants by clinical significance."""
        if not self.significance:
            return variants
        
        filtered = []
        for variant in variants:
            variant_significance = variant.get("clinical_significance", "")
            if variant_significance in self.significance:
                filtered.append(variant)
        return filtered


class GeneFilter(VariantFilter):
    """Filter variants by gene."""
    
    genes: List[str] = Field(default_factory=list, description="Allowed genes")
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter variants by gene."""
        if not self.genes:
            return variants
        
        filtered = []
        for variant in variants:
            variant_gene = variant.get("gene", "")
            if variant_gene in self.genes:
                filtered.append(variant)
        return filtered


class CompositeFilter(VariantFilter):
    """Composite filter that applies multiple filters."""
    
    filters: List[VariantFilter] = Field(default_factory=list, description="List of filters to apply")
    
    def apply(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all filters in sequence."""
        filtered = variants
        for filter_obj in self.filters:
            filtered = filter_obj.apply(filtered)
        return filtered