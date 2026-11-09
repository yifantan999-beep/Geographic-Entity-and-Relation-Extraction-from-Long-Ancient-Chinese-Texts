"""
ITER Extension Modules for Wenyanwen (文言文) Geographical NER + RE.

This package contains:
- fact_decomposition: Two-stage entity/relation extraction inspired by JEDI
- multiscale_geo: Multi-scale geographic representation with gated fusion
"""

from .fact_decomposition import FactDecompositionModule
from .multiscale_geo import MultiScaleGeoModule

__all__ = ["FactDecompositionModule", "MultiScaleGeoModule"]
from .document_relation import DocumentRelationConfig, DocumentRelationModel
from .long_context_ner import LongContextNERRefiner

__all__ = ["DocumentRelationConfig", "DocumentRelationModel", "LongContextNERRefiner"]
