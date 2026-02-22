"""
23andMe genome analysis and fitness data visualization tools.

This package provides tools for analyzing 23andMe raw genome data,
including GUI-based analysis and fitness data integration.
"""

__version__ = "0.1.0"
__author__ = "Michael E. OConnor"
__license__ = "MIT"

from genetics_23andme.core import GetSNPs, processSNPs
from genetics_23andme.fitness import FMF_NOTEWORTHY

__all__ = [
    "GetSNPs",
    "processSNPs",
    "FMF_NOTEWORTHY",
]
