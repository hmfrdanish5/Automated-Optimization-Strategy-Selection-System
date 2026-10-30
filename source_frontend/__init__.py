"""Restricted C-like source front end for the TAC optimizer pipeline."""

from source_frontend.compiler import CompilationResult, compile_source, compile_source_or_raise

__all__ = ["CompilationResult", "compile_source", "compile_source_or_raise"]
