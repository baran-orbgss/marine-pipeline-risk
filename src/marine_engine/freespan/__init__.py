"""Generic pipeline free-span geometry and support-loss susceptibility screening (MAR-025).

Three concepts are kept structurally separate throughout this package:

A. MEASURED / OBSERVED FREE-SPAN GEOMETRY -- derived from a real or
   synthetic operator pipe vertical profile and seabed support profile.
B. GENERIC SUPPORT-LOSS SUSCEPTIBILITY SCREENING -- what an optional,
   explicitly-typed seabed-lowering scenario would do to that measured
   geometry. Never merged with (A).
C. SOURCE-REPORTED / OBSERVED FREE-SPAN EVIDENCE -- real third-party
   evidence (NSTA registry, PL854 Table B.1) ingested and audited, never
   treated as validation of (A) or (B) and never used to derive a
   susceptibility result.

This package performs NO structural free-span integrity assessment (no
DNV-RP-F105, no VIV, no fatigue life, no ULS/FLS, no failure probability).
"""
