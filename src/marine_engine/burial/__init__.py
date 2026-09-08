"""Generic linear-asset burial/exposure state and cover-margin screening.

Zero dependency on any specific project or dataset. Three concepts stay
separate everywhere in this package (MAR-024 Section 1):

A. OBSERVED / MEASURED BURIAL STATE -- what a real survey actually measured.
B. SOURCE-INTERPRETED EXPOSURE EVIDENCE -- what the source explicitly says
   is exposed, never inferred from an unclear numeric convention.
C. FUTURE EXPOSURE SUSCEPTIBILITY -- only ever produced when a defensible
   observed or explicitly-labelled scenario seabed-lowering input exists.
"""
