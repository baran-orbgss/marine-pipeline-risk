"""Generic offshore CPT/CPTU evidence contract and liquefaction-input readiness (MAR-032).

This package is the geotechnical EVIDENCE foundation that must exist before any defensible
liquefaction-triggering model can be authorized. It normalizes evidence; it never interprets it.

Structural boundary (MAR-032 Section 11) -- these layers are kept separate and never collapsed:

    source collection declaration      (what the publisher SAYS the campaign contained)
    source package identity            (bytes, SHA-256, file inventory)
    documentary CPT evidence           (PDF/scanned/rendered logs -- never digitized)
    machine-readable measured values   (only when a structured numeric source actually exists)
    source interpretation              (report legends, literature -- never promoted to measured)
    derived values                     (QA statistics of the canonical table)
    liquefaction-model outputs         (NONE exist in MAR-032)

Permanent invariants:

    a CPT report                 != a canonical numeric CPT profile
    a canonical numeric profile  != a liquefaction conclusion
    qc                           != qt  (never silently equated, never corrected here)
    unknown unit                 -> null (never guessed from magnitude)
    unresolved depth reference   -> depth_bsf_m = null

Explicitly NOT implemented anywhere in this package (later MAR authority required): CSR, CRR,
liquefaction factor of safety, probability, LPI/LSN, settlement, lateral spreading,
post-liquefaction strength, pipeline flotation, wave-induced pore pressure. Earthquake-induced
and wave/current-induced seabed liquefaction are distinct mechanisms and are reported separately.

Provider-specific source logic (e.g. the real Sheringham Shoal 2008 GEO CPTU package) lives in
`marine_engine.providers.geotechnical`, outside this generic contract.
"""
