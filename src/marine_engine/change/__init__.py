"""Generic, project-agnostic multi-epoch seabed change / DEM-of-Difference
engine (MAR-021).

Zero dependency on any specific project, dataset, or PL854/Sheringham-
specific code -- every function takes plain arrays/facts and physical-
metre parameters. Reuses `marine_engine.terrain.canonical` for the shared
`bed_elevation_m` sign convention (both epochs must be canonicalised
through that SAME module, exactly once each, before any subtraction) and
`marine_engine.terrain.raster_io` for GeoTIFF writing -- both already
fully generic, so importing them here is not a scientific dependency, just
shared infrastructure reuse.

No risk score, no freespan/scour susceptibility, no sediment-transport
model, no future erosion/deposition prediction, no ML anywhere in this
package -- this engine computes and reports OBSERVED multi-epoch seabed
elevation change only.
"""
