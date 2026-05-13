"""
FROZEN preprocessing & color-normalization modules.

These files are copied as-is from the paper's reference pipeline
(Cell_revisions/prediction_pipline/) and intentionally NOT refactored.

The Macenko stain normalization (utils_color_norm) and the tile-quality
filter (utils_preprocessing.evaluate_tile) are extremely sensitive to:
  - SPAMS version (different versions produce different stain matrices)
  - OpenCV / NumPy / PIL version interactions

If you "clean up" or "modernize" anything in this subpackage, you will
change predictions in ways that won't match the paper. Don't.

Only dead and commented-out code has been stripped.
"""
