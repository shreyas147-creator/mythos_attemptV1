# Model Families

Use separate folders so local experiments do not get mixed with production-scale specs.

- `local/`: runnable byte-level model for direct training and efficiency work.
- `1t_model/`: 1T-class planning/specification only.

Rule: promote an idea from `local/` to `one_trillion/` only after local benchmarks show a measurable win.
