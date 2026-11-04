# Release validation

The final package is checked as a saved-data research artifact. No model inference is part of these checks.

- The main observation checksum matches its recorded manifest.
- Duplicate experimental keys and mismatched method pairs fail analysis.
- Source identities are checked against the recovered BABILong mapping.
- Repeated identical inputs are averaged without selecting favorable outcomes.
- Forty worked-example records reconcile their per-call input and output counts with the main observations.
- Tests cover nonmonotone threshold patterns, nonzero uncertainty for perfect small samples, repeated-input averaging, and provider-versus-answer failures.
- The main CSV tables are regenerated from the assembled release directory and compared with the included outputs.
- Local Markdown links, manifest hashes, publication-file selection, and credential patterns are checked.
- The report PDF is rendered and inspected for layout and readable figures.

A separate AI critic examines the statistical interpretation and reproducibility. This is an independent agent check, not external human peer review or a certification of scientific novelty. Exact historical model-run replay is outside the supported reproducibility claim.

To check the assembled release's integrity, local links, and recomputed tables:

```text
python scripts/validate_release.py
```

The validation command analyzes saved data only. It does not contact model providers.
