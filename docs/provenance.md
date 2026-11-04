# Provenance and supplementary checks

## Study history

Mikhail Sadovskiy's original project was conducted in spring 2026. The selected API observations are dated 16-18 May. This release is a completed analysis of existing observations; it does not represent a new model benchmark run.

The main export contains 9,800 records selected by the recorded final manifest. It includes unsuccessful attempts; 9,506 remain for conditional accuracy. A separate 240-record provider-failure phase is outside this selection. Original logs remain unchanged. The release includes their hashes and the export's transformation description, but omits provider error bodies and account metadata.

For BABILong, all 145 input hashes were matched to cached source examples at revision `ee0d588794c7ac098062ee0d247c733d62e94fe2`. Their source questions, answers, and ordered facts were checked. The recovered mapping is provided with the observations. This recovery is not evidence that the original runner pinned that revision. The mapping identifies 88 source-task IDs and 70 derived prefix families.

For early NIAH records lacking gold metadata, answers were recovered only when the generator reproduced the exact input hash. No gold answer was inferred from a model output. The export records this provenance. The 40-row workload evidence file preserves per-call token counts for the longest GPT-4o-mini full-context/memory comparison; its sums must agree with the main export.

## November supplementary observations

On 2 November, before preparation of this final release, a frozen 150-response diagnostic evaluated 30 derived story groups with five conditions. A separately frozen 60-response check compared short prompts on the same development tasks. These are not untouched confirmation data.

| Check | Recorded outcome |
|---|---|
| Full long context | 12/30 correct |
| RAG in retrieval order | 4/30 |
| Same retrieved chunks in source order | 4/30 |
| Complete story with matched filler | 15/30 |
| Short story | 15/30 |
| Later original short-prompt replay | 17/30 |
| Official BABILong short prompt | 16/30 |

The privileged full-story condition changes evidence, distractors, order, and layout together. The prompt comparison changes the complete prompt package, not one isolated feature. The 17/30 contemporaneous replay is its control; the earlier 15/30 is not substituted. These checks do not validate a universal prompt fix, a retrieval mechanism, or a precise length effect.

All 210 responses ended normally, and provider input-token counts matched local tokenizer counts. Responses include multiple backend fingerprint values despite a constant model alias and temperature zero. The total recorded charge was $0.09913710. No further inference was performed for this release.

Six earlier local diagnostic calls used raw repeated-word prompts to inspect serving-window token accounting. They measured no reasoning accuracy. Those observations and the API check tables are included in `data/supplementary/`; source corpora and credentials are not included.

## Attribution and assistance

Mikhail Sadovskiy independently formulated the research question, designed the experiments, selected the tasks and comparison conditions, and carried out the original study in spring 2026. He directed the implementation and analysis without day-to-day supervision.

AI agents supported coding, analysis and report preparation, including code revision, statistical checks and editing for the public release. The author made the research and experimental decisions and remains responsible for the interpretation and conclusions. A separate AI-agent critique was used as an internal check during report preparation. It was not external human peer review.

The project uses existing benchmarks and published processing ideas, credited in [Third-party notices](../THIRD_PARTY_NOTICES.md). The author's contribution is the project design, implementation, experiments and analysis.

Exact replay of the spring requests is limited by the unrecorded dirty source state, partial intermediate prompt logging, and changing service backends. The included implementation is an inspection snapshot, with later changes documented in `experiment/SNAPSHOT.json`. The saved-data analysis is the supported reproducibility target.
