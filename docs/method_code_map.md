# Method-Code Map

This is the only public release document that maps readable names to internal experiment IDs. Other public prose should prefer the readable names. The result-row table below also fixes the canonical bridge from artifact display labels to public roles.

## Result-Row Mapping

| Public name | Artifact row label | Canonical public role | Internal code bridge | Meaning |
| --- | --- | --- | --- | --- |
| GPA-Net | selected final system | protagonist | HPO-L4-02 + FUS-02 | Selected subject-held-out repeated-walk inference package. |
| Historical anchor | historical anchor | anchor | L4 + FUS-02 | Accepted pre-release repeated-walk anchor. |
| Diagnostic fusion variant | diagnostic comparator | diagnostic/support | HPO-L4-03 + logit_mean | Classification-heavy fusion variant under a different prediction-time fusion scope; not the protagonist. |
| Strict dual-attention baseline | strict dual-attention baseline | support/context | Phase 3 strict baseline | Trial-level context row, not a repeated-walk direct replacement. |
| Conservative MLP baseline | conservative MLP baseline | support/context | E2E-15 MLP baseline | Fixed-budget trial-level baseline for context. |
| Conservative CNN baseline | conservative CNN baseline | support/context | E2E-15 CNN baseline | Fixed-budget trial-level baseline for context. |

## Supporting Component Mapping

| Public name | Internal code bridge | Meaning |
| --- | --- | --- |
| robust input normalization | FE-13 | Fold-safe robust scaling of input features. |
| pressure target/export overlay | FE-08 | Pressure timing/load target-export overlay used for target construction support. |
| prediction-time repeated-walk mean fusion | FUS-02 | Mean aggregation over repeated held-out walk predictions. |

Use this map for reproducibility and source lookup only. It does not change the public claim scope in [claim_scope.md](claim_scope.md).
