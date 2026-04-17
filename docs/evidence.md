# Evidence Notes

This page keeps detailed evidence outside the README while preserving the same public naming: GPA-Net is the headline method, and comparison rows are described by a canonical public-role taxonomy.

## Headline Metrics

GPA-Net is reported under `subject_id` GroupKFold with prediction-time repeated-walk mean fusion. `Score` is the mean of Balanced accuracy, Macro F1, and AUROC, so higher is better. The selected same-scope point estimates are:

| Method | Artifact row label | Canonical public role | Scope | Score ↑ | Balanced accuracy ↑ | Macro F1 ↑ | AUROC ↑ | Upper RMSE ↓ | Lower RMSE ↓ | Total RMSE ↓ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPA-Net | selected final system | protagonist | subject-held-out repeated-walk | 0.7689 | 0.7487 | 0.7483 | 0.8098 | 0.0368 | 0.0219 | 0.0574 |

These are descriptive point estimates. The release does not claim a statistically tested win.
See [claim_scope.md](claim_scope.md) for the supported and unsupported public claims.

## Canonical Public-Role Mapping

The source artifact column named `Role` is treated here as an artifact display label. The canonical public-role taxonomy is the separate `Canonical public role` column below.

| Artifact row label | Public name | Canonical public role | How to read it |
| --- | --- | --- | --- |
| selected final system | GPA-Net | protagonist | Selected under the bounded same-scope contract for the README headline row. |
| historical anchor | Historical anchor | anchor | Same repeated-walk family used to preserve continuity with earlier accepted evidence; it remains better on rearfoot RMSE. |
| diagnostic comparator | Diagnostic fusion variant | diagnostic/support | Useful classification-heavy signal under a different prediction-time fusion scope; it does not override the GPA-Net headline contract. |
| strict dual-attention baseline | Strict dual-attention baseline | support/context | Trial-level context row, not a repeated-walk direct replacement. |
| conservative MLP baseline | Conservative MLP baseline | support/context | Fixed-budget trial-level baseline; useful for context but not an architecture-superiority proof. |
| conservative CNN baseline | Conservative CNN baseline | support/context | Fixed-budget trial-level baseline; useful for context but not an architecture-superiority proof. |

See [method_code_map.md](method_code_map.md) for the single public-to-internal mapping table.

## Source Artifacts And Hashes

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Main result table | `python_pipeline/results/phase9_1/tables/main_results.csv` | `945ee5fce84f8725d0216e35e3afe991cef774f4f9895ad5cb71882d5c683d7f` |
| Dual-attention ablation table | `python_pipeline/results/phase9_1/tables/dual_attention_ablation.csv` | `c3ab9a2cb355a43db29f8a288d8483be73fae19fa106e52a8f2f5ba6598c2298` |
| Long-epoch held-out curve history | `python_pipeline/results/phase9_2/tables/long_epoch_heldout_loss_history.csv` | `260990bf2d4e6cdcad956071d1e76f017719ec7ae7b8e36697b5f129de267c73` |
| Long-epoch held-out curve summary | `python_pipeline/results/phase9_2/tables/long_epoch_heldout_loss_summary.csv` | `a2eaade2633d6b6f524de935a6a002c8df87e1851b28bd568423b11a29492341` |
| Phase-feature sensitivity summary | `python_pipeline/results/phase9_2/tables/phase_feature_sensitivity_summary.csv` | `cd197bd16f86c228a55b7cc441457c15a075cb161cf09e5aadcd5780544d1997` |
| Final selection decision | `python_pipeline/results/phase9_1/final_selection_decision.json` | `e9be5a06b9a57dfc0c9efabc3bb269d8cdcf960410432ca9d42c03c414438ced` |
| Release figure manifest | `assets/figures/figure_manifest.csv` | `2c56019bdf1a9f0a83a6b5fa02ef1e74d4429ce23cb25e91f136e8a927323e54` |

The figure manifest records release-owned JPG paths, source paths, source hashes, dimensions, and README eligibility.

## Fixed-Control README View

The README control view uses the selected tuned GPA-Net recipe against fixed-recipe context rows. GPA-Net is classification-best within this fixed-control view, but this is not a matched-search architecture proof.

| Method | Role | Scope | Score ↑ | Balanced accuracy ↑ | Macro F1 ↑ | AUROC ↑ | Total RMSE ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPA-Net | selected tuned system | subject-held-out repeated-walk | 0.7689 | 0.7487 | 0.7483 | 0.8098 | 0.0574 |
| Strict dual-attention control | fixed-recipe context | trial-level | 0.5345 | 0.5240 | 0.4353 | 0.6441 | 0.0775 |
| Conservative MLP control | fixed-recipe context | trial-level | 0.6841 | 0.6610 | 0.6203 | 0.7709 | 0.0679 |
| Conservative CNN control | fixed-recipe context | trial-level | 0.4599 | 0.5018 | 0.2700 | 0.6080 | 0.0800 |

## Ablation Family

The ablation family keeps the same data, split, seeds, folds, features, loss weights, and repeated-walk fusion while changing attention structure.

| Variant | First attention | Second attention | Score ↑ | AUROC ↑ | Upper RMSE ↓ | Lower RMSE ↓ | Total RMSE ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full dual-attention | true | true | 0.7689 | 0.8098 | 0.0368 | 0.0219 | 0.0574 |
| First-attention-only | true | false | 0.7798 | 0.8114 | 0.0368 | 0.0209 | 0.0588 |
| Second-attention-only | false | true | 0.7371 | 0.7727 | 0.0369 | 0.0215 | 0.0582 |
| No-attention control | false | false | 0.7687 | 0.7753 | 0.0379 | 0.0216 | 0.0599 |

The bounded reading is a tradeoff: full dual-attention keeps the strongest TotalAsymmetry RMSE in this family, while first-attention-only raises classification metrics and gives back TotalAsymmetry RMSE. This is not an uncaveated architecture superiority claim.

## Interpretability Summary

The interpretability figure is plausibility evidence only. It helps inspect whether target-level salience patterns are compatible with gait-phase reasoning across Right Initial Contact, First Double Support, Right Single Support, Left Initial Contact, Second Double Support, and Left Single Support.

The phase-feature sensitivity panel is predictive evidence from outer held-out perturbations. It replaces the earlier exploratory feature-space projection because that view did not provide a useful separation claim. The sensitivity map reports positive loss increases after median replacement of phase-specific feature families; it is not causal proof.

Attention evidence is not causal proof and does not show that a single phase causally explains any output.

## Figure Manifest References

The release-owned figure paths used by the README are:

- `assets/figures/gpa-net-architecture-overview.jpg`
- `assets/figures/overall-system-framework.jpg`
- `assets/figures/performance-summary.jpg`
- `assets/figures/heldout-training-curves.jpg`
- `assets/figures/fixed-control-comparison.jpg`
- `assets/figures/phase-feature-sensitivity.jpg`
- `assets/figures/interpretability-summary.jpg`

The stricter attention-ablation plot is available for method support at `assets/figures/dual-attention-ablation.jpg`. The gait-cycle schematic is available for method explanation at `assets/figures/gait-cycle-schematic.jpg`.
