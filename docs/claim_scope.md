# Claim Scope

This page defines what the release does and does not claim.

## Supported Claims

- GPA-Net is the public release name for the selected subject-held-out repeated-walk inference package. Earlier internal artifacts may refer to the same bounded package as the `subject-heldout repeated-walk inference package`.
- The headline row uses `subject_id` GroupKFold with prediction-time repeated-walk mean fusion.
- The README figures and tables trace back to on-disk artifacts through [evidence.md](evidence.md) and the release figure manifest.
- The dual-attention ablation supports a bounded tradeoff reading inside the fixed same-scope family: full dual-attention keeps the strongest TotalAsymmetry RMSE in that family, while some simpler variants can raise classification metrics.
- The README fixed-control view compares the selected GPA-Net recipe with fixed-recipe controls for compact context.
- The phase-feature sensitivity panel is an outer held-out perturbation summary over gait phases and feature families.
- Attention summaries are plausibility evidence that help inspect whether learned salience is biomechanically coherent across the six gait phases.

## Unsupported Claims

- GPA-Net is not a single-trial replacement.
- The release does not claim uncaveated diagnostic superiority for the diagnostic fusion variant.
- The release does not claim unqualified architecture superiority over the conservative MLP baseline, conservative CNN baseline, or every ablation variant.
- The fixed-control comparison is not a matched-search architecture proof.
- The phase-feature sensitivity panel is not a causal proof and does not prove that retraining without a feature family would produce the same ranking.
- The release does not claim causal attention, causal proof, or that one gait phase causally explains an output.
- The release does not claim robustness across floor construction, room transfer, sensor placement, or other acquisition shifts that were not established in the measured evidence.

## Reading Guide

- Use the README for the bounded public narrative.
- Use [method.md](method.md) for phase names, gait-cycle label mapping, and the public dual-attention description.
- Use [evidence.md](evidence.md) for detailed metrics, provenance, and role definitions.
- Use [method_code_map.md](method_code_map.md) when you need the single bridge from public names to internal experiment IDs.

## Policy Summary

GPA-Net is the bounded protagonist for this release. Diagnostic and support rows remain useful evidence, but they do not override the headline scope. Interpretability remains plausibility evidence, not causal proof.
