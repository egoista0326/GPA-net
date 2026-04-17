# GPA-Net Data Contract

This release expects one canonical CSV file and does not redistribute the source
dataset. Raw data are not bundled, and neither `Processed*` folders, `dataset.zip`,
nor ambiguous processed datasets are included.

Dataset access pending permission verification.

License pending.

## Required Feature Columns

The model input uses 96 phase-structured vibration features plus one scalar side
channel:

- `feature_001` through `feature_096`: six gait phases with 16 vibration features per
  phase.
- `feature_097`: scalar auxiliary feature passed to the scalar branch.

The six event-based gait phases are:

1. Right Initial Contact
2. First Double Support
3. Right Single Support
4. Left Initial Contact
5. Second Double Support
6. Left Single Support

## Required Metadata Columns

The release loader requires these metadata columns:

- `subject_id`
- `trial_id`
- `trial_key`
- `condition_label`
- `abnormal_type`
- `source_file`
- `pressure_source_file`
- `gait_cycle_index`

## Required Pressure Columns

The target builder expects these pressure summary columns:

- `leftUpper_combined`
- `rightUpper_combined`
- `leftLower_combined`
- `rightLower_combined`
- `lefttotal`
- `righttotal`

## Target Outputs

`gpanet.targets.attach_targets` derives four supervised outputs:

- forefoot pressure-balance asymmetry (UpperAsymmetry)
- rearfoot pressure-balance asymmetry (LowerAsymmetry)
- entire-foot pressure-balance asymmetry (TotalAsymmetry)
- Classification

Target construction uses left/right regional pressure summaries. For each pressure
region, the asymmetry ratio is the absolute left/right difference divided by the
left/right sum. The three asymmetry outputs are continuous pressure-balance ratios.
`Classification` is derived from `condition_label`, where rows with an abnormal
condition label map to class 1 and other rows map to class 0.

## No-Bundled-Data Policy

Users must bring their own authorized CSV that matches this contract. This release
tree is intentionally allowlist-built and excludes raw/local datasets, ambiguous
processed outputs, `Processed*`, and `dataset.zip`.
