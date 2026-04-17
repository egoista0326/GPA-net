# GPA-Net Data Access

Raw data are not bundled in this release. The release does not ship raw vibration files,
raw pressure files, `Processed*` folders, `dataset.zip`, or ambiguous processed
datasets.

Dataset access pending permission verification.

License pending.

To run GPA-Net code from a clean clone, provide a canonical CSV at
`data/canonical_dataset.csv` or pass another release-relative CSV path to
`python scripts/reproduce_gpa_net.py --data <path>`. The CSV must follow the schema in
`docs/data_contract.md`.

Do not copy local raw-data folders, processed workspace outputs, or permission-unclear
dataset archives into this directory.
