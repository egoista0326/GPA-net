# Reproduction

This release is designed for a clean clone. Raw data are not bundled, so commands that execute training or evaluation require an authorized canonical CSV matching [data_contract.md](data_contract.md).

## Environment

```bash
conda env create -f environment.yml
conda activate gpa-net
```

The environment file pins the measured stack used by the release validation surface. Do not replace the setup with machine-local interpreter paths.

## Smoke Checks

```bash
python scripts/reproduce_gpa_net.py --help
python scripts/reproduce_gpa_net.py --list-recipes
python scripts/validate_release.py
```

## Data Input

Default data path:

```bash
data/canonical_dataset.csv
```

Custom release-relative path:

```bash
python scripts/reproduce_gpa_net.py --recipe gpa_net_final --data data/canonical_dataset.csv
```

Use `--check-data` to load the CSV through the release data loader before printing a recipe:

```bash
python scripts/reproduce_gpa_net.py --recipe gpa_net_final --data data/canonical_dataset.csv --check-data
```

## Recipe Policy

The wrapper prints release-local reproduction entrypoints from `docs/source_reproduce_commands.yaml`. Some recipes summarize historical evidence paths rather than rerunning every upstream research workflow inside this release subtree.

## Validation

Run the public clean-clone checks from the repository root:

```bash
python scripts/validate_release.py
pytest tests/test_package_contract.py -q
git diff --check
```

These commands validate the shipped release tree, public docs, and release wrapper metadata without requiring the parent monorepo.

## Maintainer Provenance Audit

If you also have the canonical workspace that contains the pipeline outputs and original figure-source files, you can opt into stricter figure-manifest provenance checks from the repository root:

```bash
python scripts/validate_release.py --check-source-provenance --project-root ..
```

That audit re-hashes each manifest `source_path` against the canonical workspace and is not part of the default clean-clone path.

## Policy Status

License pending.

Dataset access pending permission verification.
