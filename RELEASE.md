# Releasing SPINNME

The package and runtime versions are `0.1.0`; the release tag is `v0.1.0`.
The repository contains the PyPI workflow, Read the Docs configuration and
release notes in [CHANGELOG.md](CHANGELOG.md).

## One-time account setup

1. On PyPI, configure a pending Trusted Publisher for project `spinnme`:
   owner `RobertJaro`, repository `spinnme`, workflow `release.yml`,
   environment `pypi`. If the project already exists, add that publisher to it.
2. In the GitHub repository, create the `pypi` environment. Restrict deployment
   to release tags and configure any desired release reviewers.
3. Import `RobertJaro/spinnme` into Read the Docs with project slug `spinnme`.
   Use `.readthedocs.yaml`, enable the GitHub webhook and activate `v0.1.0`
   when the tag exists. Set `stable` to the release version and verify
   https://spinnme.readthedocs.io/en/stable/.

The PyPI publisher uses GitHub's identity token; no PyPI API token is stored in
this repository. Instructions: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [Read the Docs setup](https://docs.readthedocs.com/platform/stable/tutorial/index.html).

## Validate and publish

From a development environment with `.[dev,download,synthetic,docs]` installed:

```bash
python -m ruff check src tests
python -m pytest -q
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check --strict dist/*
```

Review the wheel and source distribution contents and test the installed wheel
outside the checkout with the documented circular example. Confirm that
`pyproject.toml`, `spinnme.__version__`, `CITATION.cff` and the changelog agree.

After committing, pushing and passing CI, create the annotated release tag:

```bash
git tag -a v0.1.0 -m "SPINNME v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --verify-tag --draft --title "SPINNME v0.1.0" --notes-file CHANGELOG.md
```

Review and publish the draft GitHub release. Publishing triggers `release.yml`,
which runs CI, checks the tag against package versions, builds and validates the
wheel and source distribution, publishes them to PyPI and attaches both files
plus SHA-256 checksums to the GitHub release. Draft creation alone does not publish
anything to PyPI. Once a version has been uploaded to PyPI, use a new version for
subsequent changes.

Verify installation from PyPI in a fresh Python environment and check the stable
Read the Docs build and release assets. Update this checklist's version for the
next release.
