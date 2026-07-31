# Releasing mlx-kfac

Releases are built once from an annotated, optionally signed version tag and
promoted through TestPyPI to PyPI with OpenID Connect trusted publishing. No
long-lived PyPI tokens are used.

## One-time repository configuration

1. Create protected GitHub environments named `testpypi` and `pypi`.
2. Require a reviewer for the `pypi` environment.
3. Configure a trusted publisher for each index with:
   - owner: `primateria`
   - repository: `mlx-kfac`
   - workflow: `release.yml`
   - environment: `testpypi` or `pypi`
4. Protect the default branch and require the CI workflow.

## Release checklist

1. Update `CHANGELOG.md` and the version in `pyproject.toml`.
2. Confirm CI passes on the supported Python/MLX matrix.
3. Build locally with `python -m build` and validate with
   `python scripts/check_dist.py dist`.
4. Create and push an annotated tag matching the package version. Sign it
   when the maintainer has a configured signing key:

   ```bash
   git tag -a vX.Y.Z -m "mlx-kfac X.Y.Z"
   git push origin vX.Y.Z
   ```

   With a configured signing key, replace `-a` with `-s`.

The release workflow verifies that the tag and package version match, builds
the sdist and wheel, checks their contents and metadata, publishes to
TestPyPI, installs the TestPyPI wheel for a smoke test, and only then requests
approval to publish the same artifacts to PyPI. It finally attaches those
artifacts to a GitHub release.

PyPI versions and files are immutable. If a release fails after publication,
increment the version; never retag or replace an uploaded artifact.
