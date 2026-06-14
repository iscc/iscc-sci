---
name: release
description: Cut and publish a new release of iscc-sci — run preflight checks, finalize the version and CHANGELOG, create the GitHub Release, and confirm the CI-driven PyPI publish. Use when asked to "release", "publish a new version", "cut a release", "ship vX.Y.Z", or "bump and release".
---

# Release iscc-sci

Drive a complete, safe release of `iscc-sci` to PyPI.

## How releasing works here

The version in `pyproject.toml` is the single source of truth. Publishing is automated by
`.github/workflows/release.yml`, which triggers when a **GitHub Release is published**:

1. Re-runs the full test matrix at the released ref (reuses `tests.yml`).
1. Guards that `uv version --short` equals the release tag (minus the `v`), aborting on mismatch.
1. `uv build` → `uv publish` to PyPI using the org-wide `PYPI_TOKEN` secret.

So a release = bump version + finalize CHANGELOG + push + `gh release create vX.Y.Z`. The rest is
CI.

> [!CAUTION]
> **PyPI uploads are irreversible.** A version can never be re-uploaded or edited — only yanked.
> Always confirm the target version with the user before cutting the release (step 6).

## Argument

Optional: a bump level (`patch` | `minor` | `major`) or an explicit version (`X.Y.Z`).

- No argument → release the version currently in `pyproject.toml` as-is.
- A bump level → run `uv version --bump <level>` first.
- An explicit `X.Y.Z` → run `uv version <X.Y.Z>` first.

## Procedure

Run steps in order. **Abort and report** if any precondition fails — never push past a failure. All
commands are cross-platform (bash shell; `git`/`gh`/`uv`/`curl` exist on Windows, macOS, Linux).

### 1. Preflight

- `gh auth status` — must be authenticated.
- `git rev-parse --abbrev-ref HEAD` — must be `main`. If not, stop and ask the user.
- `git fetch origin` then verify local `main` is level with `origin/main`
  (`git rev-list --left-right --count main...origin/main` → `0	0`). No unpushed/unpulled commits.
- `git status --porcelain` — must be empty (clean working tree).

### 2. Determine the target version

- Read current: `uv version --short`.
- Apply the argument if given:
  - bump level → `uv version --bump <level>`
  - explicit version → `uv version <X.Y.Z>`
  - (preview first with `--dry-run` if unsure)
- Capture the resulting version as `V` (`uv version --short`).
- **Guards** (stop if any hit):
  - Tag must not exist: `git tag --list "v$V"` empty **and** `git ls-remote --tags origin "v$V"`
    empty.
  - Not already on PyPI: `curl -sf "https://pypi.org/pypi/iscc-sci/$V/json"` must return non-zero
    (404). If it returns 200, that version is already published — pick a higher version.

### 3. Finalize the CHANGELOG

- `CHANGELOG.md` follows Keep-a-Changelog. Ensure the top has a dated section for this release:
  `## [V] - YYYY-MM-DD` using today's date (`date +%F`), with the human-written bullet list. If
  items live under an `## Unreleased` heading, rename it to `## [V] - <today>`.
- Re-run the markdown gate so formatting matches: `uv run poe format-md`.

### 4. Local quality gates (catch failures before tagging)

- `uv run prek run --all-files`
- `uv run poe test` (enforces 100% coverage; downloads the ONNX model on first run).
- If either fails, **stop and fix** before continuing — a failed gate here would fail CI too.

### 5. Commit & push the release prep

- Only if `pyproject.toml`, `uv.lock`, or `CHANGELOG.md` changed:
  - `git commit -am "chore: release v$V"`
  - Confirm with the user, then `git push origin main`.

### 6. Cut the GitHub Release (this triggers publishing)

- **Confirm the target `v$V` with the user** — this is the irreversible step.
- Extract the `## [V]` section body from `CHANGELOG.md` and write it to a temp notes file (e.g.
  `release-notes.md`), then: `gh release create "v$V" --title "v$V" --notes-file release-notes.md`
- This tags `v$V` at current `main` HEAD and fires `release.yml`.

### 7. Monitor CI

- Watch the run: `gh run list --workflow=Release --limit 1 --json status,conclusion,databaseId,url`,
  then `gh run watch <databaseId>` (or poll the list).
- Report the conclusion and the run URL. If it failed, go to **Recovery**.

### 8. Verify publication

- Poll PyPI until live (can lag ~1 min): `curl -sf "https://pypi.org/pypi/iscc-sci/$V/json"` returns
  200\.
- `gh release view "v$V"` to confirm the release looks right.
- Report success with the link: `https://pypi.org/project/iscc-sci/$V/`.

## Recovery

- **CI failed before publish (tests or version guard):** nothing was published, so it's safe to
  redo. Delete the release and tag, fix, then re-cut from step 5/6:
  `gh release delete "v$V" --yes --cleanup-tag` and `git push --delete origin "v$V"` if the tag
  lingers.
- **Version guard failed (tag ≠ pyproject):** set `pyproject.toml` to `V` (or re-tag), push, re-cut.
- **Transient CI failure (e.g. network) on a correct tag:** `gh run rerun <databaseId>`.
- **Publish rejected because the version already exists on PyPI:** versions are immutable — bump to
  the next patch and release again. To hide a broken release from new installs, **yank** it in the
  PyPI web UI (manual; yank does not delete).

## Notes

- Never edit or reuse a published version — always move the version forward.
- The CI version guard (`uv version --short` == tag) is the backstop against mismatched publishes;
  keeping the bump (step 2) and the tag (step 6) consistent makes it a no-op.
- Requires the repo/org secret `PYPI_TOKEN` (already available org-wide).
