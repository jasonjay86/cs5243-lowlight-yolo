# Contributing

This is a 3-person semester project. The workflow is light but consistent.

## Branching

- `main` is **protected** — no direct commits, no force-pushes.
- Each enhancement module gets its own feature branch:
  - `feature/clahe`            ← Krutin Patel
- `feature/snr-aware`        ← Jason Johnson
- `feature/yolo-detector`    ← Rich Zanni
- `feature/eval-harness`     ← unassigned (grab after detection contract is stable)

## Workflow

1. **Branch from `main`.** Use one of the feature branches above:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/<your-branch>
   ```
2. **Commit small, focused changes** with messages prefixed by your first name:
   ```
   krutin: implement LAB-CLAHE passthrough
   rich:   add YOLOv8 wrapper class
   jason:  vendor SNR-Aware weights loader
   ```
   This makes `git log --oneline | grep krutin:` immediately useful.
3. **Push your feature branch** to GitHub:
   ```bash
   git push origin feature/<your-branch>
   ```
4. **Open a Pull Request on GitHub:**
   - Go to https://github.com/jasonjay86/cs5243-lowlight-yolo/pulls
   - Click **"New pull request"**
   - **base:** `main` ← **compare:** `feature/<your-branch>`
   - Title: short imperative summary, e.g. *"Add CLAHE enhancement module"*
   - Description: 2–4 sentences on what changed and why. If your PR fixes
     or relates to an issue, reference it (`Closes #12`).
   - Click **"Create pull request"**
5. **Wait for review.** Branch protection requires **1 approval** before
   the merge button enables. **Jason Johnson** is the designated reviewer
   for this repo (he's the owner) — see "Review policy" below.
6. **Address review feedback** by pushing more commits to the same branch
   (`git push origin feature/<your-branch>`). Don't open a new PR; the
   existing one will pick up the new commits automatically.
7. **Merge.** Once approved, the reviewer (typically Jason) clicks
   **"Squash and merge"** or **"Merge pull request"**. Squash-merge is
   preferred for feature branches — it keeps `main` history linear and
   one-commit-per-feature. The PR branch is auto-deleted after merge.

## Review policy

Branch protection requires 1 approval, but **technically any user with
write access can approve a PR** — that's how GitHub's standard rule works.
There is no per-person approval lock without a CODEOWNERS file, which is
overkill for a 3-person class project.

**The social contract** (please follow this even if GitHub doesn't enforce it):

- **Jason Johnson** reviews and merges PRs for `feature/clahe`,
  `feature/snr-aware`, `feature/yolo-detector`, and `feature/eval-harness`.
- Krutin and Rich should **not approve each other's PRs** even though
  GitHub would let them. The intent is that one non-author review happens
  before merge — typically Jason's. If Jason is unavailable for >24 hours
  and a PR is blocking the team, the other reviewer can step in as a
  fallback, and they should note that in the PR comment.
- All three should feel free to **comment** on any PR — review discussion
  is encouraged. Only the **approval** ("Approve" button) is the
  gatekeeper action.

## Code style

- Python 3.10+ (conda env from `environment.yml`).
- Type hints on all new code.
- One Enhancement class per module under `enhancements/<method>/`.
- Every Enhancement must be `@register`-decorated so the eval harness can find it.

## The one rule that matters

**Implement `enhancements/base.py` and the first registered `Enhancement` BEFORE anyone
writes enhancement code.** Otherwise we'll spend the first week re-aligning interfaces
instead of doing science. The raw baseline (`enhancements/raw/`) is the cheapest way
to nail down the contract — write that first.

## Datasets and weights

- Datasets → under `data/` (gitignored). Don't commit.
- Pretrained weights → under `enhancements/<method>/weights/` (gitignored).
  Use Git LFS or external storage if a weight is small enough to share.
