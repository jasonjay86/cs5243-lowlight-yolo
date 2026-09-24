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
5. **Wait for review.** Branch protection requires **1 approval** from a
   non-author teammate before the merge button enables. See
   "Review policy" below for how peer review works in this repo.
6. **Address review feedback** by pushing more commits to the same branch
   (`git push origin feature/<your-branch>`). Don't open a new PR; the
   existing one will pick up the new commits automatically.
7. **Merge.** Once approved, **the PR author merges** (not the reviewer).
   This keeps the reviewer in the "reviewer" role and avoids the
   awkwardness of someone else merging your work. Use **"Squash and merge"**
   to keep `main` history linear and one-commit-per-feature. The PR
   branch is auto-deleted after merge.

## Review policy

We review each other's work. **Any non-author teammate can approve a PR.**
Branch protection enforces that 1 approval is required before merge, and
GitHub prevents the PR author from approving their own PR — so the model
is naturally peer-review by default:

- **Krutin** opens `feature/clahe` PR → Jason or Rich reviews and approves
- **Jason** opens `feature/snr-aware` PR → Krutin or Rich reviews and approves
- **Rich** opens `feature/yolo-detector` PR → Krutin or Jason reviews and approves

There is no "lead reviewer" or hierarchy. Whoever has time and context
approves. **Reviewers should:**

- Actually read the diff (not rubber-stamp).
- Comment on anything questionable — code style, missing tests, unclear
  naming, scope creep. Comments are encouraged; only the explicit
  **"Approve"** click is the gatekeeper.
- Be specific. "Looks good" is less useful than "the contract in
  `base.py` is satisfied, but the `flops_per_megapixel()` returns `None`
  instead of a real number — please measure and fill it in before merge."

**The PR author merges** after approval (not the reviewer). This separates
the "review" role from the "merge" role cleanly.

**Admin bypass (for emergencies):**

`jasonjay86` is configured as a [bypass actor on the `main` branch
protection rule](https://github.com/jasonjay86/cs5243-lowlight-yolo/settings/branches),
so the repo owner can merge without an external approval. **This is a
backdoor by design** — it exists so Jason can land urgent hotfixes when
the team is unavailable. The expectation is:

- Use it sparingly. The default path is still: branch → PR → wait for
  peer review → author merges.
- When you use it, **say so in the PR description** ("Self-merging: <reason>").
  Transparency matters more than the bypass itself.
- Krutin and Rich can ask Jason at any time to remove the bypass if it
  gets misused. The setting is in one click and reversible.

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
