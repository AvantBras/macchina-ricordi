# Minimal repository protection

## Status: prepared, not active

The repository is intentionally still private. GitHub's current plan returns an upgrade-required
403 for branch protection and rulesets on this private repository. The files below are prepared
API payloads, not automatically applied configuration. No branch protection is claimed yet.

At preparation time, AvantBras was the only collaborator with write/admin access. Public visitors
cannot push or merge just because they can view or fork a repository. Keep future contributors
on the fork-and-pull-request path unless you deliberately trust them with write access.

Existing Actions defaults are already read-only, and workflows cannot approve pull requests.
Those settings were verified and left unchanged.

## Policy to activate

`.github/main-protection.json` protects `main` with:

- Pull requests instead of direct pushes, including for the administrator.
- Successful `Tests (Python 3.11)`, `Tests (Python 3.12)`, and
  `Lint, lockfiles and distributions` checks from the GitHub Actions app (ID 15368).
- An up-to-date branch before merge; new commits to main may require updating an older PR.
- No force-pushes and no deletion of main.
- Resolution of review conversations before merge.
- Zero mandatory approving reviews, so a sole maintainer can merge their own passing PR.

The MusicGen advisory is intentionally not a required check. No GPU, signed commits, linear
history, merge queue, or second maintainer is required. Branches are not locked against normal
PR merges. The owner can still deliberately edit the settings; this prevents accidents, not a
determined administrator.

`.github/fork-pr-approval.json` separately requires maintainer approval before workflows run for
**all external fork contributors**, including returning contributors. This is permission to run
CI, not approval to merge. Review workflow/script changes before approving execution; approval
is not a sandbox and does not prove submitted code is safe. This setting is available once the
repository is public; GitHub rejected it while this repository was private.

## Activation after publication (or an eligible plan upgrade)

Changing visibility remains a separate, explicit owner decision. Do not publish as a side effect
of applying this policy. Ideally activate protection immediately after making it public, before
granting write access or sharing widely. A private plan upgrade can enable main protection
without publication; the public fork-workflow setting is a separate step and may remain unavailable.

From the source checkout, with `gh` authenticated as the owner:

```bash
# Inspect first. If rules/protection already exist, reconcile them rather than blindly replacing them.
gh api repos/AvantBras/macchina-ricordi/branches/main/protection
gh api repos/AvantBras/macchina-ricordi/rulesets

# After verifying the policy and eligibility, apply main protection.
gh api --method PUT repos/AvantBras/macchina-ricordi/branches/main/protection \
  --input .github/main-protection.json

# Public repositories: require approval for external fork workflows.
gh api --method PUT repos/AvantBras/macchina-ricordi/actions/permissions/fork-pr-contributor-approval \
  --input .github/fork-pr-approval.json

# Read back the exact settings; a committed file or a successful CI run is not enforcement.
gh api repos/AvantBras/macchina-ricordi/branches/main/protection
gh api repos/AvantBras/macchina-ricordi/actions/permissions/fork-pr-contributor-approval
gh api repos/AvantBras/macchina-ricordi/actions/permissions/workflow
```

Confirm that administrator enforcement, pull requests, all three required check names, strict
status checks, and resolved conversations are enabled; force-pushes and deletions must be off.
The fork approval policy must be `all_external_contributors`. Workflow defaults should still be
`read` with `can_approve_pull_request_reviews: false`. If any call fails, report that protection
is incomplete rather than assuming the payload was applied. Update the status above only after
read-back proves activation. No destructive push/delete probe is needed to verify these settings.

Renaming a required CI job later also requires updating its protected-check name. The local
policy tests catch drift between the prepared list and the current ordinary CI job names.
