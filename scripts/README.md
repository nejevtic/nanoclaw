# scripts/

Utility and maintenance scripts for NanoClaw.

| Script | How it runs | Purpose |
|--------|-------------|---------|
| `refresh-oauth-token.mjs` | systemd timer (`nanoclaw-token-refresh.timer`, every 30 min) | Reads the refresh token from `~/.claude/.credentials.json`, fetches a new access token from the Anthropic OAuth endpoint, and syncs it back into `.env` as `CLAUDE_CODE_OAUTH_TOKEN`. |
| `apply-skill.ts` | `npm run apply-skill` | Apply a contributed skill to your fork. |
| `post-update.ts` | Called by `/update` skill after merge | Runs post-merge fixups (migrations, dependency sync). |
| `rebase.ts` | Called by `/update` skill | Rebases your customizations on top of upstream changes. |
| `run-migrations.ts` | `npm run migrate` | Applies pending SQLite schema migrations. |
| `update-core.ts` | Called by `/update` skill | Fetches and merges upstream nanoclaw changes. |
| `generate-ci-matrix.ts` | CI | Generates the test matrix for GitHub Actions. |
| `generate-resolutions.ts` | CI | Generates package resolution overrides. |
| `run-ci-tests.ts` | CI | Orchestrates CI test runs. |
| `uninstall-skill.ts` | `npm run uninstall-skill` | Remove a previously applied skill. |
