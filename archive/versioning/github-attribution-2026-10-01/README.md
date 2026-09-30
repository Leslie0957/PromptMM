# GitHub attribution repair — 2026-10-01

User Leslie0957 confirmed authorship and explicitly requested history rewriting. All254 original branch commits now use the account noreply email `130903403+Leslie0957@users.noreply.github.com` for author and committer. Display names, author/committer dates and timezones, messages, tree IDs and parent topology were preserved.

- [Exact old-to-new commit map](commit-map.json)
- [Preservation evidence and original references](preservation-audit.json)

Old research-source IDs remain historical evidence. Do not bulk-replace them in run declarations, manifests, completed audits or immutable snapshots. Use the map when resolving an old source ID against the rewritten branch. All original baseline tags stay on their original objects. A verified local `before.bundle` preserves all original references/history; its path, size and SHA256 are recorded in the audit. It does not back up ignored datasets, checkpoints or raw run outputs.

The repository-local commit email was corrected for future commits; GitHub account settings and global Git configuration were not modified. The final documentation trace adds new files and append-only log records after the254 content-identical rewritten commits. All five existing remote branches were rewritten in one atomic push with explicit expected-old-value force-with-lease. REST API checks of all254 rewritten main commits show both author and committer account Leslie0957. These verified receipts are in the preservation audit. The final documentation trace is published afterward as a normal fast-forward; its SHA is reported in the task handoff. No experiment or data/Test/probe/sealed access occurred.
