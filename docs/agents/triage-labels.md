# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Size label

`bse` (bug or small enhancement) is optional and sits alongside the category and state labels. It marks throw-in work: a minor change that needs little testing, oversight or judgment, built and released on its own with no significant dependencies, or folded into a larger initiative's run. Apply it in triage when the change is small and self-contained (e.g. web-only, no snapshot shape change, no live writes). Leave it off anything that needs a spec, a grill, a schema bump or a multi-release rollout.
