# Project documentation

This directory is for maintained software documentation: installation, usage, API behavior, architecture, and reproducible examples.

## Software entry points

- [Project overview and setup](../README.md)
- [Python package documentation](../src/python/README.md)
- [Workflow examples](../example/README.md)
- [Downstream examples](../example/downstream/README.md)

## Development records

Task plans, progress, and run-specific reports are maintained in [Linear](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f). Historical files formerly stored here are listed in the [migration index](https://linear.app/jiahaoh/document/historical-record-index-and-migration-provenance-881cc5edc52b), with their original paths, checksums, and preserved source snapshots. Historical results require evidence review before reuse.

Keep benchmark runners, reusable configurations, evaluation code, and small fixtures in Git. Store large or run-specific outputs outside the checkout and link their provenance from the corresponding issue. See [agent instructions](../AGENTS.md) for the handoff and artifact convention.
