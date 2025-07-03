# Contributing Guidelines

Thank you for considering contributing to this project!

## What You Can Contribute

We welcome the following types of contributions:

- **Code**: New models, features, refactoring, or bug fixes.
- **Tests**: Unit and integration tests to ensure pipeline stability.
- **Documentation**: Improvements to the README, tutorials, or code comments.
- **Visualizations**: Enhancements to EDA or output graphs.
- **Ideas**: Feature suggestions, use case proposals, or performance insights.


## Branch Policy
- Use `main` for stable releases.
- Use `17-development_branch` for the latest ongoing changes.
- Branch off from `17-development_branch` using the format:
  - `feature/<short-description>`
  - `bugfix/<issue-description>`
  - `docs/<doc-area>`

## Workflow
1. Fork the repository and clone your fork locally.
2. Create a feature branch.
3. Commit your changes with meaningful messages.
4. Open a Merge Request (MR) to `main`.

## Code Style
- Use `black` and `isort` for formatting.
- Follow PEP8 conventions.
- Ensure your code passes all `pytest` tests before committing.
- Avoid unused imports, commented code blocks, and large merge commits.

## Licensing
All contributions must comply with the REUSE licensing scheme.
