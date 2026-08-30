---
description: Make the repository legally usable and discoverable, then take it public.
allowed-tools: Bash(gh repo view:*), Bash(gh repo edit:*), Bash(git status:*), Read, Write, Edit
---

# Make it reachable

Nothing else in this repository is observable until this is done. The work is
finished; the distribution has not started.

## Why this is first

A public repository with no LICENSE is not "open by default" — it is
all-rights-reserved by default. Someone who finds this, likes it, and wants to
adapt a notebook for a class legally cannot. The project's whole purpose is to be
read and reused, so the missing file negates the purpose.

## Do

1. Confirm the current state rather than assuming it:
   `gh repo view --json isPrivate,licenseInfo,description,repositoryTopics`
2. Add `LICENSE` — **Apache-2.0**. Not MIT: Apache's explicit patent grant matches
   what Qiskit and PennyLane themselves ship under, which matters for anything
   people may build on. Set the copyright line to the author, current year.
3. Add the author to `pyproject.toml` `[project]` as `license = "Apache-2.0"`, and
   confirm `authors` is right.
4. Set the repository description. Lead with the differentiator, not the topic —
   the honest-comparison stance is what makes this findable, not the word
   "quantum":
   > Runnable quantum algorithm tutorials, each measured against a classical
   > baseline and reported honestly — including when the classical method wins.
5. Set topics: `quantum-computing`, `qiskit`, `qaoa`, `vqe`,
   `quantum-machine-learning`, `benchmarking`, `pennylane`, `tutorial`.
6. Add a CI badge to the top of `README.md`.
7. Flip to public **last**, once the above is committed.

## Done when

- `gh repo view` reports a license, a description, topics, and `isPrivate: false`.
- `LICENSE` exists and `pyproject.toml` agrees with it.
- The README's first screen states what the project is and shows CI passing.

## Do not

Do not draft a CONTRIBUTING or CODE_OF_CONDUCT in the same pass. They are cheap to
add and cost nothing to defer; the license is the only blocking item here.
