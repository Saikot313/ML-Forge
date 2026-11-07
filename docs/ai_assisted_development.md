# AI-Assisted Development Workflow

An AI coding assistant such as Claude is a fast drafting and review aid. It does not guarantee
correctness. Code is trustworthy only after the developer has read it, understood it and verified it by
running tests and checks. This document describes how that boundary is kept.

## Status labels

| Label | Meaning |
|---|---|
| AI-suggested | Produced by the assistant; unreviewed |
| Reviewed | The developer read and understood every line |
| Validated | Reviewed, plus tests/static checks pass locally and behaviour was observed |

Only "Validated" code is merged. Commit messages may record which parts began as AI suggestions.

## Where Claude helps

- **Code generation:** scaffolding modules from a precise specification (interfaces, error behaviour,
  config keys).
- **Refactoring:** extracting duplicated logic, improving names, adding type hints.
- **Debugging assistance:** explaining a traceback, proposing hypotheses and minimal repro cases.
- **Test-case design:** edge cases (empty input, NaN, mismatched shapes, extreme values), property
  ideas, tolerances.
- **Code review:** a second reading for bugs, numerical issues, resource leaks and API consistency.
- **Documentation:** drafting docstrings and docs, which are then checked against the code.

## Verification workflow for AI-generated Python

1. **Specify first.** Give the interface, inputs, outputs, error behaviour and constraints (CPU-only
   default, Windows compatibility, no new dependencies).
2. **Read everything.** Do not paste code you cannot explain. Look for invented APIs, wrong argument
   names, version-specific behaviour and silent exception handling.
3. **Check against documentation.** Confirm uncertain library calls in the official docs for your
   installed versions.
4. **Run static checks.** Import the module, run a linter/type checker if configured.
5. **Write or review tests independently.** Tests drafted by the same assistant that wrote the code can
   share its misunderstanding; check each assertion against a hand calculation or a trusted reference
   (for example `torch.nn.functional.cross_entropy`, hand-computed metrics).
6. **Run the tests locally** and read failures rather than regenerating code until they pass.
7. **Probe edge cases manually:** NaN/Inf, empty batches, wrong shapes, missing files, CPU-only
   behaviour.
8. **Measure, don't trust, performance claims.** Any statement about speed or memory must come from
   `scripts/benchmark.py` on your hardware.
9. **Review the diff** before committing. Keep commits small so each can be reviewed.
10. **Record what you learned** when the assistant was wrong; this improves future prompts.

## Prompting practices

- State the environment (Python 3.11, Windows, PyTorch version, CPU-only possible).
- Ask for failure modes and tests alongside code.
- Ask the assistant to flag assumptions and uncertain API usage.
- Provide the error output and a minimal reproduction when debugging.
- Ask for a review of a specific function with specific concerns (numerical stability, resource use).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Plausible but wrong code | Hand-verified tests and reference implementations |
| Hallucinated functions/arguments | Run the code; check official docs |
| Overstated performance claims | Benchmarks only; no unmeasured claims in docs |
| Inconsistent names across files | Import and run tests after each change |
| Over-engineering | Review for unnecessary abstraction and unused code |
| Secrets or private data in prompts | Never paste credentials or confidential data |

## Applied to this project

The test suite checks several implementations against independent references: stable softmax and
cross-entropy against PyTorch, the loop matrix product against BLAS, the manual linear layer against
`nn.Linear`, gradient clipping against `torch.nn.utils.clip_grad_norm_`, and metrics against
hand-computed values. These checks are the evidence of correctness, not the fact that an assistant
helped write the code.