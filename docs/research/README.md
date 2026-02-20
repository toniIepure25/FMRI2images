# Research Workspace

This directory is the **shared workspace** for the multi-agent research workflow. Four specialist agents collaborate here by reading and writing structured Markdown documents.

## Directory Structure

```
docs/research/
├── README.md              # This file
├── neuro/                 # Neuroscience researcher outputs
│   └── (neuro-review-YYYY-MM-DD.md files)
├── ml/                    # ML researcher outputs
│   └── (ml-review-YYYY-MM-DD.md files)
└── decisions/             # Architecture Decision Records
    └── (ADR-NNN-title.md files)
```

## Agents

| Agent                      | Rule File                                  | Activation Phrase                  |
|----------------------------|--------------------------------------------|------------------------------------|
| Neuroscience Researcher    | `.cursor/rules/neuroscience-researcher.mdc`| "Act as the neuroscience researcher" |
| Machine Learning Researcher| `.cursor/rules/ml-researcher.mdc`          | "Act as the ML researcher"         |
| Implementer                | `.cursor/rules/implementer.mdc`            | "Act as the implementer"           |
| Tester / Validator         | `.cursor/rules/tester-validator.mdc`       | "Act as the tester"                |

Supporting rules (always active or file-specific):
- `.cursor/rules/project-conventions.mdc` — always-on project context
- `.cursor/rules/python-research.mdc` — Python code standards (`**/*.py`)
- `.cursor/rules/config-conventions.mdc` — YAML config standards (`configs/**/*.yaml`)
- `.cursor/rules/paper-writing.mdc` — paper/documentation standards (`docs/paper/**/*.md`)

## Workflow

### Phase 1: Discovery (Neuroscience Researcher)

Open a new chat and say:

> "Act as the neuroscience researcher. Review our ROI-Transformer design and tell me if the attention-to-neuroscience story is scientifically sound."

The agent will:
- Apply its neuroscience domain knowledge
- Write findings to `docs/research/neuro/neuro-review-YYYY-MM-DD.md`
- Propose neuroscience-grounded hypotheses and recommended experiments

### Phase 2: Design (ML Researcher)

Open a new chat and say:

> "Act as the ML researcher. Read the neuro review at docs/research/neuro/neuro-review-YYYY-MM-DD.md and propose an architecture that addresses the concerns."

The agent will:
- Read the neuroscience review
- Verify mathematical correctness of current approaches
- Propose principled improvements with novelty assessment
- Write proposals to `docs/research/ml/ml-review-YYYY-MM-DD.md`

### Phase 3: Implementation (Implementer)

Open a new chat and say:

> "Act as the implementer. Read the ML proposal at docs/research/ml/ml-review-YYYY-MM-DD.md and implement it."

The agent will:
- Read the proposal and study existing code patterns
- Write clean, typed, documented code
- Create config entries for new features
- Follow all coding standards

### Phase 4: Validation (Tester)

Open a new chat and say:

> "Act as the tester. Validate the new implementation — write tests and check correctness."

The agent will:
- Write comprehensive pytest tests
- Validate mathematical properties
- Check numerical stability
- Produce a validation report

## Architecture Decision Records (ADRs)

For significant design choices, create a decision record in `docs/research/decisions/`:

```markdown
# ADR-NNN: [Decision Title]
Date: YYYY-MM-DD
Status: Proposed / Accepted / Superseded

## Context
[Why this decision is needed]

## Options Considered
1. [Option A] — pros / cons
2. [Option B] — pros / cons

## Decision
[What we chose and why]

## Consequences
[What changes as a result]
```

## Tips for Effective Use

1. **Be specific** in your prompts — reference exact files and point to specific concerns
2. **Chain the agents** — the neuroscience researcher identifies the problem, the ML researcher designs the solution, the implementer builds it, the tester validates it
3. **Use decisions/ for controversial choices** — document why vMF over Power Spherical, why bounded-sigmoid over softplus, etc.
4. **Review across agents** — ask the neuroscience researcher to review ML proposals for neuroscience soundness, and vice versa
5. **Keep documents dated** — this creates a research trail showing how ideas evolved
