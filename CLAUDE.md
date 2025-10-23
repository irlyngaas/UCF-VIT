# UCF-VIT Development Principles

## Core Principles (MUST FOLLOW)

### 1. Small but Certain Steps
- Take ONE small, certain step at a time
- NO rushing or trying to do everything at once
- Verify and test each step before moving to the next
- Build the software stack incrementally with confidence

### 2. Minimal Code Changes
- This codebase is shared by multiple team members
- Focus ONLY on implementing required features
- Avoid unnecessary additions or "nice-to-have" features
- Keep modifications to the absolute minimum

## Project Context
- Working on Vision Transformer with quantization support
- Target: Frontier supercomputer with AMD MI250X GPUs
- Focus: 8-bit quantization using torch.ao and quanto