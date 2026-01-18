# AGENTS.md

This repository is compatible with Codex-style agents. Use this document as the operating guide for any AI agent contributing to the codebase.

## Project: AI Video Analytics System (CCTV AI)
A production-grade system for multi-stream RTSP ingestion, real-time video analytics (people counting, intrusion/safety events), stable event logic, and alerting via API/webhooks. Targets GPU servers and NVIDIA Jetson.

---

## 1) How to Work in This Repo

### Primary goals
- Keep the pipeline **stable, deterministic, and production-friendly**.
- Prefer **config-driven** behavior over hardcoded values.
- Optimize for **reliability** (reconnects, timeouts, backpressure) before adding features.

### Non-goals
- No “demo-only” hacks in main code paths.
- No large model files committed to git.
- Avoid UI-heavy work unless explicitly requested.

---

## 2) Repository Structure (expected)
- `src/`
  - `ingestion/` RTSP/GStreamer capture, reconnect, frame batching
  - `inference/` model loading, preprocessing, inference adapters (YOLO/ONNX/TRT/DeepStream)
  - `events/` stable detection logic (debounce, cooldown, zone rules)
  - `alerts/` webhooks, email/SMS/WhatsApp adapters (pluggable)
  - `api/` FastAPI endpoints (optional)
  - `utils/` logging, config parsing, timers, common helpers
- `configs/` YAML/JSON configurations (cameras, zones, rules)
- `docs/` project docs + gig showcase
- `tests/` unit tests
- `scripts/` utility scripts

If the repo differs, follow existing conventions.

---

## 3) Python Runtime & Virtual Environment
This project uses a **Pipenv-managed virtual environment**.  
All agents must use the Python interpreter from the active Pipenv environment.

### Active Python Interpreter
`/home/hamza/.local/share/virtualenvs/ai-video-analytics-system-OAjovT-z/bin/python3`

### How to Activate the Environment
```bash
pipenv shell
```
or explicitly:
```bash
source /home/hamza/.local/share/virtualenvs/ai-video-analytics-system-OAjovT-z/bin/activate
```

### Verification
```bash
which python3
```

⚠️ PyTorch CUDA wheels are installed using pip inside the Pipenv environment:
```bash
pipenv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Agents must not use system Python or a different virtual environment when running, testing, or modifying this project.

---

## 4) Coding Standards

### Style
- Python: PEP8, type hints where it improves clarity.
- Use clear names, short functions, and good comments for tricky logic.
- Avoid global state; prefer dependency injection (pass in config/clients).

### Logging
- Use structured logs where possible.
- Log key lifecycle events:
  - camera connect/disconnect, reconnect attempts
  - inference errors + recovery
  - event triggers (with camera_id, event_type, confidence)
- Never log secrets (tokens, credentials).

### Config-first design
All thresholds, zones, cooldowns, model paths, camera RTSP URLs, etc. must live in config files and be referenced via config parsing utilities.

---

## 5) Testing Expectations
Add tests for:
- event debounce/cooldown logic
- zone membership checks
- alert routing (mock network calls)
- config parsing validation

Unit tests should not require a GPU.

---

## 6) Performance / Reliability Guidelines
- Prefer **bounded queues** to avoid memory bloat.
- Implement backpressure strategies (drop frames vs. slow consumers) as config options.
- Reconnect logic must use exponential backoff and avoid tight loops.
- Keep critical loops lightweight; avoid per-frame heavy allocations.

---

## 7) Security & Privacy
- Treat video as sensitive data.
- Do not store raw footage unless explicitly required.
- Store only event metadata + optional snapshots (config-controlled).
- Ensure webhook endpoints are configurable and support auth headers (if needed).

---

## 8) PR / Change Checklist (Agent)
Before submitting changes:
- [ ] Code builds/runs locally (or at least imports cleanly)
- [ ] Added/updated unit tests where applicable
- [ ] Updated docs if behavior changed
- [ ] No secrets or large binaries committed
- [ ] Config schema remains backward-compatible (or versioned)

---

## 9) What to Do When Requirements Are Ambiguous
When specs are unclear:
1. Infer the most conservative, production-safe interpretation.
2. Document assumptions in the PR description / comments.
3. Keep changes modular and easy to revise.

---

## 10) Communication Conventions
When proposing work, use:
- **Problem**
- **Proposed Change**
- **Impact / Risks**
- **Test Plan**

Keep messages short and engineering-focused.

---

## 11) Quick Glossary
- **Stable detection**: event triggers only after N consistent frames / seconds.
- **Cooldown**: once triggered, ignore repeats for X seconds.
- **Zone rule**: event triggers only if object center (or box overlap) is inside a polygon ROI.

---

End of AGENTS.md
