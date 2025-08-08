# Changelog

## 2025-08-07 (Unreleased)
### Added
- Capability registry (`capabilities.py`) reporting CUDA, Ollama, OpenRouter key, D: drive, optional modules.
- Bias & fitness core tests (`test_bias_fitness_core.py`).
- Protocol metadata accessor `get_protocol_metadata()` and unified `PROTOCOL_VERSION` constant.
- README sections for Capability Registry and Fitness Policy Clarification.

### Changed
- Fixed `get_protocol_text()` to return v2.6 protocol instead of undefined constant.
- Refactored Ollama call logic to restore HTTP fallback (previously unreachable) and preserve system prompt in CLI path.
- Removed duplicate enum and duplicate imports in `orchestration.py`.

### Fixed
- Missing imports (json / ollama guards) preventing orchestration runtime in limited environments.
- Unreachable code in `_call_ollama` due to early return.
- Potential NameError in protocol version reference.

### Planned (Deferred)
- Parameterization of strict perfection fitness policy.
- Extraction of large orchestration components into modular files.
- Replacement of placeholder fitness function with task-specific evaluator.
