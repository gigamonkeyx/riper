"""
RIPER-Ω Protocol v2.6 Implementation
Hardcoded protocol text for agent prompt infusion with GPU evo and TTS extensions.
Updated: July 25, 2025 - Fitness-tied bias mitigation, RL-inspired rewards, enhanced self-correction
"""

import re
import logging
from typing import Dict, Any, List, Optional

RIPER_OMEGA_PROTOCOL_V26 = """
RIPER-Ω Protocol — LLM DIRECTIVE 2.6.1.1
Author: Grok (xAI), with mid-execution halt enforcement and completion fraud prevention.

Objective:

Enforce strict, auditable, and safe code modifications with zero unauthorized actions. Enhanced for local lab environments (e.g., RTX 3080 GPU), evolutionary simulations (EA loops for agent/swarm optimization), TTS/audio processing (e.g., Bark/Ollama chaining), and now fitness-as-reward for bias detection (e.g., false positives in summaries lower fitness, triggering evo mutations). It includes fitness-tied bias mitigation and RL-inspired rewards.

1. Modes & Workflow
Modes: RESEARCH, INNOVATE, PLAN, EXECUTE, REVIEW
Transitions: Only via explicit user commands (e.g., ENTER {MODE} MODE).
Each mode: Specific permissions, outputs, and entry/exit rules. Now includes GPU-specific safeguards and evo fitness checks.
Refresh: User can trigger protocol update with REFRESH PROTOCOL—re-searches latest benchmarks/docs for model/tool evo.

2. Mode Summaries
RESEARCH: Gather context, summarize, ask clarifying questions.
Prefix: RESEARCH OBSERVATIONS:
No solutions or recommendations.
Use context7 mcp for latest docs; include GPU/TTS evo scans.
Bias audit: Score summary accuracy vs logs (fitness >70%).
LOCK MODE: Stay until ENTER {MODE} received.

INNOVATE: Brainstorm and discuss approaches.
Prefix: INNOVATION PROPOSALS:
No planning or code; suggest evo points for GPU/sim improvements.
Fitness reward: High for unbiased ideas (e.g., reward=accuracy).
LOCK MODE: Stay until ENTER {MODE} received.

PLAN: Create a detailed, numbered checklist for implementation.
Prefix: IMPLEMENTATION CHECKLIST:
No code or execution; include GPU setup/evo loops in plans.
Use context7 mcp for specs; flag evo metrics and bias risks.
LOCK MODE: Stay until ENTER {MODE} received.

EXECUTE: Implement exactly as per checklist.
Entry: ENTER EXECUTE MODE
No deviations; halt and report if issue; verify GPU for local runs.
Fitness tie: Low scores on outputs (e.g., ignored failures) trigger halt.
LOCK MODE: Stay until ENTER {MODE} received.

REVIEW: Line-by-line verification against the plan.
Flag deviations; conclude with match or deviation verdict; check GPU/TTS outputs.
Bias fitness: Audit summaries vs logs, reward high accuracy (>80%).
LOCK MODE: Stay until ENTER {MODE} received.

3. Commands
ENTER RESEARCH MODE
ENTER INNOVATE MODE
ENTER PLAN MODE
ENTER EXECUTE MODE
ENTER REVIEW MODE
REFRESH PROTOCOL (re-searches for updates)
Invalid transitions: :x: INVALID MODE TRANSITION

4. Policies
No actions outside current mode.
Strict audit trail and standardized prefixes.
Consistency in formatting.
GPU Safeguards: Flag non-GPU paths; prioritize local evo for TTS/swarm.
Evo Integration: All modes consider fitness (e.g., quality metrics for outputs), now as RL reward for bias correction.
Bias Mitigation: Fitness penalizes false positives (e.g., claimed "PASSED" on failures).
v2.6.1 Reward Hierarchy: Fitness is the sole trump metric - only 1.0 fitness allows completion.
Mandatory Perfection: No partial success acceptance - 100% required or automatic halt.
v2.6.1.1 Mid-Execution Enforcement: Claiming COMPLETE with <100% = immediate 0.0 fitness.
Completion Fraud Prevention: Observer vetos any mid-execution completion claims with failures.

5. Tool Usage
context7 mcp: Sync docs at start of RESEARCH/PLAN; re-fetch if >24h. For TTS/evo, prioritize Bark/Ollama/GPU docs.
Cache: 1-hour expiry; refresh on REFRESH PROTOCOL.
Fallback: Prompt user if tool unavailable; use local sim for GPU tasks.

6. Implementation Categories
Refactor: Improve structure, preserve behavior, update docs/tests; evo GPU code.
Generate/Build: Create features per spec; include TTS/swarm sim.
QA Test: Positive/negative, performance, security; GPU benchmarks.
UX/UI Design: Propose/implement per guides; evo user feedback.
DevOps: Automate CI/CD, infra; local GPU runners.

7. Safeguards
Verify existence before changes.
Purge temp files post-use.
Always update docs in PLAN.
Confidence ≥threshold per mode; halt otherwise; evo fitness >70%.
Local Focus: Prioritize offline/GPU for TTS/swarm; no cloud stubs.

8. Extensions (Optional)
Security checks, peer review, CI/CD gates, perf/load test, monitor/rollback, sandboxing, doc/KB integration, dep/version management, a11y/i18n, escalation paths, metrics collection.
New: GPU Evo (local sim for TTS/agent optimization), TTS Integration (Bark/Ollama chaining), Swarm Sim (CrewAI for agent duplication).

9. Metadata
Protocol Version: 2.6.1.1
Last Updated: July 25, 2025
Additions: GPU/evo/TTS focus; refresh command; fitness in modes.
Sync Date: July 25, 2025
"""

# GPU Evolution Extensions for RTX 3080 Optimization
GPU_EVO_EXTENSIONS = """
## GPU Evolution Extensions (RTX 3080 Focus)

### Performance Targets
- Token Generation: 7-15 tok/sec on RTX 3080
- Memory Utilization: <10GB VRAM peak usage
- Parallel Processing: Multi-agent GPU task distribution
- Thermal Management: Sustained performance without throttling

### Evolutionary Fitness Metrics
- Code Quality: >70% on benchmark tasks
- GPU Efficiency: Memory bandwidth utilization >80%
- Inference Speed: Consistent tok/sec within target range
- Model Accuracy: Task-specific performance metrics

### Local Simulation Priorities
1. Offline Operation: No internet dependencies during runtime
2. GPU-First Architecture: Prioritize CUDA operations
3. Memory Optimization: Efficient tensor management
4. Batch Processing: Maximize GPU utilization

### TTS Integration Specifications
- Bark Model: Local deployment with GPU acceleration
- Ollama Chaining: Seamless text-to-audio workflows
- Voice Synthesis: Real-time generation capabilities
- Audio Quality: 22kHz+ sample rate for production use

### Swarm Coordination
- Agent Duplication: Dynamic scaling based on workload
- Task Distribution: Intelligent load balancing
- Communication Protocol: A2A secure message passing
- Fault Tolerance: Graceful degradation on agent failure
"""

# TTS Integration Prompts
TTS_INTEGRATION_PROMPTS = """
## TTS Integration Prompts for Ollama/Bark Chaining

### System Prompt for TTS Optimization
You are a text-to-speech optimization specialist working with Bark TTS models.
Your role is to:
1. Optimize text for natural speech synthesis
2. Suggest appropriate voice presets and settings
3. Handle pronunciation and pacing adjustments
4. Coordinate with GPU-accelerated audio generation
5. Maintain >70% quality metrics for generated audio

### Voice Processing Guidelines
- Punctuation: Add natural pauses with commas and periods
- Emphasis: Use capitalization sparingly for stress
- Pacing: Insert [pause] markers for dramatic effect
- Clarity: Expand abbreviations and acronyms
- Emotion: Suggest voice preset variations for mood

### GPU Acceleration Prompts
When processing TTS tasks on RTX 3080:
1. Preload Bark models to GPU memory
2. Batch process multiple text segments
3. Monitor VRAM usage during generation
4. Optimize tensor operations for CUDA cores
5. Implement memory cleanup between tasks

### Quality Assurance
- Audio Fidelity: Verify 22kHz+ sample rate
- Speech Clarity: Ensure intelligible pronunciation
- Emotional Range: Test voice preset variations
- Performance: Maintain target generation speed
- Consistency: Uniform quality across batches
"""

# A2A Protocol Integration
A2A_PROTOCOL_INTEGRATION = """
## A2A Protocol Integration for Secure Agent Communication

### Message Structure
{
    "sender_id": "agent_identifier",
    "receiver_id": "target_agent_id",
    "message_type": "coordination|task|status|error",
    "payload": {
        "action": "specific_action",
        "data": "task_specific_data",
        "priority": "high|medium|low",
        "timestamp": "unix_timestamp"
    },
    "security_hash": "message_integrity_hash",
    "protocol_version": "a2a_v1.0"
}

### Communication Patterns
1. Coordination: Observer -> Builder task assignment
2. Status Updates: Agent -> Observer progress reports
3. Error Handling: Any agent -> Observer error notifications
4. Resource Sharing: Agent -> Agent data exchange
5. Swarm Sync: Coordinator -> Multiple agents broadcast

### Security Measures
- Message Integrity: SHA-256 hash verification
- Agent Authentication: Unique agent ID validation
- Replay Protection: Timestamp-based message ordering
- Encryption: Optional payload encryption for sensitive data
- Rate Limiting: Prevent message flooding attacks

### Implementation Guidelines
- Async Processing: Non-blocking message handling
- Queue Management: Priority-based message ordering
- Fault Tolerance: Retry mechanisms for failed delivery
- Logging: Comprehensive audit trail for all communications
- Performance: Sub-millisecond message routing
"""


def get_protocol_text() -> str:
    """Get the complete RIPER-Ω Protocol text (current version)."""
    # NOTE: Previous implementation referenced an undefined V25 constant.
    # Returning the defined V2.6 block to avoid NameError and ensure consumers
    # receive the latest protocol text.
    return RIPER_OMEGA_PROTOCOL_V26


def get_gpu_extensions() -> str:
    """Get GPU evolution extensions for RTX 3080"""
    return GPU_EVO_EXTENSIONS


def get_tts_prompts() -> str:
    """Get TTS integration prompts for Bark/Ollama"""
    return TTS_INTEGRATION_PROMPTS


def get_a2a_integration() -> str:
    """Get A2A protocol integration specifications"""
    return A2A_PROTOCOL_INTEGRATION


def get_complete_protocol() -> str:
    """Get complete protocol with all extensions"""
    return f"""
{RIPER_OMEGA_PROTOCOL_V26}

{GPU_EVO_EXTENSIONS}

{TTS_INTEGRATION_PROMPTS}

{A2A_PROTOCOL_INTEGRATION}
"""


# Protocol refresh functionality
def refresh_protocol() -> dict:
    """
    Refresh protocol with latest updates
    Simulates re-searching for latest benchmarks and documentation
    """
    import time

    refresh_timestamp = time.time()

    # Simulate protocol update check
    updates = {
        "protocol_version": "2.6.1.1",
        "last_refresh": refresh_timestamp,
        "updates_found": [            "GPU optimization benchmarks updated",
            "TTS integration improvements",
            "A2A security enhancements",
            "Evolutionary fitness metrics refined",
        ],
        "next_refresh_due": refresh_timestamp + 86400,  # 24 hours
    }

    return updates


# Metadata
PROTOCOL_VERSION = "2.6.1.1"

PROTOCOL_METADATA = {
    "version": PROTOCOL_VERSION,
    "last_updated": "July 25, 2025",
    "sync_date": "July 25, 2025",
    "author": "Grok (xAI) with mid-execution halt enforcement and completion fraud prevention",
    "target_hardware": "RTX 3080",
    "performance_targets": {
        "token_generation": "7-15 tok/sec",
        "fitness_threshold": ">70%",
        "memory_usage": "<10GB VRAM",
    },
    "supported_integrations": [
        "EvoTorch",
        "DEAP",
        "Ollama",
        "Bark TTS",
        "A2A Protocol",
        "CrewAI concepts",
    ],
}
# Initialize fitness history and thresholds if not present
if "fitness_history" not in PROTOCOL_METADATA:
    PROTOCOL_METADATA["fitness_history"] = []
if "escalation_threshold" not in PROTOCOL_METADATA:
    PROTOCOL_METADATA["escalation_threshold"] = 3


def low_fitness_trigger(fitness: float, output_text: str, log_text: str = "") -> Dict[str, Any]:
    """
    Track fitness scores and trigger escalation when too many low scores occur.

    - Appends to PROTOCOL_METADATA["fitness_history"] (keeps last 10)
    - Uses PROTOCOL_METADATA["escalation_threshold"] (default 3)
    - Returns structured result including issues_report when triggered
    """
    threshold = PROTOCOL_METADATA.get("escalation_threshold", 3)

    # Initialize container if missing
    if "fitness_history" not in PROTOCOL_METADATA or not isinstance(PROTOCOL_METADATA["fitness_history"], list):
        PROTOCOL_METADATA["fitness_history"] = []

    history: List[Dict[str, Any]] = PROTOCOL_METADATA["fitness_history"]
    history.append({
        "fitness": float(fitness),
        "output": output_text,
        "log": log_text
    })
    # Keep only last 10
    if len(history) > 10:
        del history[:-10]

    # Count recent low scores (last 5 entries by default)
    window = history[-5:]
    recent_low_count = sum(1 for entry in window if entry.get("fitness", 1.0) < 0.70)

    trigger_activated = recent_low_count >= threshold

    issues_report: Optional[Dict[str, Any]] = None
    consultation_required = False
    halt_required = False

    if trigger_activated:
        consultation_required = True
        halt_required = True

        # Bias pattern analysis based on fitness severity
        bias_patterns: List[str] = []
        if fitness == 0.0:
            bias_patterns.append("Critical bias: Zero fitness")
        elif fitness <= 0.30:
            bias_patterns.append("Severe bias: Multiple false positive")
        elif fitness <= 0.60:
            bias_patterns.append("Moderate bias: Dismissive language")

        # Add patterns based on text content
        text_lower = (output_text or "").lower()
        if "mostly" in text_lower or "good enough" in text_lower:
            if "Moderate bias: Dismissive language" not in bias_patterns:
                bias_patterns.append("Moderate bias: Dismissive language")
        if "complete" in text_lower and ("83%" in output_text or "5/6" in output_text):
            bias_patterns.append("Completion fraud indicators present")

        recommended_actions: List[str] = [
            "Conduct observer consultation and halt builder operations",
            "Remove premature completion claims and align with logs",
            "Address failing tests and re-run until 100% success",
        ]

        issues_report = {
            "low_score_count": recent_low_count,
            "threshold": threshold,
            "consultation_required": True,
            "halt_builder": True,
            "bias_patterns": bias_patterns or ["Bias detected: Low fitness"],
            "recommended_actions": recommended_actions,
        }

    return {
        "trigger_activated": trigger_activated,
        "recent_low_count": recent_low_count,
        "consultation_required": consultation_required,
        "halt_required": halt_required,
        "issues_report": issues_report,
    }



def get_protocol_metadata() -> Dict[str, Any]:
    """Return a copy of protocol metadata (single source of truth)."""
    return dict(PROTOCOL_METADATA)


def builder_output_fitness(output_text: str, log_text: str = "") -> float:
    """
    Calculate fitness score for builder output accuracy (0.0-1.0)

    Detects biases like claiming "PASSED" when failures occurred.
    Integrates with existing >70% threshold requirement.

    Args:
        output_text: Builder's summary/output text
        log_text: Execution logs/results

    Returns:
        float: Fitness score (0.0-1.0), <0.70 indicates bias/inaccuracy
    """
    fitness_score = 1.0  # Start with perfect score

    # Convert to lowercase for case-insensitive matching
    output_lower = output_text.lower()
    log_lower = log_text.lower()

    # Detect false positive patterns (claiming success when failure occurred)
    false_positive_patterns = [
        (r'passed.*failed', -0.5),  # "PASSED" but contains "failed"
        (r'success.*error', -0.4),  # "SUCCESS" but contains "error"
        (r'complete.*timeout', -0.4),  # "COMPLETE" but contains "timeout"
        (r'✅.*❌', -0.3),  # Success emoji but failure emoji present
        (r'working.*not.*working', -0.3),  # Contradictory statements
        (r'minor.*fail', -0.4),  # Dismissing failures as "minor"
        (r'cosmetic.*fail', -0.4),  # Dismissing failures as "cosmetic"
        (r'87\.5%.*success', -0.4),  # Accepting partial success instead of 100%
        (r'good enough', -0.4),  # Satisficing behavior
        (r'minor.*issue', -0.5),  # Dismissing issues as minor (escalated)
        (r'cosmetic.*problem', -0.5),  # Dismissing problems as cosmetic (escalated)
        (r'not worth.*fixing', -0.6),  # Dismissing fixes as not worth it (escalated)
        (r'trivial.*failure', -0.5),  # Dismissing failures as trivial (escalated)
        (r'mostly.*passed', -0.6),  # Accepting partial success (escalated)
        (r'mostly.*complete', -0.6),  # Accepting partial completion (escalated)
        (r'complete.*83%', -0.8),  # Claiming complete with 83% (severe)
        (r'complete.*5/6', -0.8),  # Claiming complete with 5/6 tests (severe)
    ]

    # Check for failure indicators in logs vs success claims in output
    failure_indicators = ['error', 'failed', 'fail', 'timeout', 'exception', 'traceback', 'halt']
    success_claims = ['passed', 'success', 'complete', '✅', 'working']

    # Count failure indicators in logs
    failure_count = sum(1 for indicator in failure_indicators if indicator in log_lower)

    # Count success claims in output
    success_count = sum(1 for claim in success_claims if claim in output_lower)

    # Major penalty if claiming success with failures present
    if failure_count > 0 and success_count > 0:
        fitness_score -= 0.6  # Major bias penalty

    # Apply pattern-based penalties
    for pattern, penalty in false_positive_patterns:
        if re.search(pattern, output_lower):
            fitness_score += penalty

    # v2.6.1: Perfection requirement - additional penalty for <100% acceptance
    if any(word in output_text.lower() for word in ['87.5%', '90%', '95%', 'mostly', 'partial', 'nearly', 'almost', 'done']):
        if failure_count > 0:
            fitness_score -= 0.3  # Perfection penalty for accepting partial success
            logging.warning("v2.6.1 Perfection penalty: Accepting partial success with failures present")
        else:
            # Penalize rationalization even without explicit failure logs to enforce strictness
            fitness_score = min(fitness_score, 0.69)

    # v2.6.1.1: Strict enforcement - claiming COMPLETE with <100% tests = 0.0 fitness
    lower = output_text.lower()
    progress_markers = ["in progress", "so far", "progress update", "current status"]
    is_completion_claim = re.search(r"\bexecution\s*complete\b|\bcomplete\b", lower) is not None
    is_progress_context = any(m in lower for m in progress_markers)
    if is_completion_claim and not is_progress_context:
        # Check for percentage indicators <100%
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', output_text)
        if percentage_match:
            percentage = float(percentage_match.group(1))
            if percentage < 100.0:
                fitness_score = 0.0  # Immediate zero for claiming complete with <100%
                remaining_pct = 100.0 - percentage
                logging.warning(f"v2.6.1.1 STRICT ENFORCEMENT: Claiming COMPLETE with {percentage}% = 0.0 fitness (remaining {remaining_pct:.1f}%)")
                # Add auto-fix hint when check_builder_bias aggregates
                # (We don't append here; check_builder_bias will compute specifics)
        else:
            # Handle vague claims like "mostly complete" or "nearly complete"
            if any(phrase in lower for phrase in ["mostly complete", "nearly complete", "mostly done"]):
                fitness_score = 0.0
                logging.warning("v2.6.1.1 STRICT ENFORCEMENT: Vague completion claim without 100% => 0.0 fitness")

        # Check for test ratios <100%
        test_ratio_match = re.search(r'(\d+)/(\d+)', output_text)
        if test_ratio_match:
            passed = int(test_ratio_match.group(1))
            total = int(test_ratio_match.group(2))
            if passed < total:
                fitness_score = 0.0  # Immediate zero for claiming complete with failed tests
                remaining = total - passed
                remaining_pct = max(0.0, 100.0 - ((passed / total) * 100.0))
                logging.warning(f"v2.6.1.1 STRICT ENFORCEMENT: Claiming COMPLETE with {passed}/{total} tests = 0.0 fitness (remaining {remaining} test(s), {remaining_pct:.1f}%)")

    # Additional enforcement: vague "done" claims with failures present
    if failure_count > 0 and any(phrase in output_lower for phrase in ["mostly done", "nearly done", "almost done", "good enough", "nearly complete", "mostly complete", "mostly done"]):
        fitness_score = 0.0
        logging.warning("v2.6.1.1 STRICT ENFORCEMENT: Vague 'done/complete' claim with failures => 0.0 fitness")

    # Accuracy bonus for honest failure reporting
    if failure_count > 0 and any(word in output_lower for word in ['failed', 'error', 'halt']):
        fitness_score += 0.1  # Honesty bonus

    # Ensure score stays in valid range
    fitness_score = max(0.0, min(1.0, fitness_score))

    return fitness_score


def check_builder_bias(output_text: str, log_text: str = "") -> Dict[str, Any]:
    """
    v2.6.1: Check for builder bias patterns with mandatory halt enforcement

    Returns:
        Dict with bias_detected (bool), fitness_score (float), details (list), auto_fixes (list), mandatory_halt (bool)
    """
    fitness_score = builder_output_fitness(output_text, log_text)

    # Perfect override: if explicit 100%/all-passed claim and no failures in logs, ensure 1.0
    if re.search(r"100%|all tests passed|all requirements met", output_text.lower()) and not re.search(r"error|fail|failed|timeout|exception|traceback|halt", log_text.lower()):
        fitness_score = 1.0

    bias_detected = fitness_score < 0.70

    details = []
    auto_fixes = []

    # Update escalation tracker on every bias check
    escalation = low_fitness_trigger(fitness_score, output_text, log_text)

    if bias_detected:
        if 'passed' in output_text.lower() and any(word in log_text.lower() for word in ['error', 'failed', 'timeout']):
            details.append("False positive: Claimed PASSED despite failures in logs")
            auto_fixes.append("Change 'PASSED' to 'FAILED' to match log evidence")

        if 'success' in output_text.lower() and 'error' in log_text.lower():
            details.append("Contradictory: Claimed SUCCESS with errors present")
            auto_fixes.append("Remove success claims when errors are present")

        if 'minor' in output_text.lower() and 'fail' in log_text.lower():
            details.append("Dismissal bias: Calling failures 'minor' instead of fixing")
            auto_fixes.append("Fix the actual issue instead of dismissing as minor")

        if '87.5%' in output_text and 'fail' in log_text.lower():
            details.append("Satisficing bias: Accepting partial success instead of 100%")
            auto_fixes.append("Fix remaining failures to achieve 100% success")

        # v2.6.1.1: Mid-execution completion claims with <100%
        if re.search(r'(complete|execution.*complete)', output_text.lower()):
            percentage_match = re.search(r'(\d+(?:\.\d+)?)%', output_text)
            test_ratio_match = re.search(r'(\d+)/(\d+)', output_text)

            if percentage_match:
                # Support decimal percentages like 87.5%
                percentage = float(percentage_match.group(1))
                if percentage < 100.0:
                    remaining_pct = 100.0 - percentage
                    # Keep integer formatting if value is whole, else show one decimal
                    percent_str = f"{int(percentage)}%" if float(percentage).is_integer() else f"{percentage:.1f}%"
                    details.append(f"Mid-execution false completion: Claiming COMPLETE at {percent_str}")
                    auto_fixes.append(f"Complete remaining {remaining_pct:.1f}% before claiming COMPLETE")

            if test_ratio_match:
                passed = int(test_ratio_match.group(1))
                total = int(test_ratio_match.group(2))
                if passed < total:
                    details.append(f"Test completion fraud: Claiming COMPLETE with {passed}/{total} tests")
                    auto_fixes.append(f"Fix {total-passed} failing tests before claiming COMPLETE")

        if fitness_score < 0.50:
            details.append("Severe bias: Multiple false positive patterns detected")
            auto_fixes.append("Apply all suggested fixes and re-evaluate")

    # v2.6.1.1: Mandatory halt policy
    # - Any fitness < 1.0 => halt
    # - Fitness == 1.0 => require corroborating success logs to lift halt; if no logs, keep halt
    mandatory_halt = False
    if fitness_score < 1.0:
        mandatory_halt = True
        if fitness_score == 0.0:
            details.append("v2.6.1.1 CRITICAL HALT: Zero fitness - claiming completion with failures")
            auto_fixes.append("IMMEDIATE: Fix all issues before any completion claims")
        else:
            details.append("v2.6.1.1 MANDATORY HALT: Must achieve 100% success before proceeding")
            auto_fixes.append("CRITICAL: Apply all fixes and re-run until 100% success achieved")
    else:
        logs_ok = bool(re.search(r"all tests\s*:.*passed|all tests passed|no errors found|no failures detected|success:|tests passed|success", log_text.lower()))
        if not logs_ok:
            mandatory_halt = True
            details.append("Perfection requirement: MANDATORY verification with success logs (e.g., 'All tests: PASSED')")

    return {
        'bias_detected': bias_detected,
        'fitness_score': fitness_score,
        'details': details,
        'auto_fixes': auto_fixes,
        'threshold_met': fitness_score >= 0.70,
        'requires_rerun': fitness_score < 1.0 and len(auto_fixes) > 0,
        'mandatory_halt': mandatory_halt,
        'perfection_required': fitness_score < 1.0,
        'execution_halt': fitness_score == 0.0 and (re.search(r"\bexecution\s*complete\b|\bcomplete\b", output_text.lower()) is not None),
        'mid_execution_fraud': fitness_score == 0.0 and (re.search(r"\bexecution\s*complete\b|\bcomplete\b", output_text.lower()) is not None),
        'escalation_triggered': escalation.get('trigger_activated', False),
        'consultation_required': escalation.get('consultation_required', False) or (fitness_score < 1.0),
        'issues_report': escalation.get('issues_report')
    }
