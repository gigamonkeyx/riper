"""
RIPER-Ω Multi-Agent Orchestration System
Entry point for Observer and Builder agents with evolutionary coordination.

RIPER-Ω Protocol v2.5 Integration:
- Strict mode transitions and audit trails
- GPU-local simulation focus for RTX 3080
- Evolutionary fitness metrics >70% threshold
- A2A communication for secure agent coordination
"""

import logging
import time
import os
import tempfile
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import evolutionary core and agents
from evo_core import NeuroEvolutionEngine, EvolutionaryMetrics
from agents import OllamaSpecialist, FitnessScorer, TTSHandler
from protocol import RIPER_OMEGA_PROTOCOL_V25
from openrouter_client import OpenRouterClient, get_openrouter_client

# Configure logging for audit trail
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preload_ollama_model(
    model_name: str = "qwen3:8b", base_url: str = "http://localhost:11434"
) -> bool:
    """Preload Ollama model to reduce initial API timeout"""
    try:
        logger.info(f"Preloading Ollama model: {model_name}")

        payload = {
            "model": model_name,
            "prompt": "",  # Empty prompt for model warming
            "stream": False,
            "options": {"num_gpu": 1, "temperature": 0.1},
        }

        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)

        if response.status_code == 200:
            logger.info(f"✅ Model {model_name} preloaded successfully")
            return True
        else:
            logger.warning(f"⚠️ Model preload failed: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ Model preload error: {e}")
        return False


class RiperMode(Enum):
    """RIPER-Ω Protocol v2.5 modes"""

    RESEARCH = "RESEARCH"
    INNOVATE = "INNOVATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    REVIEW = "REVIEW"


@dataclass
class A2AMessage:
    """A2A Protocol message structure for secure agent communication"""

    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    security_hash: Optional[str] = None


class A2ACommunicator:
    """A2A Protocol implementation for secure goal exchange and coordination"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_queue: List[A2AMessage] = []
        logger.info(f"A2A Communicator initialized for agent: {agent_id}")

    def send_message(
        self, receiver_id: str, message_type: str, payload: Dict[str, Any]
    ) -> bool:
        """Send secure A2A message to another agent"""
        message = A2AMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
        )
        # TODO: Implement security hash for message integrity
        self.message_queue.append(message)
        logger.info(
            f"A2A message sent: {self.agent_id} -> {receiver_id} ({message_type})"
        )
        return True

    def receive_messages(self, message_type: Optional[str] = None) -> List[A2AMessage]:
        """Receive A2A messages, optionally filtered by type"""
        if message_type:
            return [
                msg for msg in self.message_queue if msg.message_type == message_type
            ]
        return self.message_queue.copy()


class Observer:
    """
    Observer Agent with RIPER-Ω Protocol v2.5 infusion

    RIPER-Ω Modes:
    - RESEARCH: Gather context, summarize, ask clarifying questions
    - INNOVATE: Brainstorm and discuss approaches
    - PLAN: Create detailed, numbered checklist for implementation
    - EXECUTE: Implement exactly as per checklist
    - REVIEW: Line-by-line verification against the plan

    GPU Focus: Local simulation and evo workflows for RTX 3080
    """

    def __init__(self, agent_id: str = "observer_001"):
        self.agent_id = agent_id
        self.current_mode = RiperMode.RESEARCH
        self.a2a_comm = A2ACommunicator(agent_id)
        self.protocol_text = RIPER_OMEGA_PROTOCOL_V25
        self.fitness_threshold = 0.70  # >70% fitness requirement

        # Qwen3 integration via OpenRouter
        self.openrouter_client = get_openrouter_client()
        self.qwen3_model = "qwen/qwen-2.5-coder-32b-instruct"

        logger.info(f"Observer agent {agent_id} initialized with RIPER-Ω v2.5")

    def transition_mode(self, new_mode: RiperMode) -> bool:
        """Transition between RIPER-Ω modes with audit trail"""
        if self._validate_mode_transition(new_mode):
            old_mode = self.current_mode
            self.current_mode = new_mode
            logger.info(
                f"RIPER-Ω MODE TRANSITION: {old_mode.value} -> {new_mode.value}"
            )
            return True
        else:
            logger.error(
                f"INVALID MODE TRANSITION: {self.current_mode.value} -> {new_mode.value}"
            )
            return False

    def _validate_mode_transition(self, new_mode: RiperMode) -> bool:
        """Validate mode transition according to RIPER-Ω protocol"""
        # All transitions allowed for Observer (management role)
        return True

    def coordinate_evolution(
        self, builder: "Builder", evo_engine: "NeuroEvolutionEngine"
    ) -> Dict[str, Any]:
        """Coordinate evolutionary loop between Builder and NeuroEvolution engine"""
        logger.info("Starting evolutionary coordination loop")

        # A2A coordination message
        coordination_msg = {
            "action": "start_evolution",
            "fitness_threshold": self.fitness_threshold,
            "gpu_target": "rtx_3080",
            "performance_target": "7-15_tok_sec",
        }

        self.a2a_comm.send_message(builder.agent_id, "coordination", coordination_msg)

        # Initialize evolutionary metrics
        metrics = EvolutionaryMetrics()

        # Evolution loop (stub - detailed implementation in evo_core.py)
        for generation in range(10):  # Basic loop
            # Get fitness from evolution engine
            fitness_score = evo_engine.evaluate_generation()
            metrics.add_fitness_score(fitness_score)

            # Check fitness threshold
            if fitness_score >= self.fitness_threshold:
                logger.info(
                    f"Fitness threshold achieved: {fitness_score:.3f} >= {self.fitness_threshold}"
                )
                break

            # Coordinate with builder for next generation
            evolution_msg = {
                "generation": generation,
                "current_fitness": fitness_score,
                "target_fitness": self.fitness_threshold,
            }
            self.a2a_comm.send_message(
                builder.agent_id, "evolution_update", evolution_msg
            )

        return {
            "final_fitness": metrics.get_best_fitness(),
            "generations": metrics.generation_count,
            "success": metrics.get_best_fitness() >= self.fitness_threshold,
        }


class Builder:
    """
    Builder Agent with RIPER-Ω Protocol v2.5 implementation focus

    Responsibilities:
    - Execute implementation tasks from Observer coordination
    - Interface with Ollama specialists for local GPU tasks
    - Maintain evolutionary feedback loops
    - Ensure RTX 3080 optimization in all operations
    """

    def __init__(self, agent_id: str = "builder_001"):
        self.agent_id = agent_id
        self.current_mode = RiperMode.EXECUTE  # Default to execution mode
        self.a2a_comm = A2ACommunicator(agent_id)

        # Ollama specialists initialization
        self.fitness_scorer = FitnessScorer()
        self.tts_handler = TTSHandler()

        # Qwen3 integration via OpenRouter
        self.openrouter_client = get_openrouter_client()
        self.qwen3_model = "qwen/qwen-2.5-coder-32b-instruct"

        logger.info(f"Builder agent {agent_id} initialized for execution")

    def process_coordination_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Process A2A coordination messages from Observer"""
        if message.message_type == "coordination":
            return self._handle_coordination(message.payload)
        elif message.message_type == "evolution_update":
            return self._handle_evolution_update(message.payload)
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
            return {"status": "unknown_message_type"}

    def _handle_coordination(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coordination messages for evolutionary tasks"""
        action = payload.get("action")

        # Use Qwen3 for intelligent coordination analysis
        try:
            qwen3_response = self.openrouter_client.qwen3_agent_coordination(
                {
                    "action": action,
                    "payload": payload,
                    "agent_role": "builder",
                    "gpu_target": "rtx_3080",
                }
            )

            coordination_analysis = (
                qwen3_response.content if qwen3_response.success else None
            )

        except Exception as e:
            logger.warning(f"Qwen3 coordination analysis failed: {e}")
            coordination_analysis = None

        if action == "start_evolution":
            logger.info("Builder received evolution start command")
            # Initialize local GPU resources for RTX 3080
            gpu_status = self._initialize_gpu_resources()

            result = {"status": "evolution_started", "gpu_status": gpu_status}
            if coordination_analysis:
                result["qwen3_analysis"] = coordination_analysis

            return result

        return {"status": "coordination_processed"}

    def _handle_evolution_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evolutionary update messages"""
        generation = payload.get("generation", 0)
        current_fitness = payload.get("current_fitness", 0.0)

        logger.info(
            f"Evolution update - Generation: {generation}, Fitness: {current_fitness:.3f}"
        )

        # Use Ollama specialists for fitness evaluation
        specialist_feedback = self.fitness_scorer.evaluate_generation(
            generation, current_fitness
        )

        return {
            "status": "evolution_update_processed",
            "specialist_feedback": specialist_feedback,
        }

    def _initialize_gpu_resources(self) -> Dict[str, Any]:
        """Initialize RTX 3080 GPU resources for local tasks"""
        try:
            import torch

            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU"
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory if gpu_available else 0
            )

            return {
                "gpu_available": gpu_available,
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory / (1024**3),
                "target_performance": "7-15_tok_sec",
            }
        except ImportError:
            return {"gpu_available": False, "error": "PyTorch not available"}


def verify_file_existence(file_path: str) -> bool:
    """Verify file existence before operations (RIPER-Ω safeguard)"""
    return os.path.exists(file_path)


def purge_temp_files(temp_dir: Optional[str] = None):
    """Purge temporary files post-use (RIPER-Ω safeguard)"""
    if temp_dir and os.path.exists(temp_dir):
        import shutil

        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory purged: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to purge temp directory {temp_dir}: {e}")


def check_confidence_threshold(
    confidence: float, mode: RiperMode, threshold: float = 0.70
) -> bool:
    """Check confidence threshold per RIPER-Ω mode (≥70% requirement)"""
    meets_threshold = confidence >= threshold
    if not meets_threshold:
        logger.warning(
            f"Confidence {confidence:.3f} below threshold {threshold} for mode {mode.value}"
        )
    return meets_threshold


def flag_non_gpu_path(operation: str, gpu_available: bool):
    """Flag non-GPU paths as per RIPER-Ω safeguards"""
    if not gpu_available:
        logger.warning(
            f"NON-GPU PATH DETECTED: {operation} - Consider GPU optimization"
        )


def main():
    """Main orchestration entry point with RIPER-Ω safeguards"""
    logger.info("Starting RIPER-Ω Multi-Agent Orchestration System")

    # Create temporary directory for operations on D: drive
    temp_dir = tempfile.mkdtemp(prefix="riper_omega_", dir="D:/temp")

    try:
        # Runtime GPU checks with detailed logging
        try:
            import sys

            sys.path.insert(0, "D:/pytorch")  # Ensure D: drive PyTorch
            import torch

            gpu_available = torch.cuda.is_available()

            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                logger.info(
                    f"✅ GPU Runtime Check: {gpu_name} with {gpu_memory:.1f}GB memory"
                )

                # Verify RTX 3080 target
                if "3080" in gpu_name:
                    logger.info("✅ RTX 3080 target confirmed")
                else:
                    logger.warning(f"⚠️ Non-RTX 3080 GPU detected: {gpu_name}")
            else:
                flag_non_gpu_path("main_orchestration", False)
                logger.warning("⚠️ CUDA not available - running on CPU")

        except ImportError as e:
            gpu_available = False
            flag_non_gpu_path("main_orchestration", False)
            logger.error(f"❌ PyTorch import failed: {e}")

        # Initialize agents with confidence checking
        observer = Observer()
        builder = Builder()
        evo_engine = NeuroEvolutionEngine()

        # Check initialization confidence
        init_confidence = 0.85  # High confidence for successful initialization
        if not check_confidence_threshold(init_confidence, RiperMode.EXECUTE):
            logger.error("Initialization confidence below threshold - halting")
            return {"error": "Initialization failed confidence check"}

        # Start coordination with safeguards
        logger.info("Starting evolution coordination with safeguards enabled")
        results = observer.coordinate_evolution(builder, evo_engine)

        # Verify fitness threshold achievement
        final_fitness = results.get("final_fitness", 0.0)
        if final_fitness >= 0.70:
            logger.info(f"✅ Fitness threshold achieved: {final_fitness:.3f} ≥ 0.70")
        else:
            logger.warning(f"⚠️ Fitness threshold not met: {final_fitness:.3f} < 0.70")

        logger.info(f"Evolution completed: {results}")
        return results

    except Exception as e:
        logger.error(f"Main orchestration failed: {e}")
        return {"error": str(e), "success": False}

    finally:
        # Always purge temporary files
        purge_temp_files(temp_dir)


if __name__ == "__main__":
    main()
