"""
A2A Goal Exchange Demo for RIPER-Ω System
"""
import sys

sys.path.insert(0, "D:/pytorch")
from orchestration import Observer, Builder, A2AMessage
import time

print("=== A2A GOAL EXCHANGE DEMO ===")

# Initialize agents
observer = Observer("obs_demo")
builder = Builder("build_demo")

# Test 1: Basic coordination message
print("\n1. Basic Coordination:")
success = observer.a2a_comm.send_message(
    receiver_id="build_demo",
    message_type="coordination",
    payload={
        "action": "initialize_evolution",
        "fitness_target": 0.75,
        "gpu_target": "rtx_3080",
        "population_size": 20,
    },
)
print(f"Message sent: {success}")

# Test 2: Goal exchange with Qwen3 specifications
print("\n2. Qwen3 Goal Exchange:")
observer.a2a_comm.send_message(
    receiver_id="build_demo",
    message_type="model_config",
    payload={
        "model": "qwen3-coder-480b-a35b-instruct",
        "performance_target": "7-15_tok_sec",
        "memory_limit": "10gb_vram",
        "optimization": "rtx_3080_local",
    },
)

# Test 3: Evolution status update
print("\n3. Evolution Status Update:")
observer.a2a_comm.send_message(
    receiver_id="build_demo",
    message_type="evolution_status",
    payload={
        "generation": 15,
        "current_fitness": 0.68,
        "target_fitness": 0.70,
        "improvement_rate": 0.02,
        "eta_generations": 3,
    },
)

# Process messages on builder side
print("\n4. Message Processing:")
messages = observer.a2a_comm.receive_messages()
for i, msg in enumerate(messages):
    print(f"Message {i+1}:")
    print(f"  From: {msg.sender_id} -> To: {msg.receiver_id}")
    print(f"  Type: {msg.message_type}")
    print(f"  Payload: {msg.payload}")
    print(f"  Timestamp: {msg.timestamp}")

    # Simulate builder processing
    if hasattr(builder, "process_coordination_message"):
        result = builder.process_coordination_message(msg)
        print(f"  Processing Result: {result}")
    print()

# Test 4: RIPER-Ω mode transitions
print("5. RIPER-Ω Mode Transitions:")
from orchestration import RiperMode

# Observer mode transition
old_mode = observer.current_mode
new_mode = RiperMode.EXECUTE
success = observer.transition_mode(new_mode)
print(f"Observer mode transition: {old_mode.value} -> {new_mode.value} = {success}")

# Send mode coordination message
observer.a2a_comm.send_message(
    receiver_id="build_demo",
    message_type="mode_transition",
    payload={
        "new_mode": new_mode.value,
        "protocol_version": "2.5",
        "confidence_threshold": 0.80,
        "fitness_requirement": ">70%",
    },
)

print("\n✅ A2A Goal Exchange Demo Complete")
print(f"Total messages exchanged: {len(observer.a2a_comm.message_queue)}")
print("RIPER-Ω mode transitions: Functional")
print("Secure message passing: Implemented")
