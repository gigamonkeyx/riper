"""
GPU Rewards Module for RIPER-Ω System
Implements RL-inspired rewards for GPU utilization in evo processes.
Ties to fitness >80% with penalties for CPU runs and COBRA audits for detection equity.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class GPURewards:
    """RL rewards system for GPU utilization"""
    def __init__(self):
        self.fitness_threshold = 0.80
        self.rewards = {
            "gpu_utilization": 0.0,
            "detection_equity": 0.0,
            "cpu_penalty": 0.0
        }
        self.total_reward = 0.0

    def calculate_gpu_utilization(self, utilized: float, available: float) -> float:
        """Reward for GPU utilization"""
        utilization_rate = utilized / available if available > 0 else 0.0
        reward = utilization_rate * 1.0  # Scale to 1.0 max
        self.rewards["gpu_utilization"] = reward
        logger.info(f"GPU utilization reward: {reward:.3f} (rate: {utilization_rate:.3f})")
        return reward

    def cobra_audit_detection(self, equity_score: float) -> float:
        """COBRA audit for detection equity"""
        reward = equity_score if equity_score > 0.80 else equity_score - 0.20  # Penalty below 80%
        self.rewards["detection_equity"] = reward
        logger.info(f"COBRA detection equity reward: {reward:.3f}")
        return reward

    def apply_cpu_penalty(self, is_cpu: bool) -> float:
        """Penalty for CPU fallback"""
        penalty = -0.5 if is_cpu else 0.0
        self.rewards["cpu_penalty"] = penalty
        if is_cpu:
            logger.warning(f"CPU fallback penalty applied: {penalty:.3f}")
        return penalty

    def compute_total_reward(self) -> float:
        """Compute total RL reward and check fitness"""
        self.total_reward = sum(self.rewards.values())
        normalized_reward = max(0.0, min(1.0, (self.total_reward + 1.0) / 3.0))  # Normalize to 0-1
        if normalized_reward < self.fitness_threshold:
            logger.warning(f"Fitness below threshold: {normalized_reward:.3f} < {self.fitness_threshold}")
        else:
            logger.info(f"Fitness achieved: {normalized_reward:.3f} >= {self.fitness_threshold}")
        return normalized_reward

    def get_rewards_report(self) -> Dict[str, Any]:
        """Generate rewards report"""
        return {
            "rewards": self.rewards,
            "total_reward": self.total_reward,
            "fitness": self.compute_total_reward()
        }

# Utility function
def evaluate_gpu_rewards(evo_data: Dict[str, Any]) -> Dict[str, Any]:
    rewards = GPURewards()
    
    # Example calculations (adjust based on evo_data)
    rewards.calculate_gpu_utilization(
        utilized=evo_data.get("gpu_utilized", 8.0),
        available=evo_data.get("gpu_available", 10.0)
    )
    rewards.cobra_audit_detection(equity_score=evo_data.get("equity_score", 0.85))
    rewards.apply_cpu_penalty(is_cpu=not evo_data.get("gpu_enabled", True))
    
    return rewards.get_rewards_report()
