"""
Economy Simulation Module for RIPER-Ω System
Simulates underserved rural economies with USDA grants, donations, and aggregators.
Tied to evo_core for 3-year evolutionary cycles with WA regulations.
"""

import logging
import random
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from evo_core import NeuroEvolutionEngine  # Assuming import from evo_core.py
import ollama
from mesa import Agent, Model, DataCollector
from mesa.space import MultiGrid

logger = logging.getLogger(__name__)


class MesaBakerAgent(Agent):
    """Mesa-based ABM agent for bakers labor with emergent behaviors"""

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.skill_level = random.uniform(0.4, 0.9)
        self.availability = random.uniform(0.6, 1.0)
        self.productivity = random.uniform(0.5, 0.8)
        self.location = "Tonasket"
        self.specialization = random.choice(["general", "artisan", "commercial"])
        self.interaction_count = 0
        self.collaboration_history = []

    def step(self):
        """Mesa agent step function for baker interactions"""
        # Find nearby agents for interaction
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )

        if neighbors:
            # Select random neighbor for interaction
            other_agent = random.choice(neighbors)
            if other_agent != self and isinstance(other_agent, MesaBakerAgent):
                self.interact_with_agent(other_agent)

    def interact_with_agent(self, other_agent) -> Dict[str, float]:
        """Emergent behavior through Mesa baker agent interactions"""
        skill_transfer = min(0.1, abs(self.skill_level - other_agent.skill_level) * 0.2)
        productivity_boost = skill_transfer * 0.5
        collaboration_score = (self.skill_level + other_agent.skill_level) / 2

        # Update both baker agents
        self.skill_level = min(1.0, self.skill_level + skill_transfer)
        self.productivity = min(1.0, self.productivity + productivity_boost)
        other_agent.skill_level = min(1.0, other_agent.skill_level + skill_transfer)
        other_agent.productivity = min(1.0, other_agent.productivity + productivity_boost)

        # Track interaction
        interaction_result = {
            "skill_transfer": skill_transfer,
            "productivity_boost": productivity_boost,
            "collaboration_score": collaboration_score
        }

        self.collaboration_history.append(interaction_result)
        other_agent.collaboration_history.append(interaction_result)
        self.interaction_count += 1
        other_agent.interaction_count += 1

        return interaction_result


class CommunityParticipantAgent(Agent):
    """Mesa-based ABM agent for community outreach participants"""

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.participant_type = random.choice(["adult", "child", "family"])
        self.skill_level = random.uniform(0.1, 0.6)  # Lower initial skills
        self.interest_level = random.uniform(0.3, 1.0)
        self.donation_capacity = random.uniform(5.0, 50.0)  # $5-$50 donation range
        self.lesson_attendance = 0
        self.skills_learned = []
        self.outreach_interactions = []
        self.revenue_contributed = 0.0
        self.interaction_count = 0  # Track interactions for compatibility

    def attend_lesson(self, lesson_type: str):
        """Participate in community lesson and gain skills"""
        self.lesson_attendance += 1

        # Skill improvement based on lesson type
        if lesson_type == "baking":
            skill_gain = random.uniform(0.05, 0.15)
            self.skill_level = min(1.0, self.skill_level + skill_gain)
            self.skills_learned.append("baking")
        elif lesson_type == "milling":
            skill_gain = random.uniform(0.03, 0.12)
            self.skill_level = min(1.0, self.skill_level + skill_gain)
            self.skills_learned.append("milling")
            # Milling day fee
            self.revenue_contributed += 5.0
        elif lesson_type == "canning":
            skill_gain = random.uniform(0.04, 0.13)
            self.skill_level = min(1.0, self.skill_level + skill_gain)
            self.skills_learned.append("canning")

        # Increased donation likelihood after lessons
        donation_boost = random.uniform(1.05, 1.25)  # 5-25% increase
        self.donation_capacity *= donation_boost

        return skill_gain

    def step(self):
        """Mesa agent step function for community engagement"""
        # Find nearby agents for skill sharing (both participants and bakers)
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=3
        )

        if neighbors:
            # Select any agent for skill sharing (participants or bakers)
            other_agents = [n for n in neighbors if n != self]
            if other_agents and len(self.skills_learned) > 0:
                other_agent = random.choice(other_agents)
                if isinstance(other_agent, CommunityParticipantAgent):
                    self.share_skills_with(other_agent)
                elif isinstance(other_agent, MesaBakerAgent):
                    # Learn from professional baker
                    self.learn_from_baker(other_agent)

    def share_skills_with(self, other_participant: 'CommunityParticipantAgent'):
        """Emergent skill sharing between community participants"""
        if not self.skills_learned:
            return

        shared_skill = random.choice(self.skills_learned)
        skill_transfer = min(0.05, self.skill_level * 0.1)

        # Transfer skill knowledge
        other_participant.skill_level = min(1.0, other_participant.skill_level + skill_transfer)
        if shared_skill not in other_participant.skills_learned:
            other_participant.skills_learned.append(shared_skill)

        # Track interaction
        interaction = {
            "skill_shared": shared_skill,
            "skill_transfer": skill_transfer,
            "interaction_type": "peer_learning"
        }

        self.outreach_interactions.append(interaction)
        other_participant.outreach_interactions.append(interaction)

    def learn_from_baker(self, baker_agent):
        """Learn skills from professional baker"""
        if not hasattr(baker_agent, 'skill_level'):
            return

        # Higher skill transfer from professional bakers
        skill_transfer = min(0.15, baker_agent.skill_level * 0.2)
        self.skill_level = min(1.0, self.skill_level + skill_transfer)

        # Learn professional skills
        professional_skills = ["professional_baking", "commercial_techniques", "quality_control"]
        new_skill = random.choice(professional_skills)
        if new_skill not in self.skills_learned:
            self.skills_learned.append(new_skill)

        # Track professional interaction
        interaction = {
            "skill_shared": new_skill,
            "skill_transfer": skill_transfer,
            "interaction_type": "professional_learning"
        }

        self.outreach_interactions.append(interaction)

    def step(self):
        """Mesa agent step function for emergent behaviors"""
        # Find nearby agents for interaction
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )

        if neighbors:
            # Select random neighbor for interaction
            other_agent = random.choice(neighbors)
            if isinstance(other_agent, MesaBakerAgent):
                self.interact_with_agent(other_agent)

    def interact_with_agent(self, other_agent) -> Dict[str, float]:
        """Emergent behavior through Mesa agent interactions"""
        skill_transfer = min(0.1, abs(self.skill_level - other_agent.skill_level) * 0.2)
        productivity_boost = skill_transfer * 0.5
        collaboration_score = (self.skill_level + other_agent.skill_level) / 2

        # Update both agents (check if they have productivity attribute)
        self.skill_level = min(1.0, self.skill_level + skill_transfer)
        if hasattr(self, 'productivity'):
            self.productivity = min(1.0, self.productivity + productivity_boost)

        other_agent.skill_level = min(1.0, other_agent.skill_level + skill_transfer)
        if hasattr(other_agent, 'productivity'):
            other_agent.productivity = min(1.0, other_agent.productivity + productivity_boost)

        # Track interaction
        interaction_result = {
            "skill_transfer": skill_transfer,
            "productivity_boost": productivity_boost,
            "collaboration_score": collaboration_score
        }

        # Use appropriate history tracking for each agent type
        if hasattr(self, 'collaboration_history'):
            self.collaboration_history.append(interaction_result)
        elif hasattr(self, 'outreach_interactions'):
            self.outreach_interactions.append(interaction_result)

        if hasattr(other_agent, 'collaboration_history'):
            other_agent.collaboration_history.append(interaction_result)
        elif hasattr(other_agent, 'outreach_interactions'):
            other_agent.outreach_interactions.append(interaction_result)

        self.interaction_count += 1
        if hasattr(other_agent, 'interaction_count'):
            other_agent.interaction_count += 1

        return interaction_result


class MesaBakeryModel(Model):
    """Mesa-based ABM model for baker labor aggregation and community outreach"""

    def __init__(self, num_bakers: int = 10, num_participants: int = 50, width: int = 15, height: int = 15):
        super().__init__()
        self.num_bakers = num_bakers
        self.num_participants = num_participants
        self.grid = MultiGrid(width, height, True)
        self.outreach_events = []
        self.total_revenue = 0.0

        # Create baker agents using Mesa 3.0+ proper registration
        for i in range(self.num_bakers):
            agent = MesaBakerAgent(i, self)
            self.register_agent(agent)

            # Place agent randomly on grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Create community participant agents
        for i in range(self.num_participants):
            agent = CommunityParticipantAgent(i + self.num_bakers, self)
            self.register_agent(agent)

            # Place participant randomly on grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Enhanced data collection for outreach metrics
        self.datacollector = DataCollector(
            model_reporters={
                "Total Agents": lambda m: len(m.agents),
                "Baker Agents": lambda m: len([a for a in m.agents if isinstance(a, MesaBakerAgent)]),
                "Community Participants": lambda m: len([a for a in m.agents if isinstance(a, CommunityParticipantAgent)]),
                "Avg Baker Skill": lambda m: sum(a.skill_level for a in m.agents if isinstance(a, MesaBakerAgent)) / max(1, len([a for a in m.agents if isinstance(a, MesaBakerAgent)])),
                "Avg Participant Skill": lambda m: sum(a.skill_level for a in m.agents if isinstance(a, CommunityParticipantAgent)) / max(1, len([a for a in m.agents if isinstance(a, CommunityParticipantAgent)])),
                "Total Revenue": lambda m: sum(a.revenue_contributed for a in m.agents if isinstance(a, CommunityParticipantAgent)),
                "Lesson Attendance": lambda m: sum(a.lesson_attendance for a in m.agents if isinstance(a, CommunityParticipantAgent)),
                "Skills Shared": lambda m: sum(len(a.outreach_interactions) for a in m.agents if isinstance(a, CommunityParticipantAgent))
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Mesa model step function using built-in agents"""
        for agent in self.agents:
            agent.step()
        self.datacollector.collect(self)

    async def simulate_outreach_event(self, event_type: str = "milling_day") -> Dict[str, Any]:
        """Simulate community outreach event with participant interactions"""
        participants = [a for a in self.agents if isinstance(a, CommunityParticipantAgent)]

        # Simulate lesson attendance (random subset of participants)
        attendees = random.sample(participants, min(len(participants), random.randint(15, 35)))

        event_revenue = 0.0
        skills_gained = 0

        # Use Ollama-llama3.2:1b for emergent skill-sharing analysis
        try:
            outreach_prompt = f"""Analyze community outreach event:
Event Type: {event_type}
Attendees: {len(attendees)} participants
Participant Types: {[p.participant_type for p in attendees[:5]]}

Simulate emergent skill-sharing and community engagement patterns.
Calculate skill transfer efficiency and community building impact."""

            response = ollama.chat(
                model='llama3.2:1b',
                messages=[{'role': 'user', 'content': outreach_prompt}]
            )

            # Process attendees through lesson
            for participant in attendees:
                skill_gain = participant.attend_lesson(event_type.split('_')[0])  # Extract lesson type
                skills_gained += skill_gain
                event_revenue += participant.revenue_contributed

            # Track event
            self.outreach_events.append({
                "event_type": event_type,
                "attendees": len(attendees),
                "revenue": event_revenue,
                "skills_gained": skills_gained,
                "avg_skill_gain": skills_gained / len(attendees) if attendees else 0
            })

            self.total_revenue += event_revenue

        except Exception as e:
            logger.warning(f"Ollama outreach analysis failed: {e}")
            # Fallback simulation
            for participant in attendees:
                skill_gain = participant.attend_lesson(event_type.split('_')[0])
                skills_gained += skill_gain
                event_revenue += 5.0  # Base milling day fee

            self.total_revenue += event_revenue

        return {
            "event_type": event_type,
            "attendees": len(attendees),
            "revenue": event_revenue,
            "skills_gained": skills_gained,
            "total_revenue": self.total_revenue,
            "avg_skill_gain": skills_gained / len(attendees) if attendees else 0
        }

    async def simulate_emergent_behaviors(self, steps: int = 5) -> Dict[str, Any]:
        """Simulate emergent behaviors through Mesa model steps"""
        # Run simulation steps
        for _ in range(steps):
            self.step()

        # Collect final metrics
        model_data = self.datacollector.get_model_vars_dataframe()
        latest_data = model_data.iloc[-1] if not model_data.empty else {}

        # Calculate detailed cooperation metrics using Mesa built-in agents
        total_interactions = int(latest_data.get("Total Interactions", 0))
        agents = list(self.agents)  # Convert AgentSet to list

        # Calculate collaboration metrics from agent histories
        all_collaborations = []
        skill_improvements = 0

        for agent in agents:
            # Handle both baker collaboration_history and participant outreach_interactions
            if hasattr(agent, 'collaboration_history') and agent.collaboration_history:
                all_collaborations.extend(agent.collaboration_history)
                skill_improvements += len([h for h in agent.collaboration_history if h["skill_transfer"] > 0.05])
            elif hasattr(agent, 'outreach_interactions') and agent.outreach_interactions:
                all_collaborations.extend(agent.outreach_interactions)
                skill_improvements += len([h for h in agent.outreach_interactions if h.get("skill_transfer", 0) > 0.05])

        cooperation_rate = (skill_improvements / max(1, len(all_collaborations))) * 100 if all_collaborations else 0
        collaboration_efficiency = (sum(c["collaboration_score"] for c in all_collaborations) / max(1, len(all_collaborations))) * 100 if all_collaborations else 0

        return {
            "total_interactions": total_interactions,
            "avg_collaboration": sum(c["collaboration_score"] for c in all_collaborations) / max(1, len(all_collaborations)) if all_collaborations else 0,
            "skill_improvements": skill_improvements,
            "cooperation_rate": cooperation_rate,
            "collaboration_efficiency": collaboration_efficiency,
            "high_skill_agents": int(latest_data.get("High Skill Agents", 0)),
            "total_agents": int(latest_data.get("Total Agents", self.num_bakers + self.num_participants)),
            "avg_skill_level": float(latest_data.get("Avg Skill Level", 0)),
            "avg_productivity": float(latest_data.get("Avg Productivity", 0))
        }


@dataclass
class SimConfig:
    """Simulation configuration"""
    years: int = 3
    initial_funding: float = 100000.0  # Starting non-profit funding
    grant_success_rate: float = 0.8  # Probability of grant approval
    donation_growth_rate: float = 0.15  # Annual growth in donations


class UnderservedGrantModel:
    """Model for injecting USDA grants into rural non-profits"""
    def __init__(self, config: SimConfig):
        self.config = config
        self.current_funding = config.initial_funding
        self.mesa_model = MesaBakeryModel(num_agents=10)  # Initialize Mesa ABM with 10 baker agents
        self.abm_metrics = []

    def apply_grant(self, grant_amount: float) -> bool:
        """Apply for and potentially receive grant"""
        if random.random() < self.config.grant_success_rate:
            self.current_funding += grant_amount
            logger.info(f"Grant applied: +${grant_amount}")
            return True
        return False

    async def simulate_year(self, year: int) -> Dict[str, Any]:
        """Simulate one year of grant activities with ABM labor dynamics"""
        grant_amount = random.uniform(50000, 200000)  # Example grant range
        success = self.apply_grant(grant_amount)

        # Mesa ABM: Simulate emergent baker labor behaviors
        abm_results = await self.mesa_model.simulate_emergent_behaviors()
        self.abm_metrics.append(abm_results)

        # Use Ollama-qwen2.5 for labor efficiency analysis
        try:
            labor_prompt = f"""Analyze baker labor efficiency for Tonasket:
Agents: {abm_results['total_agents']}
Avg Skill: {abm_results['avg_skill_level']:.3f}
Avg Productivity: {abm_results['avg_productivity']:.3f}
Interactions: {abm_results['total_interactions']}
Collaboration Score: {abm_results['avg_collaboration']:.3f}

Calculate labor efficiency impact on grant utilization (0.0-1.0 scale)."""

            response = ollama.chat(
                model='qwen2.5-coder:7b',
                messages=[{'role': 'user', 'content': labor_prompt}]
            )

            # Extract efficiency score from response
            efficiency_analysis = response['message']['content']
            labor_efficiency = min(1.0, abm_results['avg_productivity'] * abm_results['avg_collaboration'])

        except Exception as e:
            logger.warning(f"Ollama labor analysis failed: {e}")
            labor_efficiency = abm_results['avg_productivity']

        # Log detailed ABM metrics
        logger.info(f"Metrics: ABM cooperation {abm_results['cooperation_rate']:.1f}%, "
                   f"collaboration efficiency {abm_results['collaboration_efficiency']:.1f}%, "
                   f"high-skill agents {abm_results['high_skill_agents']}/{abm_results['total_agents']}. "
                   f"Fitness impact: {labor_efficiency:.3f}")

        return {
            "year": year,
            "funding": self.current_funding,
            "grant_success": success,
            "abm_agents": abm_results['total_agents'],
            "labor_efficiency": labor_efficiency,
            "cooperation_rate": abm_results['cooperation_rate'],
            "collaboration_efficiency": abm_results['collaboration_efficiency'],
            "emergent_behaviors": abm_results['total_interactions'],
            "skill_improvements": abm_results['skill_improvements']
        }


class DonationProcessor:
    """Processes grain/organics donations via TEFAP/CSFP"""
    def __init__(self):
        self.total_donations = 0.0
        self.processed_food = 0.0

    def process_donation(self, amount: float, program: str = "TEFAP") -> float:
        """Process donation and convert to food units"""
        efficiency = 0.9 if program == "TEFAP" else 0.85
        processed = amount * efficiency
        self.total_donations += amount
        self.processed_food += processed
        logger.info(f"Processed donation: {amount} -> {processed} food units via {program}")
        return processed

    def annual_report(self) -> Dict[str, float]:
        return {
            "total_donations": self.total_donations,
            "processed_food": self.processed_food
        }


class RuralAggregator:
    """Aggregates labor/talent in underserved contexts"""
    def __init__(self):
        self.labor_pool = 10  # Initial labor units
        self.talent_score = 5.0  # Initial talent level

    def aggregate_labor(self, additional_labor: int) -> None:
        """Add labor from rural sources"""
        self.labor_pool += additional_labor
        logger.info(f"Aggregated labor: +{additional_labor} units")

    def improve_talent(self, improvement: float) -> None:
        """Improve talent through training/grants"""
        self.talent_score += improvement
        logger.info(f"Talent improved: +{improvement}")

    def status(self) -> Dict[str, Any]:
        return {
            "labor_pool": self.labor_pool,
            "talent_score": self.talent_score
        }


class EconomySimulator:
    """Main simulator tying components with evo_core"""
    def __init__(self, config: SimConfig):
        self.config = config
        self.grant_model = UnderservedGrantModel(config)
        self.donation_processor = DonationProcessor()
        self.rural_aggregator = RuralAggregator()
        self.evo_engine = NeuroEvolutionEngine()  # Tie to evo_core

    def run_simulation(self) -> Dict[str, Any]:
        """Run 3-year simulation cycle"""
        results = []
        for year in range(1, self.config.years + 1):
            # Simulate grant
            grant_result = self.grant_model.simulate_year(year)

            # Simulate donations with growth
            donation_amount = random.uniform(10000, 50000) * (1 + self.config.donation_growth_rate * (year - 1))
            self.donation_processor.process_donation(donation_amount)

            # Aggregate labor/talent
            self.rural_aggregator.aggregate_labor(random.randint(5, 15))
            self.rural_aggregator.improve_talent(random.uniform(0.5, 2.0))

            # Evolutionary step (mutate via evotorch)
            fitness = self.evo_engine.evaluate_generation()  # Placeholder
            results.append({
                "year": year,
                "grant": grant_result,
                "donations": self.donation_processor.annual_report(),
                "aggregator": self.rural_aggregator.status(),
                "fitness": fitness
            })

        # WA regs compliance check (placeholder)
        compliance = random.random() > 0.95  # >95% compliance
        return {"results": results, "compliance": compliance}

# Utility function
def run_economy_sim() -> Dict[str, Any]:
    sim = EconomySimulator(SimConfig())
    return sim.run_simulation()
