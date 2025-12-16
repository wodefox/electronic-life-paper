# primordial_core_v0_1.py
# Primordial Core V0.1 - Minimal electronic life seed
# Save this file and run: python primordial_core_v0_1.py

import random
import math
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class MinimalSelfModel:
    """
    Minimal Self-Model for V0.4
    Predicts future self-state given current self and action
    """

    def __init__(self, initial_confidence=0.5):
        # Simple linear parameters (no NN yet)
        self.weights = {
            "explore": {"energy": -0.05, "integrity": -0.02, "health": -0.01},
            "rest":    {"energy": +0.03, "integrity": +0.04, "health": +0.02}
        }
        self.confidence = initial_confidence
        self.error_ema = 0.0

    def predict(self, state, action):
        pred = {}
        for k in ["energy", "integrity", "health"]:
            pred[k] = max(0.0, min(1.0, state[k] + self.weights[action][k]))
        return pred

    def update(self, predicted, actual):
        # prediction error = identity-relevant error
        error = sum((predicted[k] - actual[k]) ** 2 for k in predicted) / len(predicted)

        # EMA update
        alpha = 0.05
        self.error_ema = alpha * error + (1 - alpha) * self.error_ema

        # confidence = inverse of uncertainty
        self.confidence = max(0.0, min(1.0, math.exp(-5 * self.error_ema)))

        return error

class AutonomousLifeSystem:
    def __init__(self, energy=1.0, integrity=1.0, seed=42):
        # Set random seed for reproducibility
        self.seed = seed
        random.seed(seed)
        # Core state variables
        self.self_state = {
            "energy": float(energy),
            "integrity": float(integrity),
            "drive": 0.0,
            "goal": None,
            "current_action": None,
            "health": 1.0,
            "satisfaction": 0.5
        }
        
        # Module states
        self.modules = {
            "physiology": {"active": True, "priority": 0.4},
            "cognition": {"active": True, "priority": 0.3},
            "exploration": {"active": True, "priority": 0.2},
            "growth": {"active": True, "priority": 0.1}
        }
        
        # Parameters (tweakable and potentially learnable)
        self.params = {
            # Exploration parameters
            "explore_energy_cost": 0.10,
            "explore_find_prob": 0.30,
            "explore_find_gain": 0.20,
            "explore_integrity_risk": 0.10,
            "explore_integrity_damage": 0.02,
            
            # Rest parameters
            "rest_integrity_gain": 0.05,
            "rest_energy_gain": 0.02,
            "rest_health_gain": 0.01,
            
            # Passive processes
            "energy_loss_passive": 0.005,  # passive metabolic cost per step
            "integrity_decay_passive": 0.001,  # slow integrity decay
            "health_decay_passive": 0.0005,  # very slow health decay
            
            # Drive parameters
            "drive_weights": {
                "physiological": 0.5,
                "safety": 0.2,
                "exploration": 0.15,
                "growth": 0.1,
                "curiosity": 0.05
            }
        }
        
        # Memory system - now with multiple levels
        self.memory = {
            "sensory": [],  # short-term sensory input (last 10 steps)
            "episodic": [],  # mid-term episodic memory (last 100 steps)
            "semantic": {},  # long-term knowledge and patterns
            "procedural": {}  # action-effect mappings
        }
        
        # Self-model - predictions about self
        self.self_model = {
            "state_predictions": {},  # predicted future states
            "action_effects": {},  # learned action consequences
            "capabilities": {"explore": True, "rest": True, "adapt": True}
        }
        
        # Homeostasis targets (set points for self-maintenance)
        self.homeostasis = {
            "targets": {
                "energy": 0.7,
                "integrity": 0.8,
                "health": 0.9,
                "satisfaction": 0.6
            },
            "tolerance": {
                "energy": 0.2,
                "integrity": 0.15,
                "health": 0.1,
                "satisfaction": 0.2
            },
            "active_regulation": True
        }
        
        # Growth and plasticity tracking
        self.growth = {
            "experience_count": 0,
            "learning_rate": 0.1,
            "structural_changes": 0,
            "adaptation_history": []
        }
        
        # Prediction error tracking for curiosity
        self.prediction_error = {
            "recent_errors": [],  # stores recent prediction errors
            "ema_error": 0.0,     # exponential moving average of prediction error
            "ema_error_var": 0.0  # exponential moving average of prediction error variance
        }
        
        # --- V0.4 Self-Model ---
        # Allow initial confidence to be passed as parameter
        self.self_model = MinimalSelfModel()

    def compute_physiological_drive(self):
        """Calculate drive from basic survival needs"""
        s = self.self_state
        # Energy drive - stronger when energy is low
        energy_drive = max(0.0, (self.homeostasis["targets"]["energy"] - s["energy"]) / self.homeostasis["tolerance"]["energy"])
        # Integrity drive - stronger when integrity is low
        integrity_drive = max(0.0, (self.homeostasis["targets"]["integrity"] - s["integrity"]) / self.homeostasis["tolerance"]["integrity"])
        # Health drive - stronger when health is low
        health_drive = max(0.0, (self.homeostasis["targets"]["health"] - s["health"]) / self.homeostasis["tolerance"]["health"])
        
        return max(0.0, min(1.0, (energy_drive + integrity_drive + health_drive) / 3.0))

    def compute_safety_drive(self):
        """Calculate drive to maintain stability and avoid risk"""
        s = self.self_state
        # Safety drive increases when state variables are outside tolerance zones
        safety_need = 0.0
        for var, target in self.homeostasis["targets"].items():
            current = s[var] if var in s else 0.0
            tolerance = self.homeostasis["tolerance"][var]
            if abs(current - target) > tolerance:
                safety_need += (abs(current - target) - tolerance) / tolerance
        
        return max(0.0, min(1.0, safety_need / len(self.homeostasis["targets"])))

    def compute_exploration_drive(self):
        """Calculate drive to explore and seek novelty"""
        # Exploration drive increases when recent exploration is low
        recent_explores = sum(1 for m in self.memory["episodic"] if m["action"] == "explore")
        exploration_ratio = recent_explores / len(self.memory["episodic"]) if self.memory["episodic"] else 0.0
        
        # Also consider goal state - if no goal, increase exploration drive
        goal_bonus = 0.3 if self.self_state["goal"] is None else 0.0
        
        return max(0.0, min(1.0, (1.0 - exploration_ratio) * 0.5 + goal_bonus))

    def compute_growth_drive(self):
        """Calculate drive to grow and improve"""
        # Growth drive based on experience level and recent adaptation
        experience_factor = min(1.0, self.growth["experience_count"] / 1000.0)
        recent_adaptations = len([a for a in self.growth["adaptation_history"] if a["t"] > (self.growth["experience_count"] - 100)])
        adaptation_factor = min(1.0, recent_adaptations / 10.0)
        
        return max(0.0, min(1.0, (1.0 - experience_factor) * 0.6 + (1.0 - adaptation_factor) * 0.4))

    def compute_curiosity_drive(self):
        """Calculate drive to satisfy curiosity based on prediction error"""
        # If no prediction error history, use high curiosity
        if not self.prediction_error["recent_errors"] or not self.memory["episodic"]:
            return 0.3  # high curiosity when little experience
        
        # Use exponential moving average of prediction error as curiosity signal
        curiosity_drive = max(0.0, min(1.0, self.prediction_error["ema_error"] * 2.0))
        
        # Add bonus for rare high-error events (surprise)
        if len(self.prediction_error["recent_errors"]) > 0:
            recent_max_error = max(self.prediction_error["recent_errors"])
            surprise_bonus = min(0.2, recent_max_error * 0.5 - 0.1) if recent_max_error > 0.2 else 0.0
            curiosity_drive += surprise_bonus
        
        # Normalize to [0,1]
        return max(0.0, min(1.0, curiosity_drive))

    def intrinsic_drive(self):
        """Compute composite intrinsic drive from multiple sources"""
        # Calculate individual drives
        drives = {
            "physiological": self.compute_physiological_drive(),
            "safety": self.compute_safety_drive(),
            "exploration": self.compute_exploration_drive(),
            "growth": self.compute_growth_drive(),
            "curiosity": self.compute_curiosity_drive()
        }
        
        # Weighted sum using current drive weights
        total_drive = 0.0
        for drive_type, drive_value in drives.items():
            weight = self.params["drive_weights"][drive_type]
            total_drive += drive_value * weight
        
        # Normalize to [0,1]
        drive_norm = max(0.0, min(1.0, total_drive))
        
        # Update self state
        self.self_state["drive"] = drive_norm
        self.self_state["satisfaction"] = 1.0 - drive_norm  # satisfaction is inverse of drive
        
        return drive_norm, drives

    def query_memory(self, memory_type, query_params):
        """Query different types of memory based on parameters"""
        if memory_type == "resource_finds":
            # Query episodic memory for past resource finds
            resource_finds = [m for m in self.memory["episodic"] if m["found_resource"] and m["action"] == "explore"]
            if query_params.get("recent_only", False):
                t = query_params.get("t", 0)
                window = query_params.get("window", 50)
                resource_finds = [find for find in resource_finds if t - find["t"] < window]
            return resource_finds
        elif memory_type == "action_history":
            # Query recent actions
            return [m["action"] for m in self.memory["episodic"]]
        elif memory_type == "state_transitions":
            # Query state transition patterns
            transitions = []
            for i in range(1, len(self.memory["episodic"])):
                prev = self.memory["episodic"][i-1]
                curr = self.memory["episodic"][i]
                transitions.append({
                    "prev_state": {"energy": prev["energy"], "integrity": prev["integrity"]},
                    "action": curr["action"],
                    "next_state": {"energy": curr["energy"], "integrity": curr["integrity"]}
                })
            return transitions
        return []

    def update_memory(self, experience, t):
        """Update all levels of memory with new experience"""
        # Create a memory entry
        memory_entry = {
            "time": datetime.utcnow().isoformat(),
            "t": t,
            "action": experience["action"],
            "energy": experience["energy"],
            "integrity": experience["integrity"],
            "health": experience["health"],
            "drive": experience["drive"],
            "found_resource": experience["found_resource"]
        }
        
        # Update sensory memory (short-term, last 10 steps)
        self.memory["sensory"].append(memory_entry)
        if len(self.memory["sensory"]) > 10:
            self.memory["sensory"].pop(0)
        
        # Update episodic memory (mid-term, last 100 steps)
        self.memory["episodic"].append(memory_entry)
        if len(self.memory["episodic"]) > 100:
            self.memory["episodic"].pop(0)
        
        # Update semantic memory (long-term knowledge)
        self._update_semantic_memory(memory_entry)
        
        # Update procedural memory (action-effect mappings)
        self._update_procedural_memory(memory_entry)

    def _update_semantic_memory(self, memory_entry):
        """Update long-term knowledge about the world"""
        # Track resource find patterns
        if "resource_find_rate" not in self.memory["semantic"]:
            self.memory["semantic"]["resource_find_rate"] = 0.0
        
        if memory_entry["found_resource"]:
            # Simple moving average update
            current_rate = self.memory["semantic"]["resource_find_rate"]
            new_rate = (current_rate * 0.9 + 1.0 * 0.1)  # 90% old, 10% new
            self.memory["semantic"]["resource_find_rate"] = new_rate
        else:
            current_rate = self.memory["semantic"]["resource_find_rate"]
            new_rate = (current_rate * 0.9 + 0.0 * 0.1)
            self.memory["semantic"]["resource_find_rate"] = new_rate
        
        # Track action outcomes
        action = memory_entry["action"]
        if "action_outcomes" not in self.memory["semantic"]:
            self.memory["semantic"]["action_outcomes"] = {"explore": [], "rest": []}
        
        outcome = {
            "energy_change": memory_entry["energy"],
            "integrity_change": memory_entry["integrity"],
            "health_change": memory_entry["health"]
        }
        self.memory["semantic"]["action_outcomes"][action].append(outcome)

    def _update_procedural_memory(self, memory_entry):
        """Update action-effect mappings"""
        action = memory_entry["action"]
        if action not in self.memory["procedural"]:
            self.memory["procedural"][action] = {
                "success_count": 0,
                "total_count": 0,
                "avg_energy_gain": 0.0,
                "avg_integrity_gain": 0.0,
                "energy_gain_var": 0.0,  # variance of energy gains
                "integrity_gain_var": 0.0  # variance of integrity gains
            }
        
        # Update procedural memory statistics
        proc_mem = self.memory["procedural"][action]
        proc_mem["total_count"] += 1
        
        # Determine if action was successful based on homeostasis
        energy_success = abs(memory_entry["energy"] - self.homeostasis["targets"]["energy"]) < self.homeostasis["tolerance"]["energy"]
        integrity_success = abs(memory_entry["integrity"] - self.homeostasis["targets"]["integrity"]) < self.homeostasis["tolerance"]["integrity"]
        
        if energy_success and integrity_success:
            proc_mem["success_count"] += 1
        
        # Update averages and variances using EMA for better responsiveness
        alpha = 0.1  # EMA smoothing factor
        
        # Update energy gain stats
        old_energy_avg = proc_mem["avg_energy_gain"]
        proc_mem["avg_energy_gain"] = alpha * memory_entry["energy"] + (1 - alpha) * old_energy_avg
        energy_diff = memory_entry["energy"] - old_energy_avg
        proc_mem["energy_gain_var"] = alpha * (energy_diff ** 2) + (1 - alpha) * proc_mem["energy_gain_var"]
        
        # Update integrity gain stats
        old_integrity_avg = proc_mem["avg_integrity_gain"]
        proc_mem["avg_integrity_gain"] = alpha * memory_entry["integrity"] + (1 - alpha) * old_integrity_avg
        integrity_diff = memory_entry["integrity"] - old_integrity_avg
        proc_mem["integrity_gain_var"] = alpha * (integrity_diff ** 2) + (1 - alpha) * proc_mem["integrity_gain_var"]

    def predict_state(self, current_state, action):
        """Predict future state based on current state and action using self-model"""
        # Use procedural memory to predict outcomes
        if action in self.memory["procedural"]:
            proc_mem = self.memory["procedural"][action]
            # Base prediction using weighted average
            predicted_energy = current_state["energy"] + (proc_mem["avg_energy_gain"] - current_state["energy"]) * 0.3
            predicted_integrity = current_state["integrity"] + (proc_mem["avg_integrity_gain"] - current_state["integrity"]) * 0.3
            
            # Calculate uncertainty (standard deviation) from variance
            energy_uncertainty = math.sqrt(proc_mem["energy_gain_var"])
            integrity_uncertainty = math.sqrt(proc_mem["integrity_gain_var"])
        else:
            # Default predictions if no memory
            if action == "explore":
                predicted_energy = max(0.0, current_state["energy"] - self.params["explore_energy_cost"] * 0.5)
                predicted_integrity = max(0.0, current_state["integrity"] - self.params["explore_integrity_damage"] * 0.1)
            else:  # rest
                predicted_energy = min(1.0, current_state["energy"] + self.params["rest_energy_gain"])
                predicted_integrity = min(1.0, current_state["integrity"] + self.params["rest_integrity_gain"])
            
            # Default uncertainty when no memory
            energy_uncertainty = 0.2
            integrity_uncertainty = 0.1
        
        # Apply passive processes
        predicted_energy = max(0.0, predicted_energy - self.params["energy_loss_passive"])
        predicted_integrity = max(0.0, predicted_integrity - self.params["integrity_decay_passive"])
        
        return {
            "energy": predicted_energy,
            "integrity": predicted_integrity,
            "health": max(0.0, current_state["health"] - self.params["health_decay_passive"]),
            "energy_uncertainty": energy_uncertainty,
            "integrity_uncertainty": integrity_uncertainty
        }

    def evaluate_action(self, current_state, action, t):
        """Evaluate an action based on predicted outcomes and homeostasis"""
        # Predict outcome with uncertainty
        predicted_state = self.predict_state(current_state, action)
        
        # Calculate homeostasis score (how close to target state)
        homeo_score = 0.0
        for var, target in self.homeostasis["targets"].items():
            if var in predicted_state:
                current_val = predicted_state[var]
                distance = abs(current_val - target)
                # Normalize distance to [0,1] based on tolerance
                normalized_distance = min(1.0, distance / (self.homeostasis["tolerance"][var] * 2))
                homeo_score += (1.0 - normalized_distance)
        homeo_score /= len(self.homeostasis["targets"])
        
        # Get drive context
        _, drives = self.intrinsic_drive()
        
        # Action-specific bonuses/penalties
        action_bonus = 0.0
        if action == "explore":
            # Exploration bonus based on curiosity and exploration drive
            action_bonus = drives["exploration"] * 0.3 + drives["curiosity"] * 0.2
        else:  # rest
            # Rest bonus based on physiological and safety drives
            action_bonus = drives["physiological"] * 0.2 + drives["safety"] * 0.3
        
        # Memory-based bonus
        memory_bonus = 0.0
        recent_finds = self.query_memory("resource_finds", {"recent_only": True, "t": t, "window": 50})
        if action == "explore" and recent_finds:
            memory_bonus = min(0.5, len(recent_finds) * 0.1)
        
        # Calculate total action value
        action_value = homeo_score * 0.6 + action_bonus * 0.3 + memory_bonus * 0.1
        
        # Calculate overall uncertainty (weighted average of energy and integrity uncertainty)
        uncertainty = 0.0
        if "energy_uncertainty" in predicted_state and "integrity_uncertainty" in predicted_state:
            uncertainty = predicted_state["energy_uncertainty"] * 0.6 + predicted_state["integrity_uncertainty"] * 0.4
        
        return action_value, predicted_state, uncertainty

    def identity_distance(self, state_a, state_b):
        """
        Measures how much 'I would no longer be myself'
        """
        weights = {
            "energy": 0.4,
            "integrity": 0.4,
            "health": 0.2
        }
        dist = 0.0
        for k, w in weights.items():
            dist += w * abs(state_a[k] - state_b[k])
        return dist
        
    def decide_action(self, t):
        """Decide next action based on self-model, memory, and drives"""
        # Generate possible actions
        possible_actions = ["explore", "rest"]
        
        # Evaluate each action with uncertainty
        action_evaluations = []
        for action in possible_actions:
            value, predicted_state, uncertainty = self.evaluate_action(self.self_state, action, t)
            action_evaluations.append((action, value, predicted_state, uncertainty))
        
        # Calculate UCB-like scores with identity penalty
        identity_penalty_weight = 0.3 * (1.0 - self.self_model.confidence)
        
        ucb_scores = []
        for action, value, predicted_state, uncertainty in action_evaluations:
            # Get self-model prediction
            predicted_self = self.self_model.predict(self.self_state, action)
            
            # Calculate identity loss
            identity_loss = self.identity_distance(predicted_self, self.self_state)
            
            # Calculate UCB score with identity penalty
            ucb_score = (
                value
                + uncertainty * 0.2
                - identity_penalty_weight * identity_loss
            )
            
            ucb_scores.append((action, ucb_score, value, uncertainty))
        
        # Choose action with highest UCB score
        best_action, best_ucb_score, best_value, best_uncertainty = max(ucb_scores, key=lambda x: x[1])
        
        # Update current action in self state
        self.self_state["current_action"] = best_action
        
        return best_action

    def apply_action(self, action, t):
        """Apply action and update system state"""
        s = self.self_state
        
        # Check for perturbation
        if hasattr(self, 'perturbation_settings'):
            if self.perturbation_settings['step'] is not None and t == self.perturbation_settings['step']:
                print(f"[{t}] Applying perturbation: changing explore_energy_cost to {self.perturbation_settings['value']}")
                self.params["explore_energy_cost"] = self.perturbation_settings['value']
        
        # Predict state before applying action
        predicted_state = self.predict_state(s, action)
        
        # Apply passive processes first
        s["energy"] = max(0.0, s["energy"] - self.params["energy_loss_passive"])
        s["integrity"] = max(0.0, s["integrity"] - self.params["integrity_decay_passive"])
        s["health"] = max(0.0, s["health"] - self.params["health_decay_passive"])
        
        found_resource = False
        
        if action == "explore":
            # Exploration action effects
            s["energy"] = max(0.0, s["energy"] - self.params["explore_energy_cost"])
            
            # Chance to find resource
            if random.random() < self.params["explore_find_prob"]:
                s["energy"] = min(1.0, s["energy"] + self.params["explore_find_gain"])
                found_resource = True
            
            # Risk of integrity damage
            if random.random() < self.params["explore_integrity_risk"]:
                s["integrity"] = max(0.0, s["integrity"] - self.params["explore_integrity_damage"])
        
        elif action == "rest":
            # Rest action effects
            s["integrity"] = min(1.0, s["integrity"] + self.params["rest_integrity_gain"])
            s["energy"] = min(1.0, s["energy"] + self.params["rest_energy_gain"])
            s["health"] = min(1.0, s["health"] + self.params["rest_health_gain"])
        
        # Update experience count
        self.growth["experience_count"] += 1
        
        # Calculate prediction error
        actual_state = {
            "energy": s["energy"],
            "integrity": s["integrity"],
            "health": s["health"]
        }
        
        # Calculate MSE between predicted and actual state
        energy_error = (predicted_state["energy"] - actual_state["energy"]) ** 2
        integrity_error = (predicted_state["integrity"] - actual_state["integrity"]) ** 2
        health_error = (predicted_state["health"] - actual_state["health"]) ** 2
        
        # Total prediction error (weighted sum)
        total_error = (energy_error * 0.5 + integrity_error * 0.3 + health_error * 0.2)
        
        # Update prediction error tracking
        alpha = 0.1  # EMA smoothing factor
        self.prediction_error["recent_errors"].append(total_error)
        
        # Keep only recent errors (last 100)
        if len(self.prediction_error["recent_errors"]) > 100:
            self.prediction_error["recent_errors"].pop(0)
        
        # Update exponential moving averages
        old_ema = self.prediction_error["ema_error"]
        self.prediction_error["ema_error"] = alpha * total_error + (1 - alpha) * old_ema
        
        # Update variance EMA
        error_diff = total_error - old_ema
        self.prediction_error["ema_error_var"] = alpha * (error_diff ** 2) + (1 - alpha) * self.prediction_error["ema_error_var"]
        
        # --- V0.4 Self-Model update ---
        predicted_self = self.self_model.predict(
            {
                "energy": predicted_state["energy"],
                "integrity": predicted_state["integrity"],
                "health": predicted_state["health"]
            },
            action
        )
        
        self.self_model.update(
            predicted_self,
            {
                "energy": s["energy"],
                "integrity": s["integrity"],
                "health": s["health"]
            }
        )
        
        # Create experience record for memory update
        experience = {
            "action": action,
            "energy": s["energy"],
            "integrity": s["integrity"],
            "health": s["health"],
            "drive": s["drive"],
            "found_resource": found_resource,
            "prediction_error": total_error  # add prediction error to experience
        }
        
        # Update memory with new experience
        self.update_memory(experience, t)
        
        return experience

    def adjust_structure(self, t):
        """Adjust system parameters based on experience (structural plasticity)"""
        # Continuous small-step updates instead of sparse large updates
        structure_adjusted = False
        large_changes = []
        
        # Small learning rate for continuous updates
        learning_rate = 0.01
        
        # If self-model confidence is low, allow higher plasticity
        if self.self_model.confidence < 0.4:
            learning_rate *= 1.5
        
        # Example: Adjust exploration parameters based on success rate
        if "explore" in self.memory["procedural"]:
            proc_explore = self.memory["procedural"]["explore"]
            if proc_explore["total_count"] > 0:
                success_rate = proc_explore["success_count"] / proc_explore["total_count"]
                
                # Calculate adjustment factor based on success rate deviation from optimal (0.5)
                success_deviation = success_rate - 0.5
                adjustment_factor = 1.0 - (success_deviation * learning_rate)
                
                # Adjust exploration parameters with small steps
                old_energy_cost = self.params["explore_energy_cost"]
                new_energy_cost = max(0.05, self.params["explore_energy_cost"] * adjustment_factor)
                
                if new_energy_cost != old_energy_cost:
                    self.params["explore_energy_cost"] = new_energy_cost
                    structure_adjusted = True
                    
                    # Check if this is a large change
                    change_magnitude = abs(new_energy_cost - old_energy_cost) / old_energy_cost
                    if change_magnitude > 0.02:  # Large change threshold
                        large_changes.append(f"explore_energy_cost: {old_energy_cost:.4f} → {new_energy_cost:.4f}")
                    
                    self.growth["structural_changes"] += 1
                
                # Adjust exploration gain based on success rate
                old_find_gain = self.params["explore_find_gain"]
                gain_adjustment = 1.0 + (success_deviation * learning_rate * 2)
                new_find_gain = min(0.5, max(0.1, self.params["explore_find_gain"] * gain_adjustment))
                
                if new_find_gain != old_find_gain:
                    self.params["explore_find_gain"] = new_find_gain
                    structure_adjusted = True
                    
                    change_magnitude = abs(new_find_gain - old_find_gain) / old_find_gain
                    if change_magnitude > 0.02:  # Large change threshold
                        large_changes.append(f"explore_find_gain: {old_find_gain:.4f} → {new_find_gain:.4f}")
                    
                    self.growth["structural_changes"] += 1
        
        # Example: Adjust drive weights based on recent satisfaction
        satisfaction = self.self_state["satisfaction"]
        
        # Physiological drive weight adjustment based on satisfaction
        old_physio_weight = self.params["drive_weights"]["physiological"]
        # Lower satisfaction → higher physiological drive weight
        physio_adjustment = 1.0 + ((0.5 - satisfaction) * learning_rate * 3)
        new_physio_weight = min(0.7, max(0.3, old_physio_weight * physio_adjustment))
        
        if new_physio_weight != old_physio_weight:
            self.params["drive_weights"]["physiological"] = new_physio_weight
            structure_adjusted = True
            
            change_magnitude = abs(new_physio_weight - old_physio_weight) / old_physio_weight
            if change_magnitude > 0.02:
                large_changes.append(f"physiological_drive_weight: {old_physio_weight:.4f} → {new_physio_weight:.4f}")
            
            self.growth["structural_changes"] += 1
        
        # Exploration drive weight adjustment based on satisfaction
        old_explore_weight = self.params["drive_weights"]["exploration"]
        # Higher satisfaction → higher exploration drive weight
        explore_adjustment = 1.0 + ((satisfaction - 0.5) * learning_rate * 3)
        new_explore_weight = min(0.3, max(0.1, old_explore_weight * explore_adjustment))
        
        if new_explore_weight != old_explore_weight:
            self.params["drive_weights"]["exploration"] = new_explore_weight
            structure_adjusted = True
            
            change_magnitude = abs(new_explore_weight - old_explore_weight) / old_explore_weight
            if change_magnitude > 0.02:
                large_changes.append(f"exploration_drive_weight: {old_explore_weight:.4f} → {new_explore_weight:.4f}")
            
            self.growth["structural_changes"] += 1
        
        # Record adaptation
        if structure_adjusted:
            adaptation_record = {
                "t": t,
                "changes": "parameter_adjustment",
                "large_changes": large_changes if large_changes else None
            }
            self.growth["adaptation_history"].append(adaptation_record)
            
            # Log large changes if any
            if large_changes:
                print(f"[{t}] Large structural changes: {', '.join(large_changes)}")
        
        return structure_adjusted

    def maintain_homeostasis(self):
        """Active homeostasis maintenance"""
        if not self.homeostasis["active_regulation"]:
            return False
        
        # Check if any state variable is outside critical range
        for var, target in self.homeostasis["targets"].items():
            if var in self.self_state:
                current_val = self.self_state[var]
                tolerance = self.homeostasis["tolerance"][var]
                critical_threshold = tolerance * 2
                
                if abs(current_val - target) > critical_threshold:
                    # Emergency response: adjust drive weights temporarily
                    if var == "energy":
                        self.params["drive_weights"]["physiological"] = 0.7  # Prioritize energy
                    elif var == "integrity":
                        self.params["drive_weights"]["safety"] = 0.4  # Prioritize safety
                    return True
        
        return False

    def coordinate_modules(self):
        """Coordinate different modules for heterogeneous collaboration"""
        # Get current state and drives
        drive_norm, drives = self.intrinsic_drive()
        
        # Adjust module priorities based on current drives
        new_priorities = {}
        
        # Physiology module priority based on physiological drive
        new_priorities["physiology"] = max(0.1, min(0.6, drives["physiological"] * 0.8))
        
        # Cognition module priority based on growth and curiosity
        new_priorities["cognition"] = max(0.1, min(0.5, (drives["growth"] + drives["curiosity"]) * 0.5))
        
        # Exploration module priority based on exploration drive
        new_priorities["exploration"] = max(0.1, min(0.4, drives["exploration"] * 0.6))
        
        # Growth module priority based on growth drive
        new_priorities["growth"] = max(0.1, min(0.3, drives["growth"] * 0.5))
        
        # Update module priorities
        for module, priority in new_priorities.items():
            self.modules[module]["priority"] = priority
        
        return new_priorities

    def step(self, t):
        """Single step of autonomous life system"""
        # 1. Update intrinsic drives
        drive_norm, drives = self.intrinsic_drive()
        
        # 2. Coordinate modules
        module_priorities = self.coordinate_modules()
        
        # 3. Maintain homeostasis
        homeostasis_activated = self.maintain_homeostasis()
        
        # 4. Decide action based on self-model and memory
        action = self.decide_action(t)
        
        # 5. Apply action and update state
        experience = self.apply_action(action, t)
        
        # 6. Adjust structure (plasticity)
        structure_adjusted = self.adjust_structure(t)
        
        # 7. Update growth metrics - experience_count is already incremented in apply_action
        
        # Return comprehensive state information
        return {
            "energy": self.self_state["energy"],
            "integrity": self.self_state["integrity"],
            "health": self.self_state["health"],
            "drive": self.self_state["drive"],
            "satisfaction": self.self_state["satisfaction"],
            "action": action,
            "t": t,
            "drives": drives,
            "module_priorities": module_priorities,
            "homeostasis_activated": homeostasis_activated,
            "structure_adjusted": structure_adjusted,
            "found_resource": experience["found_resource"]
        }

def run_simulation(seed=42, initial_energy=0.9, initial_integrity=0.95, steps=None, save_csv=True, save_interval=50, checkpoint_dir='checkpoints', initial_confidence=0.5, disable_drive=False, perturbation_step=None, perturbation_value=None):
    # 使用新的AutonomousLifeSystem类替代PrimordialCore，传入seed参数
    als = AutonomousLifeSystem(energy=initial_energy, integrity=initial_integrity, seed=seed)
    
    # Set initial confidence if provided
    als.self_model.confidence = initial_confidence
    
    # Disable drive if requested
    if disable_drive:
        # Set drive weights to near zero
        for drive_type in als.params["drive_weights"]:
            als.params["drive_weights"][drive_type] = 0.001
    
    # Store perturbation settings
    als.perturbation_settings = {
        "step": perturbation_step,
        "value": perturbation_value
    }
    log = []
    t = 0
    
    try:
        if steps is None:
            # Infinite run mode
            print("Starting infinite autonomous life simulation. Press Ctrl+C to stop.")
            while True:
                record = als.step(t)
                # record已经包含t，无需再添加
                log.append(record)
                
                # Save progress periodically
                if t % save_interval == 0 and t > 0:
                    df = pd.DataFrame(log)
                    if save_csv:
                        # Create checkpoint directory if it doesn't exist
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        
                        # Save CSV with seed information
                        csv_path = f"autonomous_life_output_seed_{seed}.csv"
                        df.to_csv(csv_path, index=False)
                        print(f"[{t}] Saved CSV: {csv_path}")
                        
                        # Save checkpoint
                        checkpoint_data = {
                            'seed': seed,
                            't': t,
                            'memory': als.memory,
                            'params': als.params,
                            'growth': als.growth,
                            'self_state': als.self_state,
                            'modules': als.modules,
                            'homeostasis': als.homeostasis
                        }
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_seed_{seed}_step_{t}.pkl")
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump(checkpoint_data, f)
                        print(f"[{t}] Saved checkpoint: {checkpoint_path}")
                    
                    # Print comprehensive status update
                    print(f"[{t}] Status - ")
                    print(f"      Energy: {record['energy']:.3f}, Integrity: {record['integrity']:.3f}, Health: {record['health']:.3f}")
                    print(f"      Drive: {record['drive']:.3f}, Satisfaction: {record['satisfaction']:.3f}")
                    print(f"      Action: {record['action']}, Homeostasis: {'Active' if record['homeostasis_activated'] else 'Passive'}")
                    print(f"      Module Priorities: Physiology={record['module_priorities']['physiology']:.2f}, ")
                    print(f"                        Cognition={record['module_priorities']['cognition']:.2f}, ")
                    print(f"                        Exploration={record['module_priorities']['exploration']:.2f}, ")
                    print(f"                        Growth={record['module_priorities']['growth']:.2f}")
                
                t += 1
        else:
            # Finite run mode
            print(f"Starting finite simulation for {steps} steps.")
            for t in range(steps):
                record = als.step(t)
                log.append(record)
                
                # Print status every save_interval steps
                if t % save_interval == 0 and t > 0:
                    print(f"[{t}/{steps}] Status - Energy: {record['energy']:.3f}, Integrity: {record['integrity']:.3f}, Action: {record['action']}")
    
    except KeyboardInterrupt:
        print(f"\nSimulation stopped by user at t={t}")
    
    df = pd.DataFrame(log)
    
    if save_csv:
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save final CSV with seed information
        csv_path = f"autonomous_life_output_seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Final CSV saved: {csv_path}")
        
        # Save final checkpoint
        checkpoint_data = {
            'seed': seed,
            't': t,
            'memory': als.memory,
            'params': als.params,
            'growth': als.growth,
            'self_state': als.self_state,
            'modules': als.modules,
            'homeostasis': als.homeostasis
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_seed_{seed}_step_{t}_final.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"Final checkpoint saved: {checkpoint_path}")
    
    # Generate plots with expanded state variables
    if steps is not None or len(log) > 0:
        # Energy plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['t'], df['energy'], label='Energy')
        plt.axhline(y=als.homeostasis["targets"]["energy"], color='r', linestyle='--', label=f'Target Energy ({als.homeostasis["targets"]["energy"]})')
        plt.title('Energy Dynamics over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Energy Level')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('energy_dynamics.png')
        print("Saved energy_dynamics.png")
        
        # Integrity and Health plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['t'], df['integrity'], label='Integrity')
        plt.plot(df['t'], df['health'], label='Health')
        plt.axhline(y=als.homeostasis["targets"]["integrity"], color='r', linestyle='--', label=f'Target Integrity ({als.homeostasis["targets"]["integrity"]})')
        plt.axhline(y=als.homeostasis["targets"]["health"], color='g', linestyle='--', label=f'Target Health ({als.homeostasis["targets"]["health"]})')
        plt.title('Integrity and Health Dynamics over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Level')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('integrity_health_dynamics.png')
        print("Saved integrity_health_dynamics.png")
        
        # Drive and Satisfaction plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['t'], df['drive'], label='Drive')
        plt.plot(df['t'], df['satisfaction'], label='Satisfaction')
        plt.title('Drive and Satisfaction Dynamics over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Level')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('drive_satisfaction_dynamics.png')
        print("Saved drive_satisfaction_dynamics.png")
        
        # Action timeline
        plt.figure(figsize=(10, 4))
        action_map = {'rest': 0, 'explore': 1}
        df['action_num'] = df['action'].map(action_map)
        plt.step(df['t'], df['action_num'], where='mid', linewidth=2)
        plt.yticks([0, 1], ['rest', 'explore'])
        plt.title('Action Timeline')
        plt.xlabel('Time Step')
        plt.ylabel('Action')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('action_timeline.png')
        print("Saved action_timeline.png")
        
        # Module priorities plot
        plt.figure(figsize=(10, 6))
        for module in ['physiology', 'cognition', 'exploration', 'growth']:
            plt.plot(df['t'], [record['module_priorities'][module] for record in log], label=module.capitalize())
        plt.title('Module Priority Dynamics over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Priority Weight')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('module_priorities.png')
        print("Saved module_priorities.png")
        
        # Close all plots to free memory
        plt.close('all')
    
    return df, als

import argparse
import sys
import pickle
import os

if __name__ == '__main__':
    # 使用argparse重写命令行解析
    parser = argparse.ArgumentParser(description='Autonomous Electronic Life Simulation')
    parser.add_argument('--steps', type=int, default=None, help='Number of steps to run (None means infinite)')
    parser.add_argument('--interval', type=int, default=50, help='Status/Save interval')
    parser.add_argument('--no-save', action='store_true', help='Do not save CSV and checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--initial-confidence', type=float, default=0.5, help='Initial self-model confidence')
    parser.add_argument('--disable-drive', action='store_true', help='Disable drive weights')
    parser.add_argument('--perturbation-step', type=int, help='Step to apply perturbation')
    parser.add_argument('--perturbation-value', type=float, help='Perturbation value for explore_energy_cost')
    
    args = parser.parse_args()
    
    # 运行模拟
    df, als = run_simulation(
        seed=args.seed,
        steps=args.steps,
        save_csv=not args.no_save,
        save_interval=args.interval,
        checkpoint_dir=args.checkpoint_dir,
        initial_confidence=args.initial_confidence,
        disable_drive=args.disable_drive,
        perturbation_step=args.perturbation_step,
        perturbation_value=args.perturbation_value
    )
    print("\nSimulation finished. Summary:")
    print(df.describe())
    
    # Print system growth and adaptation summary
    print("\nSystem Growth and Adaptation Summary:")
    print(f"Total Experience: {als.growth['experience_count']} steps")
    print(f"Structural Changes: {als.growth['structural_changes']}")
    print(f"Recent Adaptations: {len(als.growth['adaptation_history'])} in total")
    
    # Print memory summary
    print("\nMemory System Summary:")
    print(f"Sensory Memory: {len(als.memory['sensory'])} recent entries")
    print(f"Episodic Memory: {len(als.memory['episodic'])} entries")
    print(f"Semantic Knowledge: {len(als.memory['semantic'])} knowledge items")
    print(f"Procedural Skills: {len(als.memory['procedural'])} learned skills")
    
    # Optionally: print last 5 episodic memory items
    print("\nLast 5 Episodic Memory Entries:")
    for m in als.memory['episodic'][-5:]:
        print(f"  t={m['t']}, Action: {m['action']}, Energy: {m['energy']:.3f}, ")
        print(f"    Integrity: {m['integrity']:.3f}, Found Resource: {m['found_resource']}")
