# primordial_core_v0_1.py
# Primordial Core V0.1 - Minimal electronic life seed
# Save this file and run: python primordial_core_v0_1.py

import random
import math
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class PrimordialCore:
    def __init__(self, energy=1.0, integrity=1.0):
        self.self_state = {
            "energy": float(energy),
            "integrity": float(integrity),
            "drive": 0.0,
            "goal": None,
            "memory": []
        }
        # parameters (tweakable)
        self.explore_energy_cost = 0.10
        self.explore_find_prob = 0.30
        self.explore_find_gain = 0.20
        self.rest_integrity_gain = 0.05
        self.energy_loss_passive = 0.005  # passive metabolic cost per step

    def intrinsic_drive(self):
        s = self.self_state
        energy_drive = max(0.0, 1.0 - s["energy"])
        integrity_drive = max(0.0, 1.0 - s["integrity"])
        exploration_drive = 0.3 if s["goal"] is None else 0.0
        drive = energy_drive + integrity_drive + exploration_drive
        # normalize to [0,1] by dividing by plausible max (3.0)
        drive_norm = max(0.0, min(1.0, drive / 3.0))
        return drive_norm

    def query_memory_for_resource_finds(self):
        """Query episodic memory for past resource finds"""
        memory = self.self_state["memory"]
        # Find all instances where resource was found while exploring
        resource_finds = [m for m in memory if m["found_resource"] and m["action"] == "explore"]
        return resource_finds

    def decide_action(self, t):
        drive = self.self_state["drive"]
        
        # Base exploration probability
        p_explore = drive
        
        # Check memory for resource finds
        resource_finds = self.query_memory_for_resource_finds()
        
        if resource_finds:
            # If we've found resources before, increase exploration probability
            # Calculate how recent the finds were and how many there were
            recent_finds = [find for find in resource_finds if t - find["t"] < 50]  # Last 50 steps
            find_count = len(recent_finds)
            
            if find_count > 0:
                # Increase exploration probability based on recent finds
                # Each recent find adds 0.1 to exploration probability
                memory_bonus = min(0.5, find_count * 0.1)
                p_explore = min(1.0, p_explore + memory_bonus)
        
        return "explore" if random.random() < p_explore else "rest"

    def apply_action(self, action, t):
        s = self.self_state
        # passive energy consumption each step
        s["energy"] = max(0.0, s["energy"] - self.energy_loss_passive)

        if action == "explore":
            # exploration consumes energy
            s["energy"] = max(0.0, s["energy"] - self.explore_energy_cost)
            # chance to find resource
            if random.random() < self.explore_find_prob:
                s["energy"] = min(1.0, s["energy"] + self.explore_find_gain)
                found = True
            else:
                found = False
            # exploration slightly damages integrity sometimes
            if random.random() < 0.1:
                s["integrity"] = max(0.0, s["integrity"] - 0.02)
        elif action == "rest":
            # resting recovers integrity and small energy
            s["integrity"] = min(1.0, s["integrity"] + self.rest_integrity_gain)
            # resting also restores a small amount of energy
            s["energy"] = min(1.0, s["energy"] + 0.02)
            found = False
        else:
            found = False

        # update memory with a minimal episodic record
        s["memory"].append({
            "time": datetime.utcnow().isoformat(),
            "t": t,
            "action": action,
            "energy": s["energy"],
            "integrity": s["integrity"],
            "found_resource": found
        })

    def step(self, t):
        # compute drive
        self.self_state["drive"] = self.intrinsic_drive()
        # decide action based on drive and memory
        action = self.decide_action(t)
        # apply action effects
        self.apply_action(action, t)
        return {
            "energy": self.self_state["energy"],
            "integrity": self.self_state["integrity"],
            "drive": self.self_state["drive"],
            "action": action
        }

def run_simulation(seed=42, initial_energy=0.9, initial_integrity=0.95, steps=300, save_csv=True):
    random.seed(seed)
    pc = PrimordialCore(energy=initial_energy, integrity=initial_integrity)
    log = []
    for t in range(steps):
        record = pc.step(t)
        record["t"] = t
        log.append(record)

    df = pd.DataFrame(log)

    if save_csv:
        csv_path = "primordial_core_v0_1_output.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    # Simple plots
    plt.figure()
    plt.plot(df['t'], df['energy'])
    plt.title('Energy over time')
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_over_time.png')
    plt.show()

    plt.figure()
    plt.plot(df['t'], df['integrity'])
    plt.title('Integrity over time')
    plt.xlabel('t')
    plt.ylabel('integrity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('integrity_over_time.png')
    plt.show()

    plt.figure()
    plt.plot(df['t'], df['drive'])
    plt.title('Drive over time')
    plt.xlabel('t')
    plt.ylabel('drive')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('drive_over_time.png')
    plt.show()

    action_map = {'rest':0, 'explore':1}
    df['action_num'] = df['action'].map(action_map)
    plt.figure()
    plt.step(df['t'], df['action_num'], where='mid')
    plt.yticks([0,1], ['rest','explore'])
    plt.title('Action timeline (rest=0, explore=1)')
    plt.xlabel('t')
    plt.ylabel('action')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('action_timeline.png')
    plt.show()

    return df, pc

if __name__ == '__main__':
    df, pc = run_simulation()
    print("Simulation finished. Summary:")
    print(df.describe())
    # Optionally: print last 5 memory items
    print("Last 5 memory entries:")
    for m in pc.self_state['memory'][-5:]:
        print(m)
