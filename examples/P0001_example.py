# 📓 Example: Simulate Oxidation of Methyl Oleate
# Problem ID: P0001

from gsagent import GSagent

# Step 1: Initialize the GS agent
agent = GSagent(registry_path="bricks/registry.json")

# Step 2: Define the problem input
problem_input = {
    "mixture": "methyl_oleate",
    "temp": 60,
    "oxygen": 0.21,
    "time": 72
}

# Step 3: Run the simulation
try:
    result = agent.run("radigen.solve", **problem_input)
    print("Simulation output:")
    print(result)
except Exception as e:
    print("Simulation error:", e)

# Step 4: Archive result (manual or automated step)
# Suggest saving to problems/P0001.json under "response"

