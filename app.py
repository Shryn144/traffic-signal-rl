import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from traffic_simulation import TrafficSimulation
from q_learning import QLearningAgent

st.set_page_config(page_title="AI Traffic Simulator", page_icon="🚦", layout="wide")

st.title("🚦 AI Traffic Signal Simulator")
st.write("Watch a Q-learning agent control traffic at a four-road intersection.")

speed = st.slider("Animation speed (seconds per step)", 0.1, 1.0, 0.3, 0.1)
steps = st.slider("Simulation steps", 20, 150, 60, 10)

def draw_intersection(traffic, green_road, step):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#47734b")
    fig.patch.set_facecolor("#47734b")

    ax.add_patch(patches.Rectangle((-3, -10), 6, 20, color="#454545"))
    ax.add_patch(patches.Rectangle((-10, -3), 20, 6, color="#454545"))

    ax.plot([0, 0], [-10, 10], color="white", linestyle="--", linewidth=1)
    ax.plot([-10, 10], [0, 0], color="white", linestyle="--", linewidth=1)

    positions = [
        (-1.5, 4.0, 0, 1),
        (4.0, -1.5, 1, 0),
        (1.5, -4.0, 0, -1),
        (-4.0, 1.5, -1, 0)
    ]

    for road, count in enumerate(traffic):
        for j in range(min(count, 12)):
            if road == 0:
                x, y = -1.5, 4.0 + j * 0.85
            elif road == 1:
                x, y = 4.0 + j * 0.85, -1.5
            elif road == 2:
                x, y = 1.5, -4.0 - j * 0.85
            else:
                x, y = -4.0 - j * 0.85, 1.5

            ax.add_patch(
                patches.Rectangle(
                    (x - 0.28, y - 0.4), 0.56, 0.8,
                    color="#f4c542", ec="black", linewidth=0.5
                )
            )

    lights = [(-2.0, 3.0), (3.0, 2.0), (2.0, -3.0), (-3.0, -2.0)]

    for road, (x, y) in enumerate(lights):
        color = "#36e06b" if road == green_road else "#ff4545"
        ax.add_patch(patches.Circle((x, y), 0.42, color=color, ec="white"))

    names = ["North", "East", "South", "West"]
    ax.set_title(
        f"Step {step} | Green signal: {names[green_road]}",
        color="white", fontsize=15, pad=12
    )
    ax.text(-9.5, -9.2, "Yellow cars = waiting vehicles",
            color="white", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig

if st.button("Train AI and Start Simulation", type="primary"):
    sim = TrafficSimulation()
    agent = QLearningAgent()

    progress = st.progress(0, text="Training Q-learning agent...")

    for ep in range(500):
        state = sim.reset(1000 + ep)
        agent.epsilon = max(0.03, 0.3 * (1 - ep / 500))

        for _ in range(80):
            action = agent.choose_action(state)
            next_state, passed, reward = sim.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

        if ep % 10 == 0:
            progress.progress((ep + 1) / 500,
                              text=f"Training episode {ep + 1}/500")

    progress.empty()
    agent.epsilon = 0

    st.success(f"Training complete! Learned states: {len(agent.q_table)}")

    sim.reset(42)
    state = sim.get_state()

    display = st.empty()
    metrics = st.empty()
    total_passed = 0

    for step in range(1, steps + 1):
        action = agent.choose_action(state)
        state, passed, reward = sim.step(action)
        total_passed += passed

        fig = draw_intersection
