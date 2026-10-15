
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from traffic_simulation import TrafficSimulation
from q_learning import QLearningAgent

st.set_page_config(page_title="AI Traffic Simulation", layout="wide")

st.title("AI Traffic Signal Simulation")
st.write("Watch the Q-learning agent select a road and control its traffic signal.")

speed = st.slider("Delay between steps (seconds)", 0.1, 1.0, 0.3, 0.1)
steps = st.slider("Number of simulation steps", 20, 150, 60, 10)

def draw_scene(traffic, green, step, passed):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#e8eee8")
    ax.set_facecolor("#47734b")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")

    # Roads and intersection
    ax.add_patch(patches.Rectangle((-3, -10), 6, 20, color="#42464b"))
    ax.add_patch(patches.Rectangle((-10, -3), 20, 6, color="#42464b"))

    for p in range(-9, 10, 2):
        ax.plot([0, 0], [p, p + 1], color="white", lw=1)
        ax.plot([p, p + 1], [0, 0], color="white", lw=1)

    # Stop lines
    ax.plot([-3, 3], [3.2, 3.2], color="white", lw=2)
    ax.plot([-3, 3], [-3.2, -3.2], color="white", lw=2)
    ax.plot([3.2, 3.2], [-3, 3], color="white", lw=2)
    ax.plot([-3.2, -3.2], [-3, 3], color="white", lw=2)

    # Queued cars: North, East, South, West
    for road, count in enumerate(traffic):
        for j in range(min(count, 12)):
            if road == 0:
                x, y = -1.5, 4.0 + j * 0.9
                w, h = 0.65, 0.95
            elif road == 1:
                x, y = 4.0 + j * 0.9, -1.5
                w, h = 0.95, 0.65
            elif road == 2:
                x, y = 1.5, -4.0 - j * 0.9
                w, h = 0.65, 0.95
            else:
                x, y = -4.0 - j * 0.9, 1.5
                w, h = 0.95, 0.65

            ax.add_patch(patches.FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.04",
                facecolor="#ffd34e", edgecolor="black", linewidth=0.7
            ))

    # Traffic lights
    lights = [
        (-2.4, 3.7),  # North
        (3.7, 2.4),   # East
        (2.4, -3.7),  # South
        (-3.7, -2.4)  # West
    ]
    names = ["NORTH", "EAST", "SOUTH", "WEST"]

    for road, (x, y) in enumerate(lights):
        color = "#16d96b" if road == green else "#f04444"
        ax.add_patch(patches.Circle((x, y), 0.32, color=color,
                                    ec="white", linewidth=1.5))
        ax.text(x, y - 0.65, names[road], ha="center",
                color="white", fontsize=8, weight="bold")

    ax.set_title(
        f"STEP {step}  |  GREEN: {names[green]}  |  CARS PASSED: {passed}",
        color="#17251a", fontsize=13, weight="bold", pad=12
    )
    ax.axis("off")
    fig.tight_layout()
    return fig

if st.button("Train AI and Run Live Simulation", type="primary"):
    sim = TrafficSimulation()
    agent = QLearningAgent()
    progress = st.progress(0, text="Training Q-learning...")

    for ep in range(500):
        state = sim.reset(1000 + ep)
        agent.epsilon = max(0.03, 0.3 * (1 - ep / 500))

        for _ in range(80):
            action = agent.choose_action(state)
            next_state, passed, reward = sim.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

        if ep % 10 == 0:
            progress.progress(min((ep + 1) / 500, 1.0),
                              text=f"Training episode {ep + 1}/500")

    progress.empty()
    agent.epsilon = 0
    
    # Evaluate AI against fixed-time signals
    rl_queues = []
    fixed_queues = []

    for trial in range(10):
        seed = 50000 + trial

        sim.reset(seed)
        state = sim.get_state()
        total_queue = 0

        for _ in range(100):
            action = agent.choose_action(state)
            state, passed, reward = sim.step(action)
            total_queue += sum(sim.traffic)

        rl_queues.append(total_queue / 100)

        sim.reset(seed)
        total_queue = 0

        for step in range(100):
            action = step % 4
            state, passed, reward = sim.step(action)
            total_queue += sum(sim.traffic)

        fixed_queues.append(total_queue / 100)

    ai_queue = sum(rl_queues) / len(rl_queues)
    fixed_queue = sum(fixed_queues) / len(fixed_queues)

    score = (
        (fixed_queue - ai_queue) / fixed_queue * 100
        if fixed_queue > 0 else 0
    )

    st.subheader("AI Model Performance")
    c1, c2, c3 = st.columns(3)

    c1.metric("AI Performance Score", f"{score:.1f}%",
              help="Percentage reduction in average queue compared with fixed-time control. A negative value means the AI performed worse.")
    c2.metric("AI Average Queue", f"{ai_queue:.2f} cars")
    c3.metric("Fixed-Time Average Queue", f"{fixed_queue:.2f} cars")

    st.caption(
        "Evaluation uses 10 test runs of 100 steps each. "
        "This score measures queue reduction, not prediction accuracy."
    )

    st.success(f"Training complete — {len(agent.q_table)} states learned")

    sim.reset(42)
    state = sim.get_state()

    scene = st.empty()
    status = st.empty()
    chart = st.empty()
    total_passed = 0
    history = [[], [], [], []]

    for step in range(1, steps + 1):
        action = agent.choose_action(state)
        state, cars_passed, reward = sim.step(action)
        total_passed += cars_passed

        for i in range(4):
            history[i].append(sim.traffic[i])

        fig = draw_scene(sim.traffic, action, step, cars_passed)
        scene.pyplot(fig)
        plt.close(fig)

        names = ["North", "East", "South", "West"]
        with status.container():
            a, b, c = st.columns(3)
            a.metric("Cars currently waiting", sum(sim.traffic))
            b.metric("Total cars passed", total_passed)
            c.metric("AI-selected road", names[action])

            st.write("**Cars waiting by road**")
            cols = st.columns(4)
            for i, col in enumerate(cols):
                col.metric(names[i], sim.traffic[i])

        with chart.container():
            fig2, ax = plt.subplots(figsize=(8, 2.5))
            for i, name in enumerate(names):
                ax.plot(history[i], label=name)
            ax.set_xlabel("Simulation step")
            ax.set_ylabel("Waiting cars")
            ax.legend(ncol=4)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

        time.sleep(speed)

    st.success("Simulation finished!")
    st.write("This is a simplified simulation, not a real-world traffic prediction.")

else:
    st.info("Click the button to train the AI and watch the four-way intersection.")
