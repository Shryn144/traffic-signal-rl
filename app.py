import streamlit as st
import matplotlib.pyplot as plt
from traffic_simulation import TrafficSimulation
from q_learning import QLearningAgent

st.set_page_config(page_title="Traffic Signal Optimizer", page_icon="🚦", layout="wide")

st.title("🚦 Traffic Signal Timing Optimizer")
st.write("Using Q-Learning to reduce traffic congestion at a four-road intersection.")

st.info("The simulation compares a reinforcement learning agent with a fixed-time traffic signal.")

if st.button("Run Simulation", type="primary"):
    with st.spinner("Training the Q-Learning agent and testing both methods..."):
        sim = TrafficSimulation()
        agent = QLearningAgent()
        rewards = []

        for ep in range(1500):
            state = sim.reset(10000 + ep)
            total_reward = 0
            agent.epsilon = max(0.03, 0.3 * (1 - ep / 1500))

            for _ in range(100):
                action = agent.choose_action(state)
                next_state, passed, reward = sim.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        agent.epsilon = 0
        rl_results = []
        fixed_results = []

        for trial in range(10):
            seed = 50000 + trial
            sim.reset(seed)
            state = sim.get_state()
            total = 0

            for _ in range(200):
                action = agent.choose_action(state)
                state, passed, reward = sim.step(action)
                total += sum(sim.traffic)

            rl_results.append(total / 200)

            sim.reset(seed)
            total = 0

            for step in range(200):
                action = step % 4
                state, passed, reward = sim.step(action)
                total += sum(sim.traffic)

            fixed_results.append(total / 200)

        rl_avg = sum(rl_results) / len(rl_results)
        fixed_avg = sum(fixed_results) / len(fixed_results)
        improvement = (fixed_avg - rl_avg) / fixed_avg * 100 if fixed_avg else 0

    st.session_state["results"] = {
        "rewards": rewards,
        "rl": rl_results,
        "fixed": fixed_results,
        "rl_avg": rl_avg,
        "fixed_avg": fixed_avg,
        "improvement": improvement,
        "states": len(agent.q_table)
    }

if "results" in st.session_state:
    result = st.session_state["results"]

    st.subheader("Simulation Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Q-Learning Average Traffic", round(result["rl_avg"], 2))
    c2.metric("Fixed-Time Average Traffic", round(result["fixed_avg"], 2))
    c3.metric("Traffic Reduction", f'{result["improvement"]:.2f}%')

    st.caption("Average number of cars remaining in the system per simulation step. Results are from this simplified simulation, not real-world traffic measurements.")

    st.subheader("Q-Learning vs Fixed-Time Control")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), result["rl"], marker="o", label="Q-Learning")
    ax.plot(range(1, 11), result["fixed"], marker="o", label="Fixed-Time")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average Waiting Traffic")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Training Progress")
    fig2, ax2 = plt.subplots()
    ax2.plot(result["rewards"])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Q-Learning Training")
    ax2.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)

    st.write("Learned states:", result["states"])
else:
    st.write("Click **Run Simulation** to train the agent and view the results.")
