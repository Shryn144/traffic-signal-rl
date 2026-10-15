
import random
import matplotlib.pyplot as plt
from traffic_simulation import TrafficSimulation
from q_learning import QLearningAgent

random.seed(7)
sim=TrafficSimulation()
agent=QLearningAgent()
rewards=[]

for ep in range(2500):
    s=sim.reset(10000+ep)
    total=0

    agent.epsilon=max(0.03,0.3*(1-ep/2500))

    for i in range(100):
        a=agent.choose_action(s)
        ns,c,r=sim.step(a)
        agent.learn(s,a,r,ns)
        s=ns
        total+=r

    rewards.append(total)

print("Training completed")
print("Learned states:",len(agent.q_table))

agent.epsilon=0
rl=[]
fixed=[]

for trial in range(10):
    seed=50000+trial
    s=sim.reset(seed)
    total=0

    for i in range(200):
        a=agent.choose_action(s)
        s,c,r=sim.step(a)
        total+=sum(sim.traffic)

    rl.append(total/200)

    sim.reset(seed)
    total=0

    for i in range(200):
        a=i%4
        s,c,r=sim.step(a)
        total+=sum(sim.traffic)

    fixed.append(total/200)

x=sum(rl)/len(rl)
y=sum(fixed)/len(fixed)

print("RL average:",round(x,2))
print("Fixed average:",round(y,2))

if y>0:
    print("Improvement:",round((y-x)/y*100,2),"%")

plt.plot(range(1,11),rl,marker="o",label="Q-learning")
plt.plot(range(1,11),fixed,marker="o",label="Fixed-time")
plt.xlabel("Trial")
plt.ylabel("Average Waiting Traffic")
plt.title("RL vs Fixed-Time")
plt.legend()
plt.grid()
plt.show()

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.grid()
plt.show()
