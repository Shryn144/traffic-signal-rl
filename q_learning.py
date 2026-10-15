
import random

class QLearningAgent:
    def __init__(self):
        self.q_table={}
        self.lr=0.15
        self.gamma=0.9
        self.epsilon=0.3

    def get_q(self,s):
        if s not in self.q_table:
            self.q_table[s]=[0,0,0,0]
        return self.q_table[s]

    def choose_action(self,s):
        q=self.get_q(s)

        if random.random()<self.epsilon:
            return random.randint(0,3)

        m=max(q)
        choices=[i for i in range(4) if q[i]==m]
        return random.choice(choices)

    def learn(self,s,a,r,ns):
        q=self.get_q(s)
        nq=self.get_q(ns)
        q[a]+=self.lr*(r+self.gamma*max(nq)-q[a])
