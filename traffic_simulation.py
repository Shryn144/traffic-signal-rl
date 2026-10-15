
import random

class TrafficSimulation:
    def __init__(self):
        self.rates=[0.4,0.8,1.2,1.6]
        self.traffic=[0,0,0,0]
        self.rng=random.Random()

    def reset(self,seed=None):
        self.traffic=[0,0,0,0]
        if seed is not None:
            self.rng=random.Random(seed)
        return self.get_state()

    def get_state(self):
        return tuple(min(x//3,4) for x in self.traffic)

    def step(self,road):
        passed=min(self.traffic[road],5)
        self.traffic[road]-=passed

        for i in range(4):
            for j in range(4):
                if self.rng.random()<self.rates[i]/4:
                    self.traffic[i]+=1

        reward=-sum(self.traffic)
        return self.get_state(),passed,reward
