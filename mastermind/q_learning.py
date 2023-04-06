import random
from typing import Hashable


class QLearning:
    def __init__(self, env, gamma=0.9, eps=0.15, alpha=0.2):
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.env = env
        self.actions = self.env.actions
        self.q = {}

    def policy(self, state: Hashable, test: bool = False):
        if not state in self.q:
            self.initialise_q(state)
        if not test and random.random() <= self.eps:
            return random.choice(list(self.q[state].keys()))

        max_q_value = max(self.q[state].values())
        max_q_actions = [
            action for action, value in self.q[state].items() if value == max_q_value
        ]
        return random.choice(max_q_actions)

    def update_q_table(
        self, state: Hashable, action: tuple, reward, next_state: Hashable
    ):
        q_curr = self.q[state][action]
        if not next_state in self.q:
            self.initialise_q(next_state)
        q_next = max(self.q[next_state].values())
        self.q[state][action] += self.alpha * (reward + self.gamma * q_next - q_curr)

    def initialise_q(self, state):
        self.q[state] = {
            tuple(action): 0.0 for action in self.actions
        }  # initialise Q(S,:) with 0.0
