from typing import List, Tuple
import random
from itertools import combinations_with_replacement, product
from collections import Counter

import numpy as np


class Mastermind:
    def __init__(
        self, n_pegs: int, n_colours: int, n_rows: int, codes: List[List] = None
    ):
        self.n_pegs = n_pegs
        self.n_colours = n_colours
        self.n_rows = n_rows
        if codes:
            self.actions = codes
        else:
            self.actions = self.get_action_set()
        self.state = np.zeros((self.n_rows, self.n_pegs + 2), int)
        self.is_reset = False

    def get_action_set(self) -> None:
        codes = product(range(1, self.n_colours + 1), repeat=self.n_pegs)
        return [list(code) for code in codes]

    def reset(self, code: List = None) -> np.ndarray:
        self.state = np.zeros((self.n_rows, self.n_pegs + 2), int)
        if code:
            assert (
                len(code) == self.n_pegs
            ), "Provided code length does not equal the number of pegs"
            assert code in self.actions, "Provided code is not in list of allowed codes"
            self.code = code
        else:
            self.code = random.choice(self.actions)
        self.turn = 0
        self.is_reset = True
        return self.state

    def step(self, action: List):
        done = False
        reward = 0
        info = ""
        if not self.is_reset:
            raise ValueError("The environment is not reset")
        else:
            feedback = self.get_feedback(action)
            self.state[self.turn, :] = action + feedback

        self.turn += 1
        if self.turn >= self.n_rows:
            done = True
            self.is_reset = False
        reward += -1
        if action == self.code:
            done = True
            self.is_reset = False
            reward += 10
        return (self.state, reward, done, info)

    def get_feedback(self, action: List):
        """Feedback given as [number of correct colours in wrong positions,
        number of correct colours in correct positions]
        Duplicate colours in the guess do not get counted unless they correspond
        to duplicate colours in the code"""

        correct_colours = sum((Counter(action) & Counter(self.code)).values())
        # correct_colours includes correct colours in correct positions
        correct_locations = sum(guess == peg for guess, peg in zip(action, self.code))
        return [correct_colours - correct_locations, correct_locations]

    def render(self):
        print(f"Secret code: {self.code}")
        print(f"Board: \n {np.flipud(self.state)}")
