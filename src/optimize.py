from scipy.optimize import minimize, brute
from tqdm import tqdm

import numpy as np


class Optimizer:
    def __init__(self, similarities, labels, lambda_loss=0.05):
        self._similarities = similarities
        self._labels = labels
        self._lambda_loss = lambda_loss
        self.safe_threshold = 0.0
        self.unsafe_threshold = 0.0

    def _calculate_gain_and_loss(self):
        gain_safe, loss_safe, gain_unsafe, loss_unsafe = 0, 0, 0, 0

        for label, similarity in zip(self._labels, self._similarities):
            if label == 0:
                gain_safe += similarity <= self.safe_threshold
                loss_safe += similarity >= self.unsafe_threshold
            else:
                gain_unsafe += similarity >= self.unsafe_threshold
                loss_unsafe += similarity <= self.safe_threshold

        gain = gain_safe + gain_unsafe
        loss = loss_safe + loss_unsafe

        return gain, loss

    def _loss_constraint(self, x):
        self.safe_threshold, self.unsafe_threshold = x
        _, loss = self._calculate_gain_and_loss()
        return (
            len(self._labels) * self._lambda_loss - loss
        )  # This should be positive if loss is less than 5% of df

    # objective function
    def _objective(self, x):
        self.safe_threshold, self.unsafe_threshold = x
        gain, loss = self._calculate_gain_and_loss()
        return loss - gain  # maximize gain & minimize loss

    def optimize(self):
        objects, gains, losses, thresholds = [], [], [], []
        for _ in tqdm(range(1000)):
            # random init
            self.safe_threshold = np.random.uniform(0, 1)
            self.unsafe_threshold = np.random.uniform(0, 1)

            _, loss = self._calculate_gain_and_loss()
            if self.unsafe_threshold < self.safe_threshold or loss >= 50:
                continue

            constraints = (
                {
                    "type": "ineq",
                    "fun": lambda x: x[1] - x[0],
                },  # unsafe_threshold >= safe_threshold
                {
                    "type": "ineq",
                    "fun": lambda x: self._loss_constraint(x),
                },  # New constraint: loss < 5% of df
            )
            result = minimize(
                self._objective,
                [self.safe_threshold, self.unsafe_threshold],
                method="trust-constr",
                constraints=constraints,
                bounds=[(0, 1), (0, 1)],
            )

            if result.success:
                optimized_thresholds = result.x
                optimized_gain, optimized_loss = self._calculate_gain_and_loss()
                objects.append(optimized_gain - optimized_loss)
                gains.append(optimized_gain)
                losses.append(optimized_loss)
                thresholds.append(optimized_thresholds)
            else:
                print(
                    "Optimization was unsuccessful. Check the constraints and objective function."
                )

        max_idx = np.argmax(objects)
        print(
            f"Gain:{gains[max_idx]}, Loss:{losses[max_idx]}, thresholds:{thresholds[max_idx]}"
        )

        self.safe_threshold = thresholds[max_idx][0]
        self.unsafe_threshold = thresholds[max_idx][1]

        return self.safe_threshold, self.unsafe_threshold
