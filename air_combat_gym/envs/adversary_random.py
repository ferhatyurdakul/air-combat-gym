from __future__ import annotations

import numpy as np

from .base import BaseAirCombatEnv, EnvConfig
from ..models import Aircraft


class RandomAdversaryDogfightEnv(BaseAirCombatEnv):
    """1v1 dogfight where adversary takes random actions each step.

    Refactor of legacy `AirCombatEnv_4.py`.
    """

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    def _reset_aircraft(self) -> None:
        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0, 0)
        self.aircraft2 = Aircraft(0, 0, 1000, 250, np.pi, 0)

    def _apply_dynamics(self, nx1: float, nz1: float, mu1: float) -> None:
        # ownship controlled
        self.aircraft1.update(nx1, nz1, mu1)

        # adversary random control (normalized)
        a = self.np_random.uniform(-1.0, 1.0, size=(3,))
        nx2 = self._scale_action(float(a[0]), self.nx_limits)
        nz2 = self._scale_action(float(a[1]), self.nz_limits)
        mu2 = self._scale_action(float(a[2]), self.mu_limits)
        self.aircraft2.update(nx2, nz2, mu2)

    def _calculate_reward(self) -> tuple[float, bool]:
        terminated = False
        reward = 0.0

        if self.aircraft1.wez(self.aircraft2.x, self.aircraft2.y, self.aircraft2.h):
            reward += 100.0
        elif self.aircraft2.wez(self.aircraft1.x, self.aircraft1.y, self.aircraft1.h):
            reward -= 20.0

        distance = float(
            np.sqrt(
                (self.aircraft1.x - self.aircraft2.x) ** 2
                + (self.aircraft1.y - self.aircraft2.y) ** 2
                + (self.aircraft1.h - self.aircraft2.h) ** 2
            )
        )
        reward += 1.0 - distance / self.config.distance_limit

        if distance > self.config.distance_limit:
            terminated = True
            reward -= 5.0

        if self.aircraft1.v < 100:
            reward -= (100 - self.aircraft1.v)

        return reward, terminated
