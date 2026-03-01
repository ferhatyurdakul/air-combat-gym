from __future__ import annotations

import numpy as np

from .base import BaseAirCombatEnv, EnvConfig
from ..models import Aircraft


class CircularTargetFollowEnv(BaseAirCombatEnv):
    """Follow a maneuvering target (circular motion baseline).

    Refactor of legacy `AirCombatEnv_2.py`.
    - Ownship controlled
    - Target aircraft executes a coordinated constant-bank turn (approx.)
    """

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    def _reset_aircraft(self) -> None:
        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0, 0)
        self.aircraft2 = Aircraft(0, 1000, 1000, 200, np.pi, 0, constant_speed=True)

    def _apply_dynamics(self, nx1: float, nz1: float, mu1: float) -> None:
        # ownship
        self.aircraft1.update(nx1, nz1, mu1)

        # target: constant bank angle coordinated turn
        bank_angle = 5 * np.pi / 12
        nz2 = 1 / np.cos(bank_angle)
        nx2 = 1.0
        mu2 = bank_angle
        self.aircraft2.update(nx2, nz2, mu2)

    def _calculate_reward(self) -> tuple[float, bool]:
        terminated = False
        reward = 0.0

        if self.aircraft1.wez(self.aircraft2.x, self.aircraft2.y, self.aircraft2.h):
            reward += 250.0
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
