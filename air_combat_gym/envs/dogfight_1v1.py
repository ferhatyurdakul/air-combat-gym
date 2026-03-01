from __future__ import annotations

import numpy as np

from .base import BaseAirCombatEnv, EnvConfig
from ..models import Aircraft


class Dogfight1v1Env(BaseAirCombatEnv):
    """1v1 dogfight environment (ownship controlled, adversary flies straight).

    This is a Gymnasium-compatible refactor of the legacy `AirCombatEnv.py`.
    """

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    def _reset_aircraft(self) -> None:
        self.aircraft1 = Aircraft(
            0,
            0,
            1000,
            250,
            0,
            0,
            wez_aperture_deg=self.config.own_wez_aperture_deg or 20.0,
            wez_height_m=self.config.own_wez_height_m or 300.0,
        )
        self.aircraft2 = Aircraft(
            0,
            1000,
            1000,
            250,
            np.pi,
            0,
            wez_aperture_deg=self.config.enemy_wez_aperture_deg or 20.0,
            wez_height_m=self.config.enemy_wez_height_m or 300.0,
        )

    def _apply_dynamics(self, nx1: float, nz1: float, mu1: float) -> None:
        self.aircraft1.update(nx1, nz1, mu1)
        # Adversary: straight (no control)
        self.aircraft2.update(0.0, 1.0, 0.0)

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
