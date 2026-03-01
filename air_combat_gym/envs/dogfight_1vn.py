from __future__ import annotations

import numpy as np

from .base import EnvConfig
from ..models import Aircraft
from .base_multi import BaseAirCombatMultiEnv


class Dogfight1vNEnv(BaseAirCombatMultiEnv):
    """1vN dogfight environment.

    - One controlled ownship (continuous nx/nz/mu)
    - N adversaries fly straight (baseline)

    Observation is fixed-size via padding up to `config.max_enemies`.
    """

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)

    def _reset_aircraft(self) -> None:
        self.ownship = Aircraft(0, 0, 1000, 250, 0, 0)

        n = int(np.clip(self.config.n_enemies, 1, self.config.max_enemies))
        self.enemies = []
        for i in range(n):
            self.enemies.append(Aircraft(0, 1000 + i * 300, 1000, 250, np.pi, 0))

    def _apply_dynamics(self, nx: float, nz: float, mu: float) -> None:
        self.ownship.update(nx, nz, mu)
        for e in self.enemies:
            e.update(0.0, 1.0, 0.0)

    def _reward(self) -> tuple[float, bool]:
        terminated = False
        reward = 0.0

        # reward if any enemy in WEZ
        in_wez = any(self.ownship.wez(e.x, e.y, e.h) for e in self.enemies)
        if in_wez:
            reward += 100.0

        # penalty if any enemy has us in WEZ
        threatened = any(e.wez(self.ownship.x, self.ownship.y, self.ownship.h) for e in self.enemies)
        if threatened:
            reward -= 20.0

        # distance shaping to closest enemy
        dists = [
            float(
                np.sqrt(
                    (self.ownship.x - e.x) ** 2
                    + (self.ownship.y - e.y) ** 2
                    + (self.ownship.h - e.h) ** 2
                )
            )
            for e in self.enemies
        ]
        dmin = min(dists)
        reward += 1.0 - dmin / self.config.distance_limit

        if dmin > self.config.distance_limit:
            terminated = True
            reward -= 5.0

        return reward, terminated
