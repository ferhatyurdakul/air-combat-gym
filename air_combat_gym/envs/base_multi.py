from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..models import Aircraft
from .base import EnvConfig


class BaseAirCombatMultiEnv(gym.Env):
    """Abstract base for multi-aircraft environments (e.g., 1vN).

    Provides:
    - Gymnasium reset/step signatures
    - action scaling
    - fixed-size padded observation helper
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._obs_dim = 3 * self.config.max_enemies + 3 + 3 * self.config.max_enemies
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self.nx_limits = [-1.0, 1.5]
        self.nz_limits = [-3.0, 9.0]
        self.mu_limits = [-5 * np.pi / 12, 5 * np.pi / 12]

        self.ownship: Aircraft
        self.enemies: list[Aircraft]
        self._steps = 0
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self._reset_aircraft()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1

        nx = self._scale(float(action[0]), self.nx_limits)
        nz = self._scale(float(action[1]), self.nz_limits)
        mu = self._scale(float(action[2]), self.mu_limits)

        self._apply_dynamics(nx, nz, mu)

        obs = self._get_obs()
        reward, terminated = self._reward()
        truncated = self._steps >= self.config.step_limit

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), bool(terminated), bool(truncated), {}

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from ..rendering import Renderer3D
            self._renderer = Renderer3D()
        states = self._build_render_states()
        self._renderer.render_frame(states, step=self._steps)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # hooks
    def _reset_aircraft(self) -> None:
        raise NotImplementedError

    def _apply_dynamics(self, nx: float, nz: float, mu: float) -> None:
        raise NotImplementedError

    def _reward(self) -> tuple[float, bool]:
        raise NotImplementedError

    # helpers
    def _scale(self, a: float, limits: list[float]) -> float:
        return limits[0] + (a + 1) * (limits[1] - limits[0]) / 2

    def _get_obs(self) -> np.ndarray:
        enemies = self.enemies[: self.config.max_enemies]
        while len(enemies) < self.config.max_enemies:
            enemies.append(None)

        rel = []
        enemy_kin = []
        for e in enemies:
            if e is None:
                rel.extend([0.0, 0.0, 0.0])
                enemy_kin.extend([0.0, 0.0, 0.0])
            else:
                rel.extend(
                    [
                        (self.ownship.x - e.x) / self.config.distance_limit,
                        (self.ownship.y - e.y) / self.config.distance_limit,
                        (self.ownship.h - e.h) / self.config.distance_limit,
                    ]
                )
                enemy_kin.extend(
                    [
                        e.v / e.max_speed_limit,
                        e.psi / e.psi_limit,
                        e.gamma / e.gamma_limit,
                    ]
                )

        own = [
            self.ownship.v / self.ownship.max_speed_limit,
            self.ownship.psi / self.ownship.psi_limit,
            self.ownship.gamma / self.ownship.gamma_limit,
        ]

        return np.array(rel + own + enemy_kin, dtype=np.float32)

    def _build_render_states(self) -> list:
        from ..rendering.renderer import AircraftState, BLUE, BLUE_DIM, BLUE_WEZ, RED, RED_DIM, RED_WEZ
        states = [
            AircraftState(
                x=self.ownship.x, y=self.ownship.y, h=self.ownship.h,
                v=self.ownship.v, psi=self.ownship.psi, gamma=self.ownship.gamma,
                label="Ownship", colour=BLUE, colour_dim=BLUE_DIM, wez_colour=BLUE_WEZ,
                wez_aperture_deg=self.ownship.wez_aperture_deg,
                wez_height_m=self.ownship.wez_height_m,
                trail=list(zip(self.ownship.log["x"], self.ownship.log["y"], self.ownship.log["h"])),
            ),
        ]
        for i, e in enumerate(self.enemies):
            states.append(AircraftState(
                x=e.x, y=e.y, h=e.h,
                v=e.v, psi=e.psi, gamma=e.gamma,
                label=f"Enemy {i}", colour=RED, colour_dim=RED_DIM, wez_colour=RED_WEZ,
                wez_aperture_deg=e.wez_aperture_deg,
                wez_height_m=e.wez_height_m,
                trail=list(zip(e.log["x"], e.log["y"], e.log["h"])),
            ))
        return states
