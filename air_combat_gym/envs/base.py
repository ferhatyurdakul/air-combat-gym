from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..models import Aircraft


@dataclass
class EnvConfig:
    distance_limit: float = 3000.0
    step_limit: int = 500

    # Multi-enemy settings (for 1vN envs)
    n_enemies: int = 1
    max_enemies: int = 4

    # Optional WEZ overrides (if None, Aircraft defaults apply)
    own_wez_aperture_deg: float | None = None
    own_wez_height_m: float | None = None
    enemy_wez_aperture_deg: float | None = None
    enemy_wez_height_m: float | None = None


class BaseAirCombatEnv(gym.Env):
    """Abstract base for 3-DoF air-combat environments.

    Goals:
    - unify Gymnasium API (reset/step signatures)
    - share action scaling and observation building utilities
    - make it easy to create new env variants
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Action space: normalized nx, nz, mu in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: 3 relative pos + (speed, psi, gamma) for own + enemy => 9
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        # Action limits for nx, nz, mu for the F-16-like model
        self.nx_limits = [-1.0, 1.5]
        self.nz_limits = [-3.0, 9.0]
        self.mu_limits = [-5 * np.pi / 12, 5 * np.pi / 12]

        self.aircraft1: Aircraft
        self.aircraft2: Aircraft
        self._steps = 0
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self._reset_aircraft()
        obs = self._get_observation()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1

        nx1 = self._scale_action(float(action[0]), self.nx_limits)
        nz1 = self._scale_action(float(action[1]), self.nz_limits)
        mu1 = self._scale_action(float(action[2]), self.mu_limits)

        self._apply_dynamics(nx1, nz1, mu1)

        obs = self._get_observation()
        reward, terminated = self._calculate_reward()
        truncated = self._steps >= self.config.step_limit

        if self.render_mode == "human":
            self.render()

        info: dict[str, Any] = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

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

    # ---- hooks for subclasses ----

    def _reset_aircraft(self) -> None:
        raise NotImplementedError

    def _apply_dynamics(self, nx1: float, nz1: float, mu1: float) -> None:
        """Update aircraft states for one env step."""
        raise NotImplementedError

    def _calculate_reward(self) -> tuple[float, bool]:
        raise NotImplementedError

    # ---- helpers ----

    def _scale_action(self, a: float, limits: list[float]) -> float:
        return limits[0] + (a + 1) * (limits[1] - limits[0]) / 2

    def _get_observation(self) -> np.ndarray:
        rel = [
            (self.aircraft1.x - self.aircraft2.x) / self.config.distance_limit,
            (self.aircraft1.y - self.aircraft2.y) / self.config.distance_limit,
            (self.aircraft1.h - self.aircraft2.h) / self.config.distance_limit,
        ]

        own_enemy = [
            self.aircraft1.v / self.aircraft1.max_speed_limit,
            self.aircraft1.psi / self.aircraft1.psi_limit,
            self.aircraft1.gamma / self.aircraft1.gamma_limit,
            self.aircraft2.v / self.aircraft2.max_speed_limit,
            self.aircraft2.psi / self.aircraft2.psi_limit,
            self.aircraft2.gamma / self.aircraft2.gamma_limit,
        ]

        return np.array(rel + own_enemy, dtype=np.float32)

    def _build_render_states(self) -> list:
        from ..rendering.renderer import AircraftState, BLUE, BLUE_DIM, BLUE_WEZ, RED, RED_DIM, RED_WEZ
        return [
            AircraftState(
                x=self.aircraft1.x, y=self.aircraft1.y, h=self.aircraft1.h,
                v=self.aircraft1.v, psi=self.aircraft1.psi, gamma=self.aircraft1.gamma,
                label="Ownship", colour=BLUE, colour_dim=BLUE_DIM, wez_colour=BLUE_WEZ,
                wez_aperture_deg=self.aircraft1.wez_aperture_deg,
                wez_height_m=self.aircraft1.wez_height_m,
                trail=list(zip(self.aircraft1.log["x"], self.aircraft1.log["y"], self.aircraft1.log["h"])),
            ),
            AircraftState(
                x=self.aircraft2.x, y=self.aircraft2.y, h=self.aircraft2.h,
                v=self.aircraft2.v, psi=self.aircraft2.psi, gamma=self.aircraft2.gamma,
                label="Adversary", colour=RED, colour_dim=RED_DIM, wez_colour=RED_WEZ,
                wez_aperture_deg=self.aircraft2.wez_aperture_deg,
                wez_height_m=self.aircraft2.wez_height_m,
                trail=list(zip(self.aircraft2.log["x"], self.aircraft2.log["y"], self.aircraft2.log["h"])),
            ),
        ]
