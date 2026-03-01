from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..models import Aircraft
from .base import EnvConfig


class SelfPlayDogfightEnv(gym.Env):
    """Multi-agent 1v1 dogfight where **two independent agents** each control
    one aircraft.

    Designed for self-play or multi-agent RL training.

    Action
        Per-agent: 3-D normalised ``[nx, nz, mu]`` in [-1, 1].

    Observation
        Per-agent: 9-D — 3 relative-position features + 3 own kinematic
        features + 3 opponent kinematic features (same layout as single-agent
        envs, but from each aircraft's perspective).

    Usage (multi-agent dict API)::

        env = SelfPlayDogfightEnv()
        obs = env.reset()                        # {"agent_0": ..., "agent_1": ...}
        obs, rewards, dones, truncs, infos = env.step({
            "agent_0": action_a0,
            "agent_1": action_a1,
        })

    Usage (flat 6-D action for backward compat)::

        obs, rewards, dones, truncs, infos = env.step(np.concatenate([a0, a1]))

    The flat API still returns dicts — unwrap with ``obs["agent_0"]``.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}
    AGENT_IDS = ("agent_0", "agent_1")

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Per-agent spaces
        self._single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Gymnasium expects these on the env; expose the joint versions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agent_0": self._single_obs_space,
            "agent_1": self._single_obs_space,
        })

        self.nx_limits = [-1.0, 1.5]
        self.nz_limits = [-3.0, 9.0]
        self.mu_limits = [-5 * np.pi / 12, 5 * np.pi / 12]

        self.aircraft1: Aircraft
        self.aircraft2: Aircraft
        self._steps = 0
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0

        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0.0, 0.0)
        self.aircraft2 = Aircraft(0, 1000, 1000, 250, np.pi, 0.0)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Accept either a dict ``{"agent_0": a0, "agent_1": a1}`` or a flat
        6-D array ``[nx0, nz0, mu0, nx1, nz1, mu1]``.
        """
        self._steps += 1

        a0, a1 = self._parse_action(action)

        nx1 = self._scale(float(a0[0]), self.nx_limits)
        nz1 = self._scale(float(a0[1]), self.nz_limits)
        mu1 = self._scale(float(a0[2]), self.mu_limits)

        nx2 = self._scale(float(a1[0]), self.nx_limits)
        nz2 = self._scale(float(a1[1]), self.nz_limits)
        mu2 = self._scale(float(a1[2]), self.mu_limits)

        self.aircraft1.update(nx1, nz1, mu1)
        self.aircraft2.update(nx2, nz2, mu2)

        obs = self._get_obs()
        r0, r1, terminated = self._reward()
        truncated = self._steps >= self.config.step_limit

        if self.render_mode == "human":
            self.render()

        rewards = {"agent_0": float(r0), "agent_1": float(r1)}
        dones = {"agent_0": bool(terminated), "agent_1": bool(terminated)}
        truncs = {"agent_0": bool(truncated), "agent_1": bool(truncated)}
        infos = {"agent_0": {}, "agent_1": {}}

        return obs, rewards, dones, truncs, infos

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

    def _build_render_states(self) -> list:
        from ..rendering.renderer import AircraftState, BLUE, BLUE_DIM, BLUE_WEZ, RED, RED_DIM, RED_WEZ
        return [
            AircraftState(
                x=self.aircraft1.x, y=self.aircraft1.y, h=self.aircraft1.h,
                v=self.aircraft1.v, psi=self.aircraft1.psi, gamma=self.aircraft1.gamma,
                label="Agent 0", colour=BLUE, colour_dim=BLUE_DIM, wez_colour=BLUE_WEZ,
                wez_aperture_deg=self.aircraft1.wez_aperture_deg,
                wez_height_m=self.aircraft1.wez_height_m,
                trail=list(zip(self.aircraft1.log["x"], self.aircraft1.log["y"], self.aircraft1.log["h"])),
            ),
            AircraftState(
                x=self.aircraft2.x, y=self.aircraft2.y, h=self.aircraft2.h,
                v=self.aircraft2.v, psi=self.aircraft2.psi, gamma=self.aircraft2.gamma,
                label="Agent 1", colour=RED, colour_dim=RED_DIM, wez_colour=RED_WEZ,
                wez_aperture_deg=self.aircraft2.wez_aperture_deg,
                wez_height_m=self.aircraft2.wez_height_m,
                trail=list(zip(self.aircraft2.log["x"], self.aircraft2.log["y"], self.aircraft2.log["h"])),
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_action(self, action) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(action, dict):
            return np.asarray(action["agent_0"]), np.asarray(action["agent_1"])
        action = np.asarray(action).flatten()
        return action[:3], action[3:]

    def _scale(self, a: float, limits: list[float]) -> float:
        return limits[0] + (a + 1) * (limits[1] - limits[0]) / 2

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "agent_0": self._obs_for(self.aircraft1, self.aircraft2),
            "agent_1": self._obs_for(self.aircraft2, self.aircraft1),
        }

    def _obs_for(self, own: Aircraft, opp: Aircraft) -> np.ndarray:
        rel = [
            (own.x - opp.x) / self.config.distance_limit,
            (own.y - opp.y) / self.config.distance_limit,
            (own.h - opp.h) / self.config.distance_limit,
        ]
        own_kin = [
            own.v / own.max_speed_limit,
            own.psi / own.psi_limit,
            own.gamma / own.gamma_limit,
        ]
        opp_kin = [
            opp.v / opp.max_speed_limit,
            opp.psi / opp.psi_limit,
            opp.gamma / opp.gamma_limit,
        ]
        return np.array(rel + own_kin + opp_kin, dtype=np.float32)

    def _reward(self) -> tuple[float, float, bool]:
        """Zero-sum reward. Returns (reward_agent0, reward_agent1, terminated)."""
        terminated = False
        r0 = 0.0

        if self.aircraft1.wez(self.aircraft2.x, self.aircraft2.y, self.aircraft2.h):
            r0 += 100.0
        if self.aircraft2.wez(self.aircraft1.x, self.aircraft1.y, self.aircraft1.h):
            r0 -= 100.0

        distance = float(
            np.sqrt(
                (self.aircraft1.x - self.aircraft2.x) ** 2
                + (self.aircraft1.y - self.aircraft2.y) ** 2
                + (self.aircraft1.h - self.aircraft2.h) ** 2
            )
        )
        dist_shaping = 1.0 - distance / self.config.distance_limit
        r0 += dist_shaping

        if distance > self.config.distance_limit:
            terminated = True

        # Zero-sum: agent_1 reward is the negative of agent_0
        return r0, -r0, terminated
