from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from .base import BaseAirCombatEnv, EnvConfig
from ..models import Aircraft

# Default model: best-performing SAC checkpoint from evaluation
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "pretrained", "opponent_sac"
)


class PretrainedOpponentEnv(BaseAirCombatEnv):
    """1v1 dogfight where the adversary is a pretrained SAC agent.

    The user controls ``aircraft1`` (ownship).  ``aircraft2`` (adversary)
    is controlled by a frozen Stable-Baselines3 SAC policy loaded from
    disk.

    The default opponent is ``sac_dofight_1M_v2`` — the best-performing
    model from benchmark evaluation (8/20 WEZ hits, 18/20 survival rate).

    Args:
        config: Environment configuration.
        render_mode: Gymnasium render mode.
        model_path: Path to a SB3-compatible ``.zip`` model file
            (without extension).  Defaults to the bundled
            ``sac_dofight_1M_v2`` checkpoint.

    Example::

        from air_combat_gym import PretrainedOpponentEnv

        env = PretrainedOpponentEnv()
        obs, info = env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        model_path: str | None = None,
    ):
        super().__init__(config=config, render_mode=render_mode)

        # Lazy import to avoid hard dependency on SB3 at package level
        try:
            from stable_baselines3 import SAC
        except ImportError as e:
            raise ImportError(
                "PretrainedOpponentEnv requires stable-baselines3. "
                "Install it with:  pip install stable-baselines3"
            ) from e

        path = model_path or _DEFAULT_MODEL_PATH
        self._opponent_model = SAC.load(path)

    def _reset_aircraft(self) -> None:
        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0, 0)
        self.aircraft2 = Aircraft(0, 1000, 1000, 250, np.pi, 0)

    def _apply_dynamics(self, nx1: float, nz1: float, mu1: float) -> None:
        # Ownship: user-controlled
        self.aircraft1.update(nx1, nz1, mu1)

        # Adversary: pretrained SAC policy
        opp_obs = self._opponent_obs()
        opp_action, _ = self._opponent_model.predict(opp_obs, deterministic=True)

        nx2 = self._scale_action(float(opp_action[0]), self.nx_limits)
        nz2 = self._scale_action(float(opp_action[1]), self.nz_limits)
        mu2 = self._scale_action(float(opp_action[2]), self.mu_limits)
        self.aircraft2.update(nx2, nz2, mu2)

    def _opponent_obs(self) -> np.ndarray:
        """Build a 9-D observation from the opponent's perspective."""
        rel = [
            (self.aircraft2.x - self.aircraft1.x) / self.config.distance_limit,
            (self.aircraft2.y - self.aircraft1.y) / self.config.distance_limit,
            (self.aircraft2.h - self.aircraft1.h) / self.config.distance_limit,
        ]
        own = [
            self.aircraft2.v / self.aircraft2.max_speed_limit,
            self.aircraft2.psi / self.aircraft2.psi_limit,
            self.aircraft2.gamma / self.aircraft2.gamma_limit,
        ]
        opp = [
            self.aircraft1.v / self.aircraft1.max_speed_limit,
            self.aircraft1.psi / self.aircraft1.psi_limit,
            self.aircraft1.gamma / self.aircraft1.gamma_limit,
        ]
        return np.array(rel + own + opp, dtype=np.float32)

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
