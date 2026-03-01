from __future__ import annotations

from typing import Callable

from .adversary_random import RandomAdversaryDogfightEnv
from .dogfight_1v1 import Dogfight1v1Env
from .dogfight_1vn import Dogfight1vNEnv
from .self_play import SelfPlayDogfightEnv
from .pretrained_opponent import PretrainedOpponentEnv
from .target_circular import CircularTargetFollowEnv


ENV_REGISTRY: dict[str, Callable[..., object]] = {
    "dogfight_1v1": Dogfight1v1Env,
    "dogfight_1vn": Dogfight1vNEnv,
    "target_circular": CircularTargetFollowEnv,
    "adversary_random": RandomAdversaryDogfightEnv,
    "self_play": SelfPlayDogfightEnv,
    "pretrained_opponent": PretrainedOpponentEnv,
}


def make_env(env_id: str, **kwargs):
    """Create an environment by id.

    Example:
        env = make_env("dogfight_1v1")
    """
    if env_id not in ENV_REGISTRY:
        raise KeyError(f"Unknown env_id={env_id!r}. Available: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_id](**kwargs)
