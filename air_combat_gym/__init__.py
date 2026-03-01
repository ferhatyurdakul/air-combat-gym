"""Air Combat Framework for RL.

Gymnasium environments built on a simple 3-DoF aircraft model.
"""

from .envs.dogfight_1v1 import Dogfight1v1Env
from .envs.dogfight_1vn import Dogfight1vNEnv
from .envs.target_circular import CircularTargetFollowEnv
from .envs.adversary_random import RandomAdversaryDogfightEnv
from .envs.self_play import SelfPlayDogfightEnv
from .envs.pretrained_opponent import PretrainedOpponentEnv
from .envs.registry import ENV_REGISTRY, make_env

__all__ = [
    "Dogfight1v1Env",
    "Dogfight1vNEnv",
    "CircularTargetFollowEnv",
    "RandomAdversaryDogfightEnv",
    "SelfPlayDogfightEnv",
    "PretrainedOpponentEnv",
    "ENV_REGISTRY",
    "make_env",
]
