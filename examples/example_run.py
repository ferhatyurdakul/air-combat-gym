import time
from air_combat_gym import PretrainedOpponentEnv
from air_combat_gym.envs.base import EnvConfig
import pygame

env = PretrainedOpponentEnv(EnvConfig(step_limit=500), render_mode="human")

NUM_EPISODES = 3

for ep in range(NUM_EPISODES):
    obs, info = env.reset(seed=ep)
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        # Random actions for ownship
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    print(f"Episode {ep + 1}/{NUM_EPISODES}: {step} steps, reward: {total_reward:.1f}")
    time.sleep(1.0)

print("\nDone! Close the window or press ESC.")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

env.close()
