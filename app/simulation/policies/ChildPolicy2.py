from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym
from stable_baselines3 import PPO
import os
import numpy as np

class ChildPolicy2(Policy):
    def __init__(self, model_title):
        super().__init__(model_title)
        self.model = None
        self.model_filename = "ppo_2" 

        save_path = self.model_filename + ".zip"
        if os.path.exists(save_path):
            print(f" Found saved model: {save_path}. Loading it now...")
            self.model = PPO.load(self.model_filename)
        else:
            print(" No saved model found. Agent will start random.")

    def _predict(self, obs, info):
        action_mask = info.get('action_mask')
        valid_indices = [i for i, m in enumerate(action_mask) if m]

        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)

            if action_mask[action]:
                return action
            else:
                if valid_indices:
                    return valid_indices[0] # Pick the Smart Sorted #1 option
                return len(action_mask) - 1 
        
        # If untrained, picking 0 is now a good strategy due to sorting
        if valid_indices:
            return valid_indices[0]
        
        return len(action_mask) - 1

    def learn(self, scenario, total_timesteps, verbose):
        learning_env = gym.make("Child_Env_2", mode=Env.MODE.TRAIN, scenario=scenario)
        
        # 256x256 is good
        policy_kwargs = dict(net_arch=[256, 256])
        
        self.model = PPO(
            "MlpPolicy", 
            learning_env, 
            verbose=verbose,
            policy_kwargs=policy_kwargs, 
            learning_rate=0.0003,
            n_steps=2048, 
            batch_size=64,
            gamma=0.99
        )
        
        print(f"Starting training for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)
        print(" Training complete.")
        
        self.model.save(self.model_filename)
        print(f"Model saved to {self.model_filename}.zip")