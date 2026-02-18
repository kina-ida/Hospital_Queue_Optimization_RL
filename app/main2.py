from app.simulation.policies.ChildPolicy2 import ChildPolicy2 
from app.data.Instance import Instance
import gymnasium as gym
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
from gymnasium.envs.registration import register
from app.data.Scenario import Scenario
from app.simulation.envs.Env import Env

register(
    id="Child_Env_2",
    entry_point="app.simulation.envs.ChildEnv2:ChildEnv2", 
)

def main():
    scenario = Scenario.from_json("app/data/config/queue_config.json")
    
    model = ChildPolicy2("PPO_Solution") 

    model.learn(scenario, 100000, 1)
    
    model.model.save("ppo_2") 

    print("Training finished and saved ppo_2.zip")

    instance = Instance.create(Instance.SourceType.FILE,
                    "app/data/data_files/timeline_0.json", 
                    "app/data/data_files/average_matrix_0.json",
                    "app/data/data_files/appointments_0.json", 
                    "app/data/data_files/unavailability_0.json")
    
    # update the test environment ID
    env = gym.make("Child_Env_2", mode=Env.MODE.TEST, instance=instance)
    
    model.simulate(env, print_logs=True, save_to_csv=True, path="app/data/results2/", file_name="result_0.csv")
    policy_evaluation = PolicyEvaluation(instance.timeline, instance.appointments, clients_history=model.customers_history)
    policy_evaluation.evaluate()

if __name__ == "__main__":
    main()