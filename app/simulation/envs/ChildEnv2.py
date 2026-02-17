import numpy as np
import gymnasium as gym
from gymnasium import spaces
from app.simulation.envs.Env import Env
from app.domain.Customer import Customer

class ChildEnv2(Env):
    MAX_QUEUE_SIZE = 50 
    NUM_FEATURES = 8

    def __init__(self, mode, instance=None, scenario=None):
        super().__init__(mode, instance, scenario)
        self.state_buffer_ids = []
        
        self.global_task_averages = {}
        if instance:
            if isinstance(instance.average_matrix, list):
                matrix_iter = enumerate(instance.average_matrix)
            else:
                matrix_iter = instance.average_matrix.items()

            for task_id, server_times in matrix_iter:
                if isinstance(server_times, dict):
                    durations = list(server_times.values())
                else:
                    durations = server_times
                
                valid_durations = [d for d in durations if d > 0]
                if valid_durations:
                    self.global_task_averages[int(task_id)] = np.mean(valid_durations)
                else:
                    self.global_task_averages[int(task_id)] = 60.0

    def _get_action_space(self):
        return spaces.Discrete(self.MAX_QUEUE_SIZE + 1)
    
    def _get_observation_space(self):
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.MAX_QUEUE_SIZE, self.NUM_FEATURES), 
            dtype=np.float32
        )
    
    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        current_server = servers[selected_server_id]

        upcoming_appt_times = [
            appt.time for appt in appointments.values() 
            if appt.time > sim_time
        ]
        if upcoming_appt_times:
            next_appt_time = min(upcoming_appt_times)
            time_to_next_appt = (next_appt_time - sim_time) / 60.0
        else:
            time_to_next_appt = 1.0

        candidates = list(waiting_customers.values())
        
        #BUCKET SORTING (15-min Buckets)
        def get_priority_score(c):
            # Death Time
            if c.id in appointments:
                death_time = appointments[c.id].time + 30.0
            else:
                death_time = c.arrival_time + 60.0
            
            # Bucket: Group by 15-minute deadlines
            # This allows Duration to break ties for patients with similar urgency
            urgency_bucket = int(death_time / 15.0)
            
            # Secondary: Duration (Shortest Job First)
            duration = current_server.avg_service_time.get(c.task, 60.0)
            
            # Tuple sorting: Bucket first, then Duration
            return (urgency_bucket, duration)
            
        candidates.sort(key=get_priority_score)
        self.state_buffer_ids = [c.id for c in candidates][:self.MAX_QUEUE_SIZE]
        
        queue_pressure = len(candidates) / 50.0

        obs = np.zeros((self.MAX_QUEUE_SIZE, self.NUM_FEATURES), dtype=np.float32)
        
        for i, customer_id in enumerate(self.state_buffer_ids):
            c = waiting_customers[customer_id]
            
            wait_time = sim_time - c.arrival_time
            norm_wait = wait_time / 60.0
            
            is_appt = 0.0
            time_to_deadline = 0.0
            if customer_id in appointments:
                is_appt = 1.0
                appt = appointments[customer_id]
                time_to_deadline = (appt.time - sim_time) / 60.0
            else:
                time_to_deadline = (60.0 - wait_time) / 60.0

            task_feat = c.task / 10.0
            can_serve = 1.0 if current_server.avg_service_time.get(c.task, 0) > 0 else 0.0
            avg_duration = current_server.avg_service_time.get(c.task, 60.0) / 60.0

            obs[i] = [norm_wait, is_appt, time_to_deadline, task_feat, can_serve, avg_duration, time_to_next_appt, queue_pressure]
            
        return obs
    
    def _get_customer_from_action(self, action) -> Customer:
        if action >= len(self.state_buffer_ids):
            return None 
        customer_id = self.state_buffer_ids[action]
        return self.customer_waiting.get(customer_id, None)

    def _get_invalid_action_reward(self) -> float: 
        return -10.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        sim_time = self.system_time
        reward = 10.0 
        
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            target_time = appt.time
            dt = sim_time - target_time 
            
            epsilon = 3
            max_early = 60
            max_late = 30 
            
            score = 0.0
            if abs(dt) <= epsilon:
                score = 100.0
            elif dt < -epsilon and dt > -max_early:
                score = 100 * (1 + (dt + epsilon) / (max_early - epsilon))
                # Penalize "Too Early" serves to reinforce the mask
                if dt < -40:
                    reward -= 5.0 
            elif dt > epsilon and dt < max_late:
                score = 100 / (max_late - epsilon) * (target_time - sim_time + max_late)
            reward += 0.4 * score
        else:
            wait_time = sim_time - customer.arrival_time
            max_wait = 60.0
            score = 0.0
            if wait_time < max_wait:
                score = 100 * (1 - wait_time / max_wait)
            reward += 0.4 * score
        return reward
    
    def action_masks(self):
        mask = [False] * (self.MAX_QUEUE_SIZE + 1)
        mask[self.MAX_QUEUE_SIZE] = True
        
        current_server = self.current_working_server
        sim_time = self.system_time
        
        for i, customer_id in enumerate(self.state_buffer_ids):
            customer = self.customer_waiting.get(customer_id)
            if customer:
                if current_server.avg_service_time.get(customer.task, 0) > 0:
                    
                    if customer.id in self.appointments:
                        # RUTHLESS: Strict 30-min window for appointments
                        appt_time = self.appointments[customer.id].time
                        if (appt_time - sim_time) > 30.0:
                            mask[i] = False
                        else:
                            mask[i] = True
                    else:
                        # RUTHLESS: Sacrifice walk-ins > 30 mins
                        wait_time = sim_time - customer.arrival_time
                        if wait_time > 30.0:
                            mask[i] = False
                        else:
                            mask[i] = True
        return mask
    
    def _get_hold_action_number(self):
        return self.MAX_QUEUE_SIZE
