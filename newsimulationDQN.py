import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNAgent to handle action selection and learning
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.00001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=10000)
        
        # Initialize Q-network and target network
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        
        # Synchronize target network with the Q-network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = target_update_freq  # Frequency for updating the target network
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
        # Exploration: choose a random action
            return np.random.choice(self.action_dim)
        else:
            # Exploitation: choose the best action based on Q-values
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(action_values).item()

    def replay(self, batch_size):
        # Training the model and decaying epsilon
        if len(self.memory) < batch_size:
            return None  # Not enough data in memory yet

        # Sample a random mini-batch from memory
        batch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            # Compute the target Q-value using the target network
            with torch.no_grad():
                target_q_value = reward + (1 - done) * self.gamma * torch.max(self.target_model(next_state)).item()
        
            # Current Q-value for the selected action
            current_q_value = self.model(state)[action]

            # Compute loss
            loss = self.loss_fn(current_q_value, torch.tensor(target_q_value))
        
            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            total_loss += loss.item()

        # Decay epsilon after each replay to gradually reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
        # Return the average loss over the batch
        return total_loss / batch_size if batch else None

    
    
class Simulation:
    def __init__(self, taxi_data_path, vehicles_data_path, grid_size=(20, 20), episode_duration=40):
        
        
        # Load taxi data
        self.taxi_data = pd.read_csv(taxi_data_path, delimiter=';')
        self.taxi_data['tpep_pickup_datetime'] = pd.to_datetime(self.taxi_data['tpep_pickup_datetime'])
        self.taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(self.taxi_data['tpep_dropoff_datetime'])

        # Load vehicle data
        self.vehicles_data = pd.read_csv(vehicles_data_path)
        self.location_to_cluster = {vehicle['Nodelocation']: vehicle['Cluster_ID'] for _, vehicle in self.vehicles_data.iterrows()}
        
        # Map Cluster_ID in taxi data based on PULocationID
        self.taxi_data['Cluster_ID'] = self.taxi_data['PULocationID'].map(self.location_to_cluster)

        # Initialize grid
        self.grid_size = grid_size
        self.episode_duration = episode_duration  # Set to 40 minutes per episode (4 timesteps)
        self.timestep_duration = 10               # Each timestep is 10 minutes
        self.timesteps_per_episode = self.episode_duration // self.timestep_duration
     
        # Initialize vehicle grids
        self.idle_vehicle_grid = np.zeros(grid_size)
        self.vehicle_status = ['idle'] * len(self.vehicles_data)
        self.service_timers = [0] * len(self.vehicles_data)
        for position in self.vehicles_data['Cluster_ID']:
            row, col = eval(position)
            self.idle_vehicle_grid[row, col] += 1

        # Initialize vehicle assignments DataFrame
        self.vehicle_assignments_df = pd.DataFrame(columns=['Vehicle_ID', 'Order_ID', 'Assigned_Cluster_ID', 'Dropoff_Cluster_ID'])

        # Initialize time
        self.start_time = self.taxi_data['tpep_pickup_datetime'].min()
        self.current_time = self.start_time
        self.agent = DQNAgent(state_dim=19, action_dim=9)  # 19 state features, 9 possible actions
        self.rewards_per_episode = []
        self.losses = []
        self.rejection_rates = []
    def get_neighbors(self, row, col):
        neighbors = []
        for r in range(max(0, row - 1), min(self.grid_size[0], row + 2)):
            for c in range(max(0, col - 1), min(self.grid_size[1], col + 2)):
                if (r, c) != (row, col):
                    neighbors.append((r, c))
        return neighbors

    def get_state(self, row, col, pending_order_grid):
        state = [row * self.grid_size[1] + col, pending_order_grid[row, col], self.idle_vehicle_grid[row, col]]
        neighbors = self.get_neighbors(row, col)
        for n_row, n_col in neighbors:
            state.extend([pending_order_grid[n_row, n_col], self.idle_vehicle_grid[n_row, n_col]])
        while len(state) < 19:
            state.extend([0, 0])
        return np.array(state)
    
    def dispatch_orders(self, pending_order_grid, timestep_orders):
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                initial_state = self.get_state(row, col, pending_order_grid)
                print(f"Initial State of cluster {row},{col}: {initial_state}")

                # Process orders in current cluster
                while pending_order_grid[row, col] > 0:
                    min_distance = None  # Set default value for min_distance

                    if self.idle_vehicle_grid[row, col] > 0:
                        # Assign vehicle within the same cluster if available
                        self.idle_vehicle_grid[row, col] -= 1
                        pending_order_grid[row, col] -= 1
                    
                        assigned_order = timestep_orders.iloc[0]
                        timestep_orders = timestep_orders.iloc[1:]
                        trip_duration = (assigned_order['tpep_dropoff_datetime'] - assigned_order['tpep_pickup_datetime']).total_seconds() / 60.0

                        # Select the first available idle vehicle in the same cluster
                        idle_vehicles_idx = np.where((np.array(self.vehicle_status) == 'idle') & 
                                                 (self.vehicles_data['Cluster_ID'] == f"({row},{col})"))[0]
                    
                        if len(idle_vehicles_idx) == 0:
                            # No idle vehicles in the same cluster, search neighbors based on Manhattan distance
                            neighbors = self.get_neighbors(row, col)
                            closest_vehicle_idx = None
                            min_distance = float('inf')

                            for n_row, n_col in neighbors:
                                # Calculate Manhattan distance
                                distance = abs(row - n_row) + abs(col - n_col)
                                neighbor_idle_vehicles_idx = np.where((np.array(self.vehicle_status) == 'idle') & 
                                                                  (self.vehicles_data['Cluster_ID'] == f"({n_row},{n_col})"))[0]
                            
                                # Select the closest idle vehicle based on Manhattan distance
                                if len(neighbor_idle_vehicles_idx) > 0 and distance < min_distance:
                                    closest_vehicle_idx = neighbor_idle_vehicles_idx[0]
                                    min_distance = distance
                        
                            vehicle_idx = closest_vehicle_idx
                            if vehicle_idx is None:
                                print(f"No idle vehicles available for order in cluster {row},{col} or neighbors.")
                                break
                        else:
                            # Assign the closest idle vehicle within the same cluster
                            vehicle_idx = idle_vehicles_idx[0]
                            min_distance = 0  # Vehicle is in the same cluster, distance is zero
                    
                     # Assign the vehicle and update status
                        self.vehicle_status[vehicle_idx] = 'busy'
                        self.service_timers[vehicle_idx] = trip_duration

                        self.vehicle_assignments_df = self.vehicle_assignments_df.append({
                            'Vehicle_ID': vehicle_idx,
                            'Order_ID': assigned_order.name,
                            'Assigned_Cluster_ID': (row, col),
                            'Dropoff_Cluster_ID': 'N/A'
                        }, ignore_index=True)

                        print(f"Vehicle {vehicle_idx} assigned to order in cluster {row},{col} with Manhattan distance {min_distance} and will be busy for {trip_duration} minutes.")
                        # Display the next state after dispatching
                        next_state = self.get_state(row, col, pending_order_grid)
                        print(f"Next State of cluster {row},{col} after dispatching: {next_state}")
                    
                    else:
                        print(f"No idle vehicles available for order in cluster {row},{col}.")
                        break


    def update_vehicle_status(self, timestep_duration):
        for i in range(len(self.vehicles_data)):
            if self.vehicle_status[i] == 'busy':
                remaining_service_time = self.service_timers[i]
                self.service_timers[i] -= min(timestep_duration, remaining_service_time)
                
                if self.service_timers[i] <= 0:  # Trip complete
                    # Get the vehicle's last assigned order from vehicle_assignments_df
                    last_order = self.vehicle_assignments_df[self.vehicle_assignments_df['Vehicle_ID'] == i].iloc[-1]
                    assigned_order_id = last_order['Order_ID']

                    # Now use the assigned_order_id to get the actual order from self.taxi_data
                    assigned_order = self.taxi_data.loc[assigned_order_id]
                    DOLocationID = assigned_order['DOLocationID']
                    new_cluster_id = self.location_to_cluster.get(DOLocationID, None)
                    
                    if new_cluster_id is not None:
                        self.vehicle_status[i] = 'idle'
                        old_cluster = self.vehicles_data.at[i, 'Cluster_ID']
                        self.vehicles_data.at[i, 'Cluster_ID'] = new_cluster_id

                        if isinstance(old_cluster, str):
                            row, col = eval(old_cluster)
                        else:
                            row, col = old_cluster

                        if self.idle_vehicle_grid[row, col] > 0:
                            self.idle_vehicle_grid[row, col] -= 1

                        if isinstance(new_cluster_id, str):
                            row, col = eval(new_cluster_id)
                        else:
                            row, col = new_cluster_id

                        self.idle_vehicle_grid[row, col] += 1

                        # Update the vehicle_assignments_df
                        self.vehicle_assignments_df.loc[self.vehicle_assignments_df['Vehicle_ID'] == i, 'Dropoff_Cluster_ID'] = new_cluster_id

                        print(f"Vehicle {i} has completed the trip and moved to Cluster {new_cluster_id}, now idle.")
                    else:
                        print(f"Warning: DOLocationID {DOLocationID} not mapped to a Cluster_ID.")
    def calculate_reward(self, action, row, col, pending_order_grid):
        current_demand = pending_order_grid[row, col]
        current_supply = self.idle_vehicle_grid[row, col]

        if action == 0:  # Action 0: Stay in current cluster
            if current_demand > current_supply:
                reward = 5
            else:
                reward = 0
        else:
            # Check if the selected action corresponds to a valid neighbor
            neighbors = self.get_neighbors(row, col)
            if action - 1 < len(neighbors):
                n_row, n_col = neighbors[action - 1]
                neighbor_demand = pending_order_grid[n_row, n_col]
                neighbor_supply = self.idle_vehicle_grid[n_row, n_col]

               
                if current_demand > current_supply:
                    reward = -5  # Moved out when there's high demand
                elif neighbor_demand > neighbor_supply and current_demand < current_supply:
                    reward = 2  # Moves to high demand neighbor
                else:
                    reward = 0  # No specific incentive
            else:
                reward = 0  # Default reward if action does not correspond to a valid neighbor

        return reward 

    def train(self, episodes=200):
        for episode in range(episodes):
            # Initialize or reset the pending orders grid for each episode
            self.pending_order_grid = np.random.randint(0, 5, size=self.grid_size)
            total_reward = 0
            total_loss = 0
            step_count = 0
            print(f"\nStarting Episode {episode + 1}")

            # Loop through each timestep in an episode
            for t in range(self.timesteps_per_episode):
                timestep_reward = 0
                for row in range(self.grid_size[0]):
                    for col in range(self.grid_size[1]):
                        # Get current state and choose action
                        state = self.get_state(row, col, self.pending_order_grid)
                        action = self.agent.act(state)
                        # Calculate reward and get the next state
                        reward = self.calculate_reward(action, row, col, self.pending_order_grid)
                        timestep_reward += reward
                        next_state = self.get_state(row, col, self.pending_order_grid)
                        self.agent.remember(state, action, reward, next_state, done=False)

                # Update DQN and track the average loss over replay
                loss = self.agent.replay(64)
                if loss is not None:
                    total_loss += loss
                    step_count += 1

                total_reward += timestep_reward
                print(f"Timestep {t + 1}/{self.timesteps_per_episode} - Reward: {timestep_reward}")

            # Store metrics for each episode
            self.rewards_per_episode.append(total_reward)
            self.losses.append(total_loss / step_count if step_count > 0 else 0)

            # Calculate and record rejection rate for each episode
            total_demand = np.sum(self.pending_order_grid)
            total_supply = np.sum(self.idle_vehicle_grid)
            rejection_rate = (total_demand - total_supply) / total_demand if total_demand > 0 else 0
            self.rejection_rates.append(rejection_rate * 100)

            # Print total reward and rejection rate for the episode
            print(f"Episode {episode + 1} - Total Reward: {total_reward}, Rejection Rate: {rejection_rate * 100:.2f}%")

            if self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= self.agent.epsilon_decay

    
    
                   
    def run(self):
        for step in range(self.timesteps_per_episode):
            timestep_start_time = self.current_time
            timestep_end_time = self.current_time + pd.Timedelta(minutes=self.timestep_duration)
            print(f"\nProcessing simulation in timestep {step + 1} from {timestep_start_time} to {timestep_end_time}")

            # Orders that occur in this timestep
            timestep_orders = self.taxi_data[
                (self.taxi_data['tpep_pickup_datetime'] >= timestep_start_time) &
                (self.taxi_data['tpep_pickup_datetime'] < timestep_end_time)
            ]
            total_pending_orders = len(timestep_orders)
            print(f"Total pending orders in this timestep: {total_pending_orders}")

            # Create a grid of pending orders for this timestep
            pending_order_grid = np.zeros(self.grid_size)

            for _, order in timestep_orders.iterrows():
                cluster_id = order['Cluster_ID']
                if pd.notna(cluster_id):
                    row, col = eval(cluster_id)
                    pending_order_grid[row, col] += 1
                else:
                    print(f"Warning: PULocationID {order['PULocationID']} not mapped to a Cluster_ID.")
            #self.visualize_idle_vehicles_grid(timestep_start_time, timestep_end_time)
            #self.visualize_pending_orders_grid(pending_order_grid, timestep_start_time, timestep_end_time)
            # Dispatch orders and assign vehicles
            self.dispatch_orders(pending_order_grid, timestep_orders)
            #self.visualize_idle_vehicles_grid_after_dispatching(timestep_start_time, timestep_end_time)
            #self.visualize_pending_orders_grid_after_dispatching(pending_order_grid, timestep_start_time, timestep_end_time)
            self.update_vehicle_status(self.timestep_duration)
            self.current_time += pd.Timedelta(minutes=self.timestep_duration)

        # Output final vehicle assignment data
        self.vehicle_assignments_df.to_csv("vehicle_assignments_after_simulation.csv", index=False)


    def visualize_idle_vehicles_grid(self, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))
        idle_vehicles_heatmap = ax.imshow(self.idle_vehicle_grid, cmap='Blues', interpolation='nearest', alpha=0.6)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idle_count = int(self.idle_vehicle_grid[i, j])
                ax.text(j, i, f'{idle_count}', ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Idle Vehicles before dispatching from {timestep_start_time} to {timestep_end_time}')

        fig.colorbar(idle_vehicles_heatmap, ax=ax, orientation='vertical', label='Idle Vehicles')
        plt.show()

    def visualize_pending_orders_grid(self, pending_order_grid, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Heatmap for pending orders (red)
        pending_orders_heatmap = ax.imshow(pending_order_grid, cmap='Reds', interpolation='nearest', alpha=0.6)

        # Annotate each cell with the number of pending orders
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pending_count = int(pending_order_grid[i, j])
                ax.text(j, i, f'{pending_count}', ha='center', va='center', color='black')

        # Set grid labels and titles
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Pending Orders before dispatching from {timestep_start_time} to {timestep_end_time}')

        # Add colorbar
        fig.colorbar(pending_orders_heatmap, ax=ax, orientation='vertical', label='Pending Orders')

        plt.show()
    def visualize_idle_vehicles_grid_after_dispatching(self, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))
        idle_vehicles_heatmap = ax.imshow(self.idle_vehicle_grid, cmap='Blues', interpolation='nearest', alpha=0.6)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idle_count = int(self.idle_vehicle_grid[i, j])
                ax.text(j, i, f'{idle_count}', ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Idle Vehicles after dispatching from {timestep_start_time} to {timestep_end_time}')

        fig.colorbar(idle_vehicles_heatmap, ax=ax, orientation='vertical', label='Idle Vehicles')
        plt.show()

    def visualize_pending_orders_grid_after_dispatching(self, pending_order_grid, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Heatmap for pending orders (red)
        pending_orders_heatmap = ax.imshow(pending_order_grid, cmap='Reds', interpolation='nearest', alpha=0.6)

        # Annotate each cell with the number of pending orders
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pending_count = int(pending_order_grid[i, j])
                ax.text(j, i, f'{pending_count}', ha='center', va='center', color='black')

        # Set grid labels and titles
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Pending Orders after dispatching from {timestep_start_time} to {timestep_end_time}')

        # Add colorbar
        fig.colorbar(pending_orders_heatmap, ax=ax, orientation='vertical', label='Pending Orders')

        plt.show()
    def plot_results(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.rewards_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        
        plt.subplot(1, 3, 2)
        plt.plot(self.rejection_rates)
        plt.xlabel("Episodes")
        plt.ylabel("Rejection Rate")
        plt.title("Rejection Rate per Episode")

        plt.subplot(1, 3, 3)
        plt.plot(self.losses)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss per Episode")
        
        plt.tight_layout()
        plt.show()
    def plot_training_progress(self):
        # Plot rewards
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.rewards_per_episode, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")

        # Plot losses
        plt.subplot(1, 3, 2)
        plt.plot(self.losses, label="Loss", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.title("Loss per Episode")

        # Plot rejection rates
        plt.subplot(1, 3, 3)
        plt.plot(self.rejection_rates, label="Rejection Rate", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Rejection Rate (%)")
        plt.title("Rejection Rate per Episode")

        plt.tight_layout()
        plt.show()

def main():
    # Redirect stdout to a text file for logging
    sys.stdout = open('simulation_output2.txt', 'w')

    taxi_data_path = 'C:/Users/hp/Desktop/new project/v1_30_10/data/taxi_data.csv'
    vehicles_data_path = 'C:/Users/hp/Desktop/new project/v1_30_10/vehicles_data.csv'

    
    # Create an instance of the Simulation class
    sim = Simulation(taxi_data_path, vehicles_data_path, grid_size=(20, 20), episode_duration=60)
    # Run the simulation
    sim.run()
    sim.train(episodes=200)

    sim.plot_training_progress()

    # Close the output file
    sys.stdout.close()

# Call the main function if this script is executed
if __name__ == "__main__":
    main()


  