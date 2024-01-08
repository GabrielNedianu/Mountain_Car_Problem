---

### How to Run the Python Notebook

This notebook demonstrates the implementation of a Q-learning agent to solve the Mountain Car problem. Follow the steps below to get started.

#### Prerequisites

Before running the notebook, ensure you have the following:
- Python 3
- Jupyter Notebook

#### Running the Notebook

1. **Clone the Repository**: Clone the repository containing the notebook to your local machine:

   ```bash
   git clone [URL of the repository]
   cd [repository name]
   ```

2. **Launch Jupyter Notebook**: Open a terminal in the repository directory and start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. **Open the Notebook**: In your browser, navigate to the notebook file (`.ipynb`) and open it.

4. **Install Dependencies**: The notebook contains cells that automatically install required packages (`numpy`, `ipywidgets`, `matplotlib`). Simply run these cells to install the dependencies.

   ```python
   import sys
   !{sys.executable} -m pip install numpy
   !{sys.executable} -m pip install ipywidgets matplotlib
   ```

5. **Run the Notebook**: Execute the cells by selecting each cell and pressing `Shift + Enter` or the icon. Alternatively, run all cells at once from the menu.

#### Interacting with the Notebook

- **Default Agent**: To run the simulation with default parameters run the following cells:

  ```python
  # Usage with default values
  default_agent = MountainCarAgent()
  default_agent.train()

  # Run a simple test with 100 episodes
  success_rate = default_agent.test_policy(num_episodes=100)
  print(f"Success rate with 1000 episodes: {success_rate * 100:.2f}%")
  ```

- **Custom Configuration**: To run the simulation with custom parameters, adjust the `config` dictionary and run the cell:

  ```python
  # Usage with custom values
  
  config = {
      'alpha': 0.1,
      'gamma': 0.99,
      'epsilon': 0.3,
      'epsilon_decay': 0.9,
      'min_epsilon': 0.01,
      'num_episodes': 5001,
      'num_states': [20,20]
  }
  
  print(f"Testing configuration: {config}")
  
  custom_agent = MountainCarAgent(**config)
  custom_agent.train()
  
  # Uncomment this to save the q-table analysys
  #custom_agent.plot_and_save_Q_table_analysis(100)
  
  success_rate = custom_agent.test_policy(num_episodes = 100)
  print(f"Success rate of the learned policy: {success_rate * 100:.2f}%")
  
  # Uncomment this to run and save the graphics for an analyzed test
  #analysis_data = custom_agent.test_policy_analyzed(num_episodes = 100)
  #custom_agent.plot_analysis_data(analysis_data)
  #print(f"Success Rate: {analysis_data['success_rate'] * 100:.2f}%, "
  #               f"Average Episode Length: {np.mean(analysis_data['episode_lengths']):.2f} steps, "
  #               f"Min Episode Length: {np.min(analysis_data['episode_lengths'])} steps, "
  #               f"Average Reward: {np.mean(analysis_data['episode_rewards']):.2f}, "
  #               f"Failures: {len(analysis_data['failures'])}, "
  #               f"Total Episodes: {analysis_data['episodes']}")
  ```

#### Troubleshooting

If you encounter issues:
- Ensure all required packages are installed.
- Check for error messages.
- Restart the Jupyter Notebook kernel if necessary.

### Mountain Car Problem Simulation with Q-learning

This Python class represents an agent that uses Q-learning to solve the Mountain Car problem. The agent operates in a discretized state space and learns an optimal policy over time.

#### Class: `MountainCarAgent`

#### Methods

---

##### `__init__(self, alpha=0.1, gamma=0.99, epsilon=0.15, ...)`
_Initializes the agent with learning parameters._

- **Parameters**:
  - `alpha`: Learning rate.
  - `gamma`: Discount factor.
  - `epsilon`: Exploration rate.
  - `epsilon_decay`: Rate at which exploration decreases.
  - `min_epsilon`: Minimum exploration rate.
  - `num_episodes`: Number of training episodes.
  - `num_states`: Number of discrete states in each dimension.
- **Functionality**: Sets up the environment, initializes the Q-table, and prepares for training.

---

##### `discretize_state(self, state)`
_Converts a continuous state into a discretized form._

- **Parameters**:
  - `state`: The continuous state to discretize.
- **Functionality**: Discretizes each dimension of the state into a predefined grid.

---

##### `epsilon_greedy_policy(self, state)`
_Chooses an action based on the epsilon-greedy policy._

- **Parameters**:
  - `state`: The current state of the agent.
- **Functionality**: Balances exploration and exploitation based on `epsilon`.

---

##### `create_directory(self, dir_path)`
_Creates a directory if it doesn't exist._

- **Parameters**:
  - `dir_path`: Path of the directory to create.
- **Functionality**: Ensures the existence of the specified directory.

---

##### `save_Q_table(self, filename, folder="mountain_car/q_tables")`
_Saves the current Q-table to a file._

- **Parameters**:
  - `filename`: Name of the file.
  - `folder`: Directory to save the file.
- **Functionality**: Saves the Q-table for later use or analysis.

---

##### `load_Q_table(self, filename, folder="mountain_car/q_tables")`
_Loads a Q-table from a file._

- **Parameters**:
  - `filename`: Name of the file.
  - `folder`: Directory of the file.
- **Functionality**: Loads a pre-trained Q-table.

---

##### `train(self)`
_Trains the agent using Q-learning._

- **Functionality**: Updates the Q-table based on interactions with the environment.

---

##### `plot_and_save_Q_table_analysis(self, test_id=0, ...)`
_Generates and saves plots analyzing the Q-table._

- **Parameters**:
  - `test_id`: Identifier for the test.
  - `folder`: Directory to save the plots.
- **Functionality**: Creates various plots for analyzing the learning process.

---

##### `test_policy(self, num_episodes=1000, max_steps=1000)`
_Tests the trained policy over a number of episodes._

- **Parameters**:
  - `num_episodes`: Number of test episodes.
  - `max_steps`: Maximum steps per episode.
- **Functionality**: Evaluates the performance of the trained policy.

---

##### `test_policy_analyzed(self, num_episodes=10000, max_steps=1000)`
_Tests the policy and gathers detailed analysis data._

- **Functionality**: Collects comprehensive data for further analysis.

---

##### `plot_analysis_data(self, analysis_data, test_id=0, ...)`
_Generates and saves plots based on the analysis data._

- **Parameters**:
  - `analysis_data`: Data collected during testing.
  - `test_id`: Identifier for the test.
  - `folder`: Directory to save the plots.
- **Functionality**: Visualizes the performance and behavior of the agent.

---

##### `test_policy_with_visualization(self, num_episodes=5)`
_Visualizes the agent's policy for a specified number of episodes._

- **Parameters**:
  - `num_episodes`: Number of episodes to visualize.
- **Functionality**: Shows the agent's decisions in the environment visually.

---

##### `custom_render(self, step, action_taken)`
_Custom function to visualize the state of the environment._

- **Parameters**:
  - `step`: Current step number.
  - `action_taken`: The action taken by the agent.
- **Functionality**: Creates a visual representation of the current state and action.

---

### Overview

This class provides a complete framework for an agent to learn and solve the Mountain Car problem using Q-learning. It includes functionalities for discretizing the state space, implementing epsilon-greedy policy for action selection, training the agent, and analyzing and visualizing its performance.

---
