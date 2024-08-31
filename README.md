# perSIMS

![image](https://github.com/user-attachments/assets/7316d346-de55-45f1-ae4b-9483f0255f60)

PerSIM is an economic simulation project that models the behavior of a Sim in a virtual environment. The simulation takes into account various needs, economic constraints, and environmental factors to create a dynamic and interactive experience.

## Features

- Simulates a Sim's life in a virtual house
- Manages various needs such as hunger, hygiene, energy, etc.
- Implements a simple economic system with money and market dynamics
- Provides visualization tools for analyzing simulation data

![image](https://github.com/user-attachments/assets/5f63459d-08c5-46d6-9429-a7567398a9f1)

## WIP

- Graphic Visualisers are scratch placeholders
- perSIM sleep logic is curly, and needs simplifying back to core item advertising method
- perSIM can't get stuck due to some competing run condtions (not as often, play with weights)
- Needs long term memory pipeline

## Files

- `config.yaml`: Configuration file for system and user prompts
- `PerSIM.py`: Main simulation script
- `visualise_SIM.py`: Script for generating visualizations from simulation data

## Requirements

- Python 3.7+
- Required Python packages: `pydantic`, `openai`, `matplotlib`, `numpy`, `pandas`, `scipy`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PerSIM.git
   cd PerSIM
   ```

2. Install the required packages:
   ```
   pip install pydantic openai matplotlib numpy pandas scipy
   ```

## Usage

### Running the Simulation

![image](https://github.com/user-attachments/assets/633c3e8f-06ea-4c2b-ad97-7206695e7d56)

To run the simulation, use the following command:

```
python PerSIM.py [options]
```

Options:
- `--model MODEL`: Specify the LLM model to use (default: "hermes3")
- `--responses N`: Generate N responses (default: continuous)
- `--config FILE`: Path to the configuration YAML file (default: "config.yaml")
- `--zeroint`: Run in zero interaction mode (disable LLM calls)
- `--turns N`: Number of turns to run for

Examples:
```
# Run the simulation with default settings
python PerSIM.py

# Run for 100 turns using a specific model
python PerSIM.py --model gpt-3.5-turbo --turns 100

# Run in zero interaction mode for 1000 turns
python PerSIM.py --zeroint --turns 1000
```

### Generating Visualizations

![image](https://github.com/user-attachments/assets/ba550690-621d-4ed4-b062-82ff6201cec6)
![image](https://github.com/user-attachments/assets/a4d19fbc-bcc7-426a-be09-82b81808f69e)

After running the simulation, you can generate visualizations using:

```
python visualise_SIM.py
```

This will create several PNG files in the current directory, including:
- `needs_over_time.png`
- `mood_distribution.png`
- `market_trends.png`
- `activity_duration_distribution.png`
- `sim_3d_trajectory.png`
- `sim_3d_location_density_plot.png`

## Customization

You can modify the `config.yaml` file to adjust the system and user prompts used in the simulation. This allows you to fine-tune the behavior and responses of the Sim.
