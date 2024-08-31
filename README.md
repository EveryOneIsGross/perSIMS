# perSIMS

![image](https://github.com/user-attachments/assets/9d7591c6-507f-4637-8601-677ec31c28db)

- **Current state:** I'm standing in our tiny bathroom, mentally preparing for the refreshing cascade of warm water that awaits me in the shower. I take a deep breath to wash away the stress of another day of monotonous work ahead.
- **Needs assessment:** My body is calling for more than just the basic needs of nourishment and rest; it's yearning for something else – a respite from the incessant routine, perhaps. Warm water on tired skin seems like a small but significant indulgence.
- **Emotional check:** As I stand here, a palpable mix of trepidation and anticipation washes over me. Part of me wants nothing more than to prolong this fleeting moment of peace before diving headlong into another workday. The other part is aware that soon enough, my brain will be consumed by familiar strains of worry and boredom.
- **Short-term plan:** My immediate plan is to fully immerse myself in the therapeutic pleasure of a good shower, allowing the water to cleanse not just my body, but also my mind, even if it's only temporary respite from the world outside these walls.
- **Long-term considerations:** As I allow myself this moment of solace, my thoughts veer toward the future – will today find me more resilient? More focused? Or will the patterns of my life continue their predictable dance to nowhere new?

>The Sim takes a deep breath before stepping under the warm, steady stream of water cascading from above. There's a sense of contentment as it allows itself this brief reprieve - an acknowledgment that even in monotony, there can often be moments of solace waiting... if one knows where to look and how to appreciate them fully. - sum perSIM


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
- [noted] I forgot to update ticker at each pos instead of each item pos. Now moves more dynamically.

---

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
python PerSIM.py --model hermes3 --turns 100

# Run in zero interaction mode for 1000 turns
python PerSIM.py --zeroint --turns 1000
```

---

![image](https://github.com/user-attachments/assets/34ec8f40-1d6f-4279-a1d3-de54252be15b)
![image](https://github.com/user-attachments/assets/f6b29ba0-b770-435d-8f27-034f1e8c60c9)

### Generating Visualizations

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
