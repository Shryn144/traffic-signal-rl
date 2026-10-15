# Traffic Signal Timing Optimizer Using Reinforcement Learning

## About the Project
This project is based on optimizing traffic signals using reinforcement learning. It simulates traffic on a four-road intersection and uses Q-learning to select which road should get the green signal.

## Features
- Simulates traffic on four roads
- Adds vehicles randomly at each step
- Uses Q-learning to select traffic signals
- Compares RL with a fixed-time signal system
- Displays graphs using Matplotlib

## Requirements
- Python
- Matplotlib

## How to Run
Install Matplotlib:

`pip install -r requirements.txt`

Run the project:

`python main.py`

## Results
The program compares the average waiting traffic for the RL agent and the fixed-time signal across multiple trials. The results depend on the random traffic generated during simulation.

## Limitations
The project uses a basic traffic simulation. It does not include real traffic data or all the conditions found at actual road intersections.
