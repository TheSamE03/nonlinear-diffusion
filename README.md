# Nonlinear Diffusion Simulation
This simulation solves the diffusion equation in 2 dimentions with various prebuilt initial and boundary conditions which can be changed using the built in UI. 

## Setup Instructions

### Prerequisites
- Python
- Pip

### For windows
Open a powershell terminal emulator window. 

To download the necessary files run the following command

```git clone https://github.com/TheSamE03/nonlinear-diffusion.git```

Open the newly installed dirtectory

```cd nonlinear-diffusion```

Install dependencies

```pip install requirements.txt```

Run the simulation

```python 2DSim.py```

You will be prompted to chose your desired initial conditions, boundary conditions, and some other parameters. Once these are selected click "ok" and the simulation will begin solving the PDE at each grid point for each timestep. This is an intensive process and the capabilities of the host computer will be a key factor in efficiency of the simulation. 

## Hardware Considerations

If you have a computer with 8GB of RAM or less, or a CPU older than intel 12th generation, or Ryzen 7th generation, you should consider modifying the parameter "N" near the top of the 2DSim.py file to something like 50 or 75 instead of the default 100. This is the "resolution" of the simulation. This number represents the number of discrete gridpoints calculated in each direction and therefor the ammount of computing needed for each timestep of the simulation is proportional to the square of this value. 

The decreased number of gridpoints can be compensated with smoothing in an interpreting software such as wolfram mathematica for similar results to higher detail simulations. 
