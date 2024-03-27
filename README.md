## Installation instructions:

### Create virtual environment
`python3 -m venv /z/home/natlouis/.virtualenvs/sim_phys_plaus`

### Install packages
`pip install pybullet numpy matplotlib cma pyyaml pyquaternion scipy==1.10.1`

### Include third party dependencies and files 
```
git clone https://github.com/bulletphysics/bullet3.git
ln -s bullet3/data/ data
```

## Test
`python scripts/sandbox.py --useGUI`

## Download and add processed pose data files
```
mkdir datasets
TODO
python scripts/simulate_trajectory.py --lower_feet_to_height=-0.001 --useGUI
```
