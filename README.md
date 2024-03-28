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

## Download and include pose detections (stored as dict in .npy files)
- [pre-processed H36M detections](https://prism.eecs.umich.edu/natlouis/sim_phys_plaus/datasets/pybullet/h36m_25fps.tar.gz)
- [pre-processed ForcePose detections](https://prism.eecs.umich.edu/natlouis/sim_phys_plaus/datasets/pybullet/forcepose_1.1.tar.gz)
```
mkdir datasets
tar -xvf h36m_25fps.tar.gz -C ./datasets
tar -xvf forcepose_1.1.tar.gz -C ./datasets
```

## Test simulate trajectory (not optimized) on ForcePose
`python scripts/simulate_trajectory.py --lower_feet_to_height=-0.001 --useGUI`
