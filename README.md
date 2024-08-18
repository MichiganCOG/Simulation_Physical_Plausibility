## Installation instructions:

### Create virtual environment
`python3 -m venv sim_phys_plaus`

### Install packages
`pip install pybullet numpy matplotlib cma pyyaml pyquaternion scipy==1.10.1`

### Include third party dependencies and files 
```
git clone https://github.com/bulletphysics/bullet3.git
ln -s bullet3/data/ data
```

## Test
`python scripts/sandbox.py --useGUI`

## Download pose detections (stored as dict in .npy files)
- [pre-processed H36M detections](https://prism.eecs.umich.edu/natlouis/sim_phys_plaus/datasets/pybullet/h36m_25fps.tar.gz)
- [pre-processed ForcePose detections](https://prism.eecs.umich.edu/natlouis/sim_phys_plaus/datasets/pybullet/forcepose_1.1.tar.gz)
```
mkdir datasets
tar -xvf h36m_25fps.tar.gz -C ./datasets
tar -xvf forcepose_1.1.tar.gz -C ./datasets
```

## Simulate raw motion trajectory
`python scripts/simulate_trajectory.py --lower_feet_to_height=-0.001 --useGUI`

## Simulate optimized motion trajectory
`python scripts/simulate_trajectory.py --cfg_file=./configs/h36m_S11_Walking_1.yaml`
`python scripts/simulate_trajectory.py --cfg_file=./configs/h36m_S19_Waiting_1.yaml`

## Optimize motion trajectory
`python scripts/optimize_trajectory --nCPUs=40 --opt_eul --match_init_contact --subject=$SUBJ --movement=$MVMT --exp_name='$SUBJ_$MVMT'`

## Other useful options
- Save to video: `--useGUI --write_video --video_name=$VID_NAME.mp4`
- Draw kinematic reference: `--useGUI --drawRef`

