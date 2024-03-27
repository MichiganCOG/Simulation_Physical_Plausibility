#Trajectory optimization to perform motion reconstruction
#Use a Stable PD controller and apply an interpolated control target at every time step

import argparse
import os
import sys
DIR = os.getcwd()
sys.path.append(DIR)

import re
import yaml
import json
import pybullet as p
import time
import pybullet_data

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
from tools.utils import *
from tools.data_utils import get_pose_data
from tools.humanoid_utils import get_kin_ref_humanoid, computeCOMposVel

import cma
from cma.fitness_transformations import EvalParallel2
#cma.CMAOptions('termination_call')

import wandb

input_dir = './input_trajectories'
subject  = 'Subject7'
movement = 'Counter_Movement_Jump02'

output_traj_file = 'temp_trajectory.npy'

parser = argparse.ArgumentParser()

parser.add_argument('--pose_format', default='mscoco', help='mscoco, mocap_47, mscoco_foot')
parser.add_argument('--useGUI', action='store_true', help='visualize with GUI')
parser.add_argument('--nCPUs', default=0, type=int, help='Number of CPUs to parallelize optimization')
parser.add_argument('--drawRef', action='store_true', help='draw kinematic reference')

#hyperparameters
parser.add_argument('--sim_freq', default=240, type=int, help='Simulator frequency (Hz), time steps per second')
parser.add_argument('--agent_urdf', default='humanoid/humanoid.urdf', type=str, help='URDF file for humanoid. humanoid/humanoid.urdf or custom_urdf/constrained_humanoid.urdf')
parser.add_argument('--kps', default='1000,100,500,500,400,400,300,500,500,400,400,300', type=str, help='proportional gains')
parser.add_argument('--kds', default='100,10,50,50,40,40,30,50,50,40,40,30', type=str, help='derivative gains')
parser.add_argument('--ctrl_rate', default=10, type=int, help='Control sampling rate for B-Spline. Max of video rate (frames per second)')
parser.add_argument('--sigma', default=0.01, type=float, help='CMA optimizer initial sigma')
parser.add_argument('--popsize', default=100, type=int, help='CMA optimizer population size')
parser.add_argument('--maxiter', default=20, type=int, help='CMA optimizer maximum number of iterations')
parser.add_argument('--w_com', default=15.0, type=float, help='Weight loss for center of mass')
parser.add_argument('--w_comv', default=0.5, type=float, help='Weight loss for velocity of center of mass')
parser.add_argument('--w_com_orn', default=4, type=float, help='Weight loss for center of mass orientation')
parser.add_argument('--w_joint', default=0.5, type=float, help='Weight loss for joints pose')
parser.add_argument('--w_vel', default=0.0005, type=float, help='Weight loss for joint velocities')
parser.add_argument('--w_acc', default=1e-10, type=float, help='Weight loss for joint velocities')
parser.add_argument('--w_feet', default=0, type=float, help='Weight loss for feet on ground')
parser.add_argument('--w_head', default=0, type=float, help='Weight loss for head (neck joint) cartesian position')
parser.add_argument('--w_chest', default=0, type=float, help='Weight loss for chest cartesian position')
parser.add_argument('--w_bos', default=0, type=float, help='Weight loss for base-of-support balance. TODO: Only enforce for stationary points')
parser.add_argument('--max_ext_force', default=0, type=float, help='Max residual forces to optimize on root node. Disabled if zero')
parser.add_argument('--window_length', default=0.5, type=float, help='Optimization window length (seconds)')
parser.add_argument('--num_windows', default=-1, type=int, help='Number of windows to optimize (if less than full length sequence)')
parser.add_argument('--opt_eul', action='store_true', help='Optimize joint angles and velocities in terms of euler angles instead of velocities')
parser.add_argument('--joint_weights', default='1,1,10,1,1,3,1,10,1,1,3,1', type=str, help='Joint weights when optimizing euler angles')
parser.add_argument('--seed', default=999, type=int, help='seed for reproducibility')

parser.add_argument('--height_adjust', default=0, type=float, help='Vertical adjustment for center of mass of humanoid. (To start on floor)')
parser.add_argument('--friction', default=0.9, type=float, help='Friction coefficient of the ground plane')
parser.add_argument('--mass_scale', default=1.0, type=float, help='Scale humanoid mass by this amount to match measured mass')
parser.add_argument('--joints_enforce', default=None, type=str, metavar='LIST', help='(DEPRECATED) Joints indices to constrain in joints pose loss')
parser.add_argument('--floor_pent_reduce', default=0.0, type=float, help='reduce floor peneration by this percentage amount')
parser.add_argument('--lower_feet_to_height', type=float, help='lower the links to height X')
parser.add_argument('--match_init_contact', action='store_true', help='Match initial contact of both feet with motion data')
parser.add_argument('--rotate_ankles', action='store_true', help='Rotate ankles to standard basis, to account for bias default angle.')
parser.add_argument('--const_seg_lens', action='store_true', help='Constrain segment lengths based on averages')

parser.add_argument('--custom_pose', default=None, type=str, help='Optimize a hand crafted pose')
parser.add_argument('--dataset_root', default='/z/home/natlouis/video_grf_pred/data/pybullet/', type=str, help='Root for dataset of source motion files')
parser.add_argument('--dataset', default='forcepose', type=str, help='dataset name')
parser.add_argument('--data_splits', default='train,val', type=str, help='list of split names from dataset')
parser.add_argument('--subject', default='Subject7', type=str, help='subject name')
parser.add_argument('--movement', default='Counter_Movement_Jump02', type=str, help='movement name')
parser.add_argument('--frame_offset', default=0, type=int, help='Offset first frame by this number')

parser.add_argument('--write_video', action='store_true', help='Write out video')
parser.add_argument('--vid_dir', default='./optimized_videos', type=str, help='Output video directory')
parser.add_argument('--video_name', type=str, help='Output video name')

parser.add_argument('--log', action='store_true', help='Log wandb')
parser.add_argument('--exp_name', default='motion_reconstruction', type=str)

parser.set_defaults(useGUI=False)
parser.set_defaults(drawRef=False)
parser.set_defaults(opt_eul=False)
parser.set_defaults(log=False)
parser.set_defaults(write_video=False)
parser.set_defaults(const_seg_lens=False)

args = parser.parse_args()

sim_freq  = args.sim_freq
time_step = 1./sim_freq

useGUI = args.useGUI
draw_ref = args.drawRef
nCPUs  = args.nCPUs
use_presets = False

agent_urdf  = args.agent_urdf
pose_format = args.pose_format

#Controller gains
kps = [int(item) for item in re.split(' |,', args.kps)]
kds = [int(item) for item in re.split(' |,', args.kds)]

ctrl_freq = 1./args.ctrl_rate

#CMA parameters
sigma   = args.sigma
popsize = args.popsize
maxiter = args.maxiter
seed    = args.seed

#Loss weights
w_com     = args.w_com
w_comv    = args.w_comv
w_com_orn = args.w_com_orn
w_joint   = args.w_joint
w_vel     = args.w_vel
w_acc     = args.w_acc
w_feet    = args.w_feet
w_head    = args.w_head
w_chest   = args.w_chest
w_bos     = args.w_bos  

max_ext_force = args.max_ext_force
num_windows   = args.num_windows
opt_eul       = args.opt_eul
opt_quat      = not opt_eul

custom_pose = args.custom_pose
dataset_root = args.dataset_root
dataset = args.dataset
data_splits = [item for item in args.data_splits.split(',')]
subj = args.subject
mvmt = args.movement
frame_offset = args.frame_offset

#height_adj  = args.height_adjust
height_adj  = 0
floor_pent_reduce = args.floor_pent_reduce
lowest_height = args.lower_feet_to_height
match_init_contact = args.match_init_contact
rotate_ankles = args.rotate_ankles
friction = args.friction
mass_scale  = args.mass_scale
const_seg_lens = args.const_seg_lens

#Write video
write_video = args.write_video
if args.video_name is None:
    if custom_pose is None:
        vid_name = subj+'_'+mvmt+'.mp4'
    else:
        vid_name = custom_pose+'.mp4'
else:
    vid_name = args.video_name
os.makedirs(args.vid_dir, exist_ok=True)
vid_options = options="--mp4=\""+args.vid_dir+"/"+vid_name+"\" --mp4fps="+str(sim_freq)

output_yaml_file = args.exp_name+'.yaml'
use_wandb = False
if args.log:
    use_wandb = True
    wandb.init(project='PyBullet', name=args.exp_name, config=args)
    wandb.config.update({'source':subject+'_'+movement,
                         'time_step':time_step,
                         'ctrl_freq':ctrl_freq})
    run_id = wandb.run.id
    output_traj_file = run_id+'.npy'
    output_yaml_file = run_id+'.yaml'

    #Save this file for future reference and config changes
    train_file = os.path.join('scripts','015_human_stable_pd.py') 
    wandb.save(train_file)

#Write out config to YAML file
with open(os.path.join('./log', output_yaml_file), 'w') as f:
    _args = vars(args)
    _args['input_traj'] = os.path.join('./output_trajectories',output_traj_file)
    yaml.dump(_args, f, default_flow_style=False)

pelvis = 0 #root
chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
rightWrist = 8
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
leftWrist = 14

joint_indices = [chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder,\
                rightElbow, leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow]
joint_names   = ['chest', 'neck', 'rightHip', 'rightKnee', 'rightAnkle', 'rightShoulder',\
        'rightElbow', 'leftHip', 'leftKnee', 'leftAnkle', 'leftShoulder', 'leftElbow']
joint_dofs    = [3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1]
joint_forces  = [200., 50., 200., 150., 90., 100., 60., 200., 150., 90., 100., 60.]
maxForces = []

joint_weights = [float(item) for item in args.joint_weights.split(',')]#Weights when computing losses
jws = []

if args.joints_enforce is None:
    joint_constraint = [True]*len(joint_indices) #Apply loss on all joints
else:
    _indices = [int(item) for item in args.joints_enforce.split(',')]
    joint_constraint = []
    for idx in joint_indices:
        if idx in _indices:
            joint_constraint.append(True)
        else:
            joint_constraint.append(False)

for jd, jf, jw in zip(joint_dofs, joint_forces, joint_weights): 
    maxForces.append([jf] * jd)
    jws.extend([jw]*jd)

if useGUI:
    if write_video:
        physicsClient = p.connect(p.GUI, options=vid_options)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    else:
        physicsClient = p.connect(p.GUI) #p.DIRECT for non-graphical version
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
else:
    physicsClient = p.connect(p.DIRECT) #p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0) #Disable mouse picking
p.setGravity(0,-9.8,0)

p.setTimeStep(time_step)
p.setPhysicsEngineParameter(numSubSteps=2)

p.resetDebugVisualizerCamera(cameraDistance=2.0,
                             cameraYaw=-20,
                             cameraPitch=-10,
                             cameraTargetPosition=[0, 0.75, 0])

y2zOrn = p.getQuaternionFromEuler([-np.pi/2, 0, 0]) 
planeId  = p.loadURDF('data/plane.urdf', [0, 0, 0], baseOrientation=y2zOrn)

#Add friction to plane
p.changeDynamics(planeId, linkIndex=-1, lateralFriction=friction)

startPos = [0,0,0]
flags    = p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
humanoid = p.loadURDF(agent_urdf,
                      startPos,
                      globalScaling=0.25,
                      useFixedBase=False,
                      flags=flags)

#Add friction to all humanoid joints
p.changeDynamics(humanoid, linkIndex=-1, lateralFriction=friction)
for j in range(p.getNumJoints(humanoid)):
    p.changeDynamics(humanoid, linkIndex=j, lateralFriction=friction)

#Set damping to zero, for accurate inverse and forward dynamics comparison
p.changeDynamics(humanoid, linkIndex=-1, linearDamping=0, angularDamping=0)

'''
#Reset alpha values to 1.0
alpha = 1.0
p.changeVisualShape(humanoid, -1, rgbaColor=[1, 1, 1, alpha])
for j in range(p.getNumJoints(humanoid)):
    p.changeVisualShape(humanoid, j, rgbaColor=[1, 1, 1, alpha])
'''

#Disables all position + velocity controls to allow for torque control
for j in joint_indices: 
    p.setJointMotorControl2(humanoid,
                            j,  
                            p.POSITION_CONTROL,
                            targetPosition=0,
                            targetVelocity=0,
                            positionGain=0,
                            force=0)

    p.setJointMotorControlMultiDof(humanoid,
                            j,  
                            p.POSITION_CONTROL,
                            targetPosition=[0,0,0,1],
                            targetVelocity=[0, 0, 0], 
                            positionGain=0,
                            velocityGain=1,
                            force=[0, 0, 0])
#Kinematics reference model
humanoid_kin = get_kin_ref_humanoid(p, agent_urdf, startPos, flags, draw_ref)

#Assuming this is the dictionary
joint_types = {0:'joint_revolute',
               1:'joint_prismatic',    
               2:'joint_spherical',    
               3:'joint_planar',    
               4:'joint_fixed'}

numJoints = p.getNumJoints(humanoid)
print(str(numJoints)+' joints')
total_mass = 0
scale = mass_scale
for j in range(numJoints):
    joint_info = p.getJointInfo(humanoid, j)

    joint_index = joint_info[0]
    joint_name  = joint_info[1]
    joint_type  = joint_types[joint_info[2]]
    joint_frict = joint_info[7]
    joint_ll    = joint_info[8]
    joint_ul    = joint_info[9]
    link_name   = joint_info[12]

    joint_mass  = p.getDynamicsInfo(humanoid, j)[0] * scale
    p.changeDynamics(humanoid, j, mass=joint_mass)
    total_mass += joint_mass 

print('Total mass: {}'.format(total_mass))
reset_body(humanoid)
controlMode = p.STABLE_PD_CONTROL

targ_base         = []
targ_base_vel     = []
targ_base_orn     = []
targ_base_orn_vel = []
targ_kin_pose       = []
targ_kin_vel   = []

joint_quats = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1] #quaternion dof for each joint
joint_euler = [3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1] #euler dof for each joint

#T-Pose (Overrides motion capture data)
if custom_pose == 't_pose':
    numFrames = 200
    vid_freq = 1./50

    targ_base     = [(0,0.78,0)]*len(targ_base)
    targ_base_orn = [(0,0,0,1)]*len(targ_base_orn)
    targ_base_vel = [(0,0,0)]*len(targ_base_vel)
    targ_base_orn_vel = [(0,0,0)]*len(targ_base_orn)
    t_pose = []
    for idx in joint_indices:
        if idx == rightShoulder:
            t_pose.extend(p.getQuaternionFromEuler((-1*np.pi/2,0,0)))
        elif idx == leftShoulder:
            t_pose.extend(p.getQuaternionFromEuler((np.pi/2,0,0)))
        else:
            t_pose.extend(p.getJointStateMultiDof(humanoid, idx)[0])
    targ_kin_pose = [t_pose]*len(targ_kin_pose)
#Squat
elif custom_pose == 'squat':
    numFrames = 200
    vid_freq = 1./50

    targ_base     = [(0,0.48,0)]*len(targ_base)
    targ_base_orn = [(0,0,0,1)]*len(targ_base_orn)
    targ_base_vel = [(0,0,0)]*len(targ_base_vel)
    targ_base_orn_vel = [(0,0,0)]*len(targ_base_orn)

    default_kin_pose = []
    default_kin_vel = []
    for idx in joint_indices:
        default_kin_pose.extend(p.getJointStateMultiDof(humanoid, idx)[0])
        default_kin_vel.extend(p.getJointStateMultiDof(humanoid, idx)[1])

    squat_pose = []
    for idx in joint_indices:
        if idx == rightHip:
            squat_pose.extend(p.getQuaternionFromEuler((-0.253,0,1.263)))
        elif idx == leftHip:
            squat_pose.extend(p.getQuaternionFromEuler((0.253,0,1.263)))
        elif idx == rightKnee:
            squat_pose.extend([-1.179])
        elif idx == leftKnee:
            squat_pose.extend([-1.179])
        elif idx == rightShoulder:
            squat_pose.extend(p.getQuaternionFromEuler((-0.211,0,0.884)))
        elif idx == leftShoulder:
            squat_pose.extend(p.getQuaternionFromEuler((0.211,0,0.884)))
        elif idx == rightElbow:
            squat_pose.extend([0.632])
        elif idx == leftElbow:
            squat_pose.extend([0.632])
        else:
            squat_pose.extend(p.getJointStateMultiDof(humanoid, idx)[0])

    #Stand, then Squat
    stop_point = int(0.3 * len(targ_kin_pose))
    stand_pose = [default_kin_pose] * stop_point
    squat_pose = [squat_pose] * (len(targ_kin_pose) - stop_point)
    targ_kin_pose = stand_pose + squat_pose

    targ_kin_vel = [default_kin_vel]*len(targ_kin_vel)
#Motions from DeepMimic
elif custom_pose in ['jump','walk','backflip','cartwheel','spin']:
    with open(os.path.join(pybullet_data.getDataPath(),'data/motions/humanoid3d_'+custom_pose+'.txt'), 'r') as f:
        motion_data = json.load(f)['Frames']

    numFrames = len(motion_data)
    vid_freq  = motion_data[0][0]
    vid_time  = (numFrames-1) * vid_freq

    t = 0 #Simulation time
    cycleCount = 0 #To keep track of loops. Not currently used
    frame = -1
    frameNext = 0

    while frameNext > frame:
        frameTime = t - (cycleCount * vid_time)
        frame = int(frameTime/vid_freq)
        frameNext = min(frame + 1, numFrames-1)

        frameFraction = (frameTime - (frame*vid_freq))/vid_freq
        t += time_step

        targetBasePos, targetBaseOrn, target_positions,\
        baseLinVel, baseAngVel, target_vels = get_pose_from_txt(frame, frameNext, frameFraction, motion_data, motion_data[0][0], return_list=True) 

        targ_base.append(targetBasePos)
        targ_base_orn.append(targetBaseOrn)
        targ_base_vel.append(baseLinVel)
        targ_base_orn_vel.append(baseAngVel)
        targ_kin_pose.append(target_positions)
        targ_kin_vel.append(target_vels)

else:
    subject  = args.subject
    movement = args.movement

    pose_data = get_pose_data(dataset_root, dataset, data_splits, subject=subj, movement=mvmt, pose_format=pose_format)
    pose             = pose_data['kpts']
    keyFrameDuration = pose_data['keyFrameDuration']
    pose = median_filter(pose, 15)
    if const_seg_lens:
        seg_lens = compute_segment_lens(pose, pose_format)
        pose = constrain_segment_lens(pose, seg_lens, pose_format)

    if pose_format == 'mscoco_foot':
        lfoot_idxs = [17,18,19]
        rfoot_idxs = [20,21,22]
    elif pose_format == 'h36m_17':
        lfoot_idxs = [6]
        rfoot_idxs = [3]
    elif pose_format == 'physcap':
        lfoot_idxs = [4]
        rfoot_idxs = [8]
    else: #pose_format=mscoco
        lfoot_idxs = [15] 
        rfoot_idxs = [16] 

    #Try estimating ground plane (not for entire sequence, but only for valid region)
    if num_windows == -1:
        numFrames = len(pose)
    else:
        numFrames = int((num_windows*args.window_length)/keyFrameDuration)  #Expected number of frames
    k = int(numFrames*0.05) #% of frames
    #_pose = np.copy(pose[frame_offset:frame_offset+numFrames])
    _pose = np.copy(pose)
    _p = np.copy(_pose[:,lfoot_idxs+rfoot_idxs]).reshape(-1,3)
    _p = _p[_p[:,1].argsort()][:k]

    lfoot_contact = np.ones(len(_pose))
    rfoot_contact = np.ones(len(_pose))

    floor_height = np.mean(_p, axis=0)[1] #Average of lowest k points
    print('floor height: {}'.format(floor_height))

    #Simple contact estimation baseline (Borrowed from Rempe etal.) 
    #Not in contact if:
    # 1) Heels and toes height > 5cm + (half shape of foot shape height) 
    height_thresh = 0.05
    lfoot_contact[np.min(_pose[:,lfoot_idxs,1],1) > (floor_height + height_thresh)] = 0   
    rfoot_contact[np.min(_pose[:,rfoot_idxs,1],1) > (floor_height + height_thresh)] = 0 

    # 2) Heels and toes distance > 2cm from previous step
    move_thresh = 0.02
    diff = _pose[1:,lfoot_idxs+rfoot_idxs] - _pose[:-1,lfoot_idxs+rfoot_idxs]
    dist = np.linalg.norm(diff,axis=-1)
    l_move = np.concatenate(([False],dist[:,0]>move_thresh))
    r_move = np.concatenate(([False],dist[:,1]>move_thresh))

    lfoot_contact[l_move] = 0
    rfoot_contact[r_move] = 0

    foot_height = 0.055
    #Bring feet to ground plane
    if pose_format == 'mscoco_foot':
        pose[:,:,1] += (floor_height + (foot_height/2))
    else:
        pose[:,:,1] += (floor_height + foot_height)

    numFrames = len(pose)
    #numFrames = 300
    vid_freq  = keyFrameDuration
    vid_time  = (numFrames-1) * vid_freq

    t = 0 #simulation time
    cycleCount = 0 #To keep track of loops. Not currently used
    frame = -1
    frameNext = 0

    #Assume contact is on floor at frame 0
    frameTime = t - (cycleCount * vid_time)
    frame     = int(frameTime/vid_freq)
    frameNext = min(frame + 1, numFrames-1)

    frameFraction = (frameTime - (frame*vid_freq))/vid_freq
    t += time_step

    targetBasePos, targetBaseOrn, target_positions,\
        baseLinVel, baseAngVel, target_vels = get_motion_data(frame, frameNext, frameFraction, pose, file_format='json', pose_format=pose_format, rotate_ankles=rotate_ankles)

    set_body_pos(humanoid, targetBasePos, targetBaseOrn, target_positions, baseLinVel, baseAngVel, target_vels)
    aabbMinL = p.getAABB(humanoid, leftAnkle)[0]
    aabbMinR = p.getAABB(humanoid, rightAnkle)[0]
    aabbMinFoot = min(aabbMinL[1], aabbMinR[1])
    aabbMaxFoot = max(aabbMinL[1], aabbMinR[1])
    height_adj = 0
    if lowest_height:
        if match_init_contact and (lfoot_contact[0] and rfoot_contact[0]): #Enforces both feet have initial ground contact, if in motion data
            if aabbMaxFoot > 0: 
                height_adj = -1*max(aabbMaxFoot - lowest_height, 0) #If above ground
            else:
                height_adj = max(lowest_height - aabbMaxFoot, 0) #If below ground
        else: #At least one foot has initial ground contact
            if aabbMinFoot > 0: 
                height_adj = -1*max(aabbMinFoot - lowest_height, 0) #If above ground
            else:
                height_adj = max(lowest_height - aabbMinFoot, 0) #If below ground
    print(aabbMinL)
    print(aabbMinR)
    print('Decreasing the interpenetration depth by {:.4f}'.format(height_adj))

    while frameNext > frame:
        frameTime = t - (cycleCount * vid_time)
        frame     = int(frameTime/vid_freq) + frame_offset
        frameNext = min(frame + 1, numFrames-1)

        frameFraction = (frameTime - ((frame-frame_offset)*vid_freq))/vid_freq
        t += time_step

        targetBasePos, targetBaseOrn, target_positions,\
            baseLinVel, baseAngVel, target_vels = get_motion_data(frame, frameNext, frameFraction, pose, file_format='json', return_list=True, pose_format=pose_format, rotate_ankles=rotate_ankles)

        targ_base.append(targetBasePos)
        targ_base_orn.append(targetBaseOrn)
        targ_kin_pose.append(target_positions)
        targ_base_vel.append(baseLinVel)
        targ_base_orn_vel.append(baseAngVel)
        targ_kin_vel.append(target_vels)

    set_body_pos(humanoid, targ_base[0], targ_base_orn[0], array_to_joint_list(np.array(targ_kin_pose), joint_quats)[0], targ_base_vel[0], targ_base_orn_vel[0], array_to_joint_list(np.array(targ_kin_vel), joint_quats)[0])

targ_base = np.array(targ_base)
targ_base += [0, height_adj, 0]  

total_steps = len(targ_base)

print(str(numFrames)+' frames. '+str(total_steps)+' total time steps')
vid_sim_ratio   = int(vid_freq/time_step)
vid_ctrl_ratio  = vid_freq/ctrl_freq
sim_ctrl_ratio  = int(ctrl_freq/time_step)

targ_kin_pose_eul = []
#Convert to euler angles
for t in range(total_steps):
    j_idx = 0
    temp = []
    for j_dof in joint_quats:
        if j_dof == 1:
            temp.extend(targ_kin_pose[t][j_idx:j_idx+j_dof])
        else:
            temp.extend(p.getEulerFromQuaternion(targ_kin_pose[t][j_idx:j_idx+j_dof]))
        j_idx += j_dof
    targ_kin_pose_eul.append(temp)

targ_base         = np.array(targ_base)
targ_base_orn     = np.array(targ_base_orn)
targ_base_vel     = np.array(targ_base_vel)
targ_base_orn_vel = np.array(targ_base_orn_vel)

targ_kin_pose     = np.array(targ_kin_pose)
targ_kin_vel = np.array(targ_kin_vel)
targ_kin_pose_eul = np.array(targ_kin_pose_eul)
targ_kin_pose_l = array_to_joint_list(targ_kin_pose, joint_quats)
targ_kin_vel_l = array_to_joint_list(targ_kin_vel, joint_euler)

#Interpolate ground contacts
_lfoot_contact = zoom(lfoot_contact[frame_offset:], vid_sim_ratio) > 0.5
_rfoot_contact = zoom(rfoot_contact[frame_offset:], vid_sim_ratio) > 0.5

#Indices for each windows
win_len   = args.window_length #Real-time length of each window (seconds)
win_steps = int(win_len/time_step)
win_idx   = np.arange(0, total_steps, win_steps)

opt_traj = np.empty((0,np.sum(joint_euler))) #Optimized windows

#Full trajectory metrics
traj_com      = []
traj_kin_com  = []
traj_comv     = []
traj_kin_comv = []

traj_curr_pose = []
traj_curr_vel  = []
traj_curr_acc  = []

traj_targ_pose = []
traj_targ_vel  = []
traj_curr_vel  = []

if num_windows == -1:
    nwin = len(win_idx)
else:
    nwin = num_windows

set_body_pos(humanoid, targ_base[0], targ_base_orn[0], targ_kin_pose_l[0], targ_base_vel[0], targ_base_orn_vel[0], targ_kin_vel_l[0])
aabbMinL,aabbMaxL = p.getAABB(humanoid, leftAnkle)
aabbMinR,aabbMaxR = p.getAABB(humanoid, rightAnkle)
print(aabbMinL)
print(aabbMinR)

print('---'*30)

#for opt_step in range(nwin-1):
for opt_step in range(nwin):
    print('Optimizing window pairs {}/{}'.format(opt_step, nwin))
    if opt_step+1 == len(win_idx): #On last window, repeat optimization previous
        w1 = win_idx[opt_step-1]
        w2 = win_idx[opt_step]
    else:
        w1 = win_idx[opt_step]
        w2 = win_idx[opt_step+1]

    w1_steps = w2-w1
    try:
        w2_steps = win_idx[opt_step+2]-w2
    except IndexError:
        w2_steps = total_steps - w2

    w1_ind = np.arange(w1,min(w1+w1_steps, total_steps),dtype=np.int32)
    w2_ind = np.arange(w2,min(w2+w2_steps, total_steps),dtype=np.int32)
    w_ind  = np.concatenate((w1_ind, w2_ind))

    window1 = targ_kin_pose_eul[w1_ind]
    window2 = targ_kin_pose_eul[w2_ind]

    num_time_steps = len(window1)+len(window2) #num timesteps for both windows
    gt_idx_start = w1
    gt_idx_end   = min(w2+w2_steps, total_steps)

    #Base and kinematic pose targets for the current windows
    win_targ_base         = targ_base[gt_idx_start:gt_idx_end]
    win_targ_base_orn     = targ_base_orn[gt_idx_start:gt_idx_end]
    win_targ_base_vel     = targ_base_vel[gt_idx_start:gt_idx_end]
    win_targ_base_orn_vel = targ_base_orn_vel[gt_idx_start:gt_idx_end]

    win_targ_kin_pose     = array_to_joint_list(targ_kin_pose[gt_idx_start:gt_idx_end], joint_quats)
    win_targ_kin_vel      = euler_to_quaternion(targ_kin_vel[gt_idx_start:gt_idx_end], joint_euler)
    win_targ_kin_pose_eul = targ_kin_pose_eul[gt_idx_start:gt_idx_end] #as flattened array
    win_targ_kin_vel_eul  = targ_kin_vel[gt_idx_start:gt_idx_end] #as flattened array

    win_lfoot_contact = _lfoot_contact[gt_idx_start:gt_idx_end]
    win_rfoot_contact = _rfoot_contact[gt_idx_start:gt_idx_end]

    #Optimize both windows together
    x0 = np.concatenate((window1, window2))

    #Optimize control points (from ctrl freq), not entire trajectory
    ctrl_points = np.linspace(0, num_time_steps-1, int(num_time_steps/sim_ctrl_ratio), dtype=np.int32)
    x0 = x0[ctrl_points]
    x0 = x0.flatten()

    reset_base_pos, reset_base_orn = p.getBasePositionAndOrientation(humanoid)
    reset_base_vel, reset_base_orn_vel = p.getBaseVelocity(humanoid)

    reset_base_pos = np.array(reset_base_pos)
    reset_base_orn = np.array(reset_base_orn).astype('float32')
    reset_base_vel = np.array(reset_base_vel)
    reset_base_orn_vel = np.array(reset_base_orn_vel)

    reset_kin_pose = []
    reset_kin_vel  = []
    for idx in joint_indices:
        reset_kin_pose.append(p.getJointStateMultiDof(humanoid, idx)[0])
        reset_kin_vel.append(p.getJointStateMultiDof(humanoid, idx)[1])

    def func(x, itr):
        x = x.reshape(-1,np.sum(joint_euler))

        #Fit each control target joint to a B-Spline
        intpl_points,_ = get_bspline(x, indices=ctrl_points, num_points=num_time_steps)

        #Convert to quaternions
        tpos = euler_to_quaternion(intpl_points, dofs=joint_euler)

        #Compute target velocities 
        tvels = []
        for t in range(len(tpos)):
            j_idx=0
            tpos0 = []
            tpos1 = []
            for j in range(len(tpos[t])):
                tpos0.append(tpos[t][j])
                try:
                    tpos1.append(tpos[t+1][j])
                except:
                    tpos1.append(tpos[t][j])
            tvels.append(getJointVels(tpos0, tpos1, vid_freq))

        #Initialize pose up to start of window
        set_body_pos(humanoid, reset_base_pos, reset_base_orn, reset_kin_pose, reset_base_vel, reset_base_orn_vel, reset_kin_vel)

        curr_base = []
        curr_base_orn = []
        curr_base_vel = []
        curr_base_orn_vel = []
        curr_kin_pose = []
        curr_kin_vel  = []
        curr_kin_pose_eul = []
        curr_kin_vel_eul  = []

        kin_com   = []
        kin_comv  = []
        curr_com  = []
        curr_comv = []

        kin_head  = []
        curr_head = []

        kin_chest  = []
        curr_chest = []

        curr_lankle_contact = []
        curr_rankle_contact = []

        convex_hull_dists = []
        #Set control targets at simulation frequency
        for t in range(num_time_steps):
            p.setJointMotorControlMultiDofArray(humanoid,
                                                joint_indices,
                                                controlMode,
                                                targetPositions=tpos[t],
                                                targetVelocities=tvels[t],
                                                positionGains=kps,
                                                velocityGains=kds,
                                                forces=maxForces,
                                                )

            set_body_pos(humanoid_kin, win_targ_base[t], win_targ_base_orn[t], win_targ_kin_pose[t], win_targ_base_vel[t], win_targ_base_orn_vel[t], win_targ_kin_vel[t])

            basePos, baseOrn = p.getBasePositionAndOrientation(humanoid)
            baseVel, baseOrnVel = p.getBaseVelocity(humanoid)

            curr_base.append(basePos)
            curr_base_orn.append(baseOrn)
            curr_base_vel.append(baseVel)
            curr_base_orn_vel.append(baseOrnVel)

            #Store center-of-mass values between kinematic reference and optimized humanoid
            k_com, k_comv = computeCOMposVel(p, humanoid_kin)
            com, comv = computeCOMposVel(p, humanoid)

            kin_com.append(k_com)
            curr_com.append(com)

            kin_comv.append(k_comv)
            curr_comv.append(comv)

            kin_head.append(p.getLinkState(humanoid_kin, neck)[0])
            curr_head.append(p.getLinkState(humanoid, neck)[0])

            kin_chest.append(p.getLinkState(humanoid_kin, chest)[0])
            curr_chest.append(p.getLinkState(humanoid, chest)[0])

            #Distance between center of gravity and centroid of convex hull
            #No positive effect observed
            #convex_hull_dists.append(centroid_dist(p, humanoid, leftAnkle, rightAnkle, k_comv))

            kin_pose = []
            kin_vel  = []
            kin_pose_eul = []
            kin_vel_eul  = []
            for idx, dof in zip(joint_indices,joint_dofs):
                js = p.getJointStateMultiDof(humanoid, idx)
                kin_pose.append(js[0])
                kin_vel_eul.extend(js[1])
                if dof == 1:
                    kin_pose_eul.extend(js[0])
                    kin_vel.append(js[1])
                else:
                    kin_pose_eul.extend(p.getEulerFromQuaternion(js[0]))
                    kin_vel.append(p.getQuaternionFromEuler(js[1]))

            curr_kin_pose.append(kin_pose)
            curr_kin_vel.append(kin_vel)
            curr_kin_pose_eul.append(kin_pose_eul)
            curr_kin_vel_eul.append(kin_vel_eul)
            
            #Query axis aligned bounding box around link
            aabbMinL = p.getAABB(humanoid, leftAnkle)[0]
            aabbMinR = p.getAABB(humanoid, rightAnkle)[0]
            curr_lankle_contact.append((aabbMinL[1]<0.0005)*1.0) #Get y-coordinate, note if lower than threshold
            curr_rankle_contact.append((aabbMinR[1]<0.0005)*1.0)

            p.stepSimulation()

            if write_video:
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
            else:
                time.sleep(time_step)

        curr_base = np.array(curr_base)
        curr_base_orn = np.array(curr_base_orn)

        curr_com  = np.array(curr_com)
        kin_com   = np.array(kin_com)
        curr_comv = np.array(curr_comv)
        kin_comv  = np.array(kin_comv)

        kin_head  = np.array(kin_head)
        curr_head = np.array(curr_head)

        kin_chest  = np.array(kin_chest)
        curr_chest = np.array(curr_chest)

        l_com  = np.sum((kin_com - curr_com)**2)
        l_comv = np.sum((kin_comv - curr_comv)**2)

        l_head  = np.sum((kin_head - curr_head)**2)
        l_chest = np.sum((kin_chest - curr_chest)**2)

        l_base = np.sum((win_targ_base - curr_base)**2)
        l_base_vel = np.sum((win_targ_base_vel - curr_base_vel)**2)
        #l_base_orn_vel = np.sum((win_targ_base_orn_vel - curr_base_orn_vel)**2)

        #UNUSED
        #l_bos = np.mean(np.array(convex_hull_dists))
        l_bos = 0

        l_base_orn  = 0
        l_joint     = 0
        l_joint_vel = 0
        l_joint_acc = 0

        for f in range(len(win_targ_base_orn)): #Iterate over frames
            l_base_orn += np.arccos(np.abs(np.clip(np.dot(win_targ_base_orn[f], curr_base_orn[f]),-1,1)))

            if opt_quat:
                #Measure joint angle losses (quaternions)
                for j, (targ_pose, curr_pose, constrain) in enumerate(zip(win_targ_kin_pose[f], curr_kin_pose[f], joint_constraint)):
                    if constrain: #Apply loss only to specified joints
                        if joint_dofs[j] == 1: #Distance for revolute angles is just difference in radians
                            l_joint += np.abs(np.array(targ_pose) - np.array(curr_pose)).item()
                        else:
                            l_joint += np.arccos(np.abs(np.clip(np.array(targ_pose) @ np.array(curr_pose),-1,1)))

                #Measure joint velocities losses (quaternions)
                for j, (targ_vel, curr_vel, constrain) in enumerate(zip(win_targ_kin_vel[f], curr_kin_vel[f], joint_constraint)):
                    if constrain: #Apply loss only to specified joints
                        if joint_dofs[j] == 1: #Distance for revolute angles is just difference in radians
                            l_joint_vel += np.abs(np.array(targ_vel - np.array(curr_vel))).item()
                        else:
                            l_joint_vel += np.arccos(np.abs(np.clip(np.array(targ_vel) @ np.array(curr_vel),-1,1)))

        if opt_eul:
            curr_kin_pose_eul = np.array(curr_kin_pose_eul)
            curr_kin_vel_eul  = np.array(curr_kin_vel_eul)

            l_joint = np.sum(jws * (win_targ_kin_pose_eul - curr_kin_pose_eul)**2)
            l_joint_vel = np.sum(jws * (win_targ_kin_vel_eul - curr_kin_vel_eul)**2)

            #Compute acceleration using finite differences? (For small movements okay, but near 0 and pi rad, could produce issues)
            curr_acc_eul     = (curr_kin_vel_eul[1:]-curr_kin_vel_eul[:-1])/vid_freq
            l_joint_acc      = np.sum(np.linalg.norm(curr_acc_eul,axis=-1)**2)

        loss_com     = w_com * l_com
        loss_comv    = w_comv * l_comv
        loss_com_orn = w_com_orn  * l_base_orn
        loss_joint   = w_joint * l_joint
        loss_vel     = w_vel * l_joint_vel
        loss_acc     = w_acc * l_joint_acc
        loss_head    = w_head * l_head
        loss_chest   = w_chest * l_chest
        loss_bos     = w_bos * l_bos

        if w_feet>0:
            #Simple difference between GT and measured contacts
            loss_lankle = np.sum(np.abs(curr_lankle_contact-win_lfoot_contact))
            loss_rankle = np.sum(np.abs(curr_rankle_contact-win_rfoot_contact))
            loss_feet   = w_feet * (loss_lankle + loss_rankle)
        else:
            loss_feet = 0

        total = loss_com + loss_comv + loss_com_orn + loss_joint + loss_feet + loss_vel + loss_acc + loss_head + loss_chest + loss_bos
        #print('total: {:.3f}, com: {:.3f}, comv: {:.3f}, com_orn: {:.3f}, joint: {:.3f}, vel: {:.3f}'.format(total, loss_com, loss_comv, loss_com_orn, loss_joint, loss_vel))
        #print('curr_base (w1): {}, curr_base_orn (w1): {}'.format(curr_base[w1_steps-1], curr_base_orn[w1_steps-1]))
        #print('curr_base: {}'.format(curr_base))
        #print('--'*30)

        if use_wandb and itr%10==0:
            wandb.log({'COM loss':loss_com,
                       'COMv loss':loss_comv,
                       'COM_orn loss':loss_com_orn,
                       'Joint loss':loss_joint,
                       'feet_loss':loss_feet,
                       'velocity_loss':loss_vel,
                       'acc_loss':loss_acc,
                       'head_loss':loss_head,
                       'chest_loss':loss_chest,
                       'balance_loss':loss_bos})

        return total
    
    #Run CMA Evolution Strategy algorithm to optimize for best gains
    #x, es = cma.fmin2(func, x0, sigma, options={'popsize':popsize, 'maxiter':maxiter})
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize':popsize, 'maxiter':maxiter, 'verb_log':0, 'verb_plot':0, 'seed':seed})

    with EvalParallel2(func, number_of_processes=nCPUs) as eval_all:
        itr=0
        while not es.stop():
            X = es.ask()
            es.tell(X, eval_all(X, args=(itr,)))
            if use_wandb and itr%10==0:
                wandb.log({'Total loss':es.result[1]})
            #es.logger.add()
            #es.disp()
            itr += 1

    final = es.result[0].reshape(-1,np.sum(joint_euler))
    final_traj,_ = get_bspline(final, indices=ctrl_points, num_points=num_time_steps)

    #if opt_step == (nwin-2): #on second to last window, save results of both windows
    #    _opt_traj = final_traj
    #    nsteps = w1_steps+w2_steps
    if opt_step+1 == len(win_idx): #On last window, save results of second window
        _opt_traj = final_traj[-w2_steps:]
        nsteps = w2_steps
    else: #Save results of first window
        _opt_traj = final_traj[:w1_steps]
        nsteps = w1_steps

    opt_traj = np.concatenate((opt_traj, _opt_traj)) 

    #Replay best optimization to reset pose for next window.
    #Measure by using base trajectory alignment
    _opt_traj = euler_to_quaternion(_opt_traj, dofs=joint_euler)
    final_traj = euler_to_quaternion(final_traj, dofs=joint_euler)
    
    _opt_vel = []
    for t in range(len(_opt_traj)):
        j_idx=0
        tpos0 = []
        tpos1 = []
        for j in range(len(_opt_traj[t])):
            tpos0.append(_opt_traj[t][j])
            try:
                tpos1.append(final_traj[t+1][j]) #Pull from full sequence, so target velocity isn't zero at end of window
            except IndexError:
                tpos1.append(_opt_traj[t][j])
        _opt_vel.append(getJointVels(tpos0, tpos1, vid_freq))

    curr_base = []
    curr_base_orn = []
    curr_base_vel = []
    curr_base_orn_vel = []
    set_body_pos(humanoid, reset_base_pos, reset_base_orn, reset_kin_pose, reset_base_vel, reset_base_orn_vel, reset_kin_vel)
    #Reset pose to best optimized trajectory so far
    for t in range(nsteps):
        p.setJointMotorControlMultiDofArray(humanoid,
                                            joint_indices,
                                            controlMode,
                                            targetPositions=_opt_traj[t],
                                            targetVelocities=_opt_vel[t],
                                            positionGains=kps,
                                            velocityGains=kds,
                                            forces=maxForces,
                                            )

        set_body_pos(humanoid_kin, win_targ_base[t], win_targ_base_orn[t], win_targ_kin_pose[t], win_targ_base_vel[t], win_targ_base_orn_vel[t], win_targ_kin_vel[t])

        basePos, baseOrn = p.getBasePositionAndOrientation(humanoid)
        baseVel, baseOrnVel = p.getBaseVelocity(humanoid)

        curr_base.append(basePos)
        curr_base_orn.append(baseOrn)
        curr_base_vel.append(baseVel)

        #Store center-of-mass values between kinematic reference and optimized humanoid
        k_com, k_comv = computeCOMposVel(p, humanoid_kin)
        com, comv = computeCOMposVel(p, humanoid)

        traj_kin_com.append(k_com)
        traj_com.append(com)

        traj_kin_comv.append(k_comv)
        traj_comv.append(comv)

        kin_pose_eul = []
        kin_vel_eul  = []
        for idx, dof in zip(joint_indices,joint_dofs):
            js = p.getJointStateMultiDof(humanoid, idx)
            kin_vel_eul.extend(js[1])
            if dof == 1:
                kin_pose_eul.extend(js[0])
            else:
                kin_pose_eul.extend(p.getEulerFromQuaternion(js[0]))

        traj_curr_pose.append(kin_pose_eul)
        traj_curr_vel.append(kin_vel_eul)

        p.stepSimulation()
        time.sleep(time_step)

    curr_base = np.array(curr_base)
    curr_base_orn = np.array(curr_base_orn)
    l_base = np.sum((win_targ_base[:len(curr_base)] - curr_base)**2)
    l_base_vel = np.sum((win_targ_base_vel[:len(curr_base_vel)] - curr_base_vel)**2)
    print('l_base: {}'.format(l_base))

    l_base_orn  = 0
    loss_feet   = 0
    for f in range(len(curr_base_orn)): #Iterate over frames
        l_base_orn += np.arccos(np.abs(np.clip(np.dot(win_targ_base_orn[f], curr_base_orn[f]),-1,1)))

    l_com_dist  = np.mean(np.linalg.norm(np.array(traj_kin_com)-np.array(traj_com), axis=-1))
    l_joint     = np.sum(jws * (targ_kin_pose_eul[:len(traj_com)] - np.array(traj_curr_pose))**2)
    l_joint_vel = np.sum(jws * (targ_kin_vel [:len(traj_com)] - np.array(traj_curr_vel))**2)

    print('l_com_dist: {}, l_joint: {}, l_joint_vel: {}'.format(l_com_dist, l_joint, l_joint_vel))
    if use_wandb:
        wandb.log({'l_com_dist':l_com_dist, 'opt_step':opt_step})

    out_path = os.path.join('output_trajectories', output_traj_file)
    np.save(out_path, opt_traj)
    print('Saving trajectory ({}) to {}'.format(opt_traj.shape, out_path))
    print('--'*30)

p.disconnect()
