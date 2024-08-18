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
import pybullet_data #Root: bullet3/examples/pybullet/gym/pybullet_data

from scipy import interpolate
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
from tools.utils import *
from tools.data_utils import get_pose_data
from tools.humanoid_utils import get_kin_ref_humanoid, computeCOMposVel, has_fallen, is_balanced

parser = argparse.ArgumentParser()

parser.add_argument('--sim_freq', type=int, help='Simulator frequency (Hz), time steps per second')
parser.add_argument('--agent_urdf', type=str, help='URDF file for humanoid. humanoid/humanoid.urdf or custom_urdf/constrained_humanoid.urdf')
parser.add_argument('--pose_format', help='mscoco, mscoco_foot, mocap_47')
parser.add_argument('--useGUI', action='store_true', help='visualize with GUI')
parser.add_argument('--loops', type=int, help='Loops in playpack')
parser.add_argument('--plot_kin', action='store_true', help='plot kinematic angle targets')
parser.add_argument('--plot_vel', action='store_true', help='plot velocity targets')
parser.add_argument('--plot_torque', action='store_true', help='plot applied joint torques')
parser.add_argument('--plot_force', action='store_true', help='plot measured contact forces') 
parser.add_argument('--override_ctrl', action='store_true', help='Override controller for kinematic reference only')
parser.add_argument('--drawRef', action='store_true', help='draw kinematic reference')

#Outputting files and videos
parser.add_argument('--save_metrics', action='store_true', help='Save above metrics into external file')
parser.add_argument('--met_group_name', default='reference', help='Group name for set of settings for output metrics.')
parser.add_argument('--write_video', action='store_true', help='Write out video')
parser.add_argument('--vid_dir', default='./output_videos', type=str, help='Output video directory')
parser.add_argument('--video_name', type=str, help='Output video name')

#Configuration file
parser.add_argument('--cfg_file', default=None, help='Configuration file to load all optimization and dataloading. Overwrites cmd line args')

#Settings from optimization
parser.add_argument('--kps', type=str, help='proportional gains')
parser.add_argument('--kds', type=str, help='derivative gains')
parser.add_argument('--input_traj', type=str, help=' Use specified saved trajectory')
parser.add_argument('--window_length', type=float, help='Optimization window length (seconds)')
parser.add_argument('--plane_adjust', type=float, help='Vertical adjustment to ground plane, so humanoid starts on floor.')
parser.add_argument('--friction', type=float, help='Friction coefficient of the ground plane')
parser.add_argument('--mass_scale', type=float, help='Scale humanoid mass by this amount to match measured mass')
parser.add_argument('--floor_pent_reduce', type=float, help='reduce floor peneration by this percentage amount')
parser.add_argument('--lower_feet_to_height', type=float, help='lower the links to height X')
parser.add_argument('--match_init_contact', action='store_true', help='Match initial contact of both feet with motion data')
parser.add_argument('--rotate_ankles', action='store_true', help='Rotate ankles to standard basis, to account for bias default angle.')
parser.add_argument('--const_seg_lens', action='store_true', help='Constrain segment lengths based on averages')

#Dataloading settings
parser.add_argument('--custom_pose',  type=str, help='Optimize a hand crafted pose')
parser.add_argument('--dataset_root', type=str, help='Root for dataset of source motion files')
parser.add_argument('--dataset',      type=str, help='dataset name')
parser.add_argument('--data_splits',  type=str, help='list of split names from dataset')
parser.add_argument('--subject',      type=str, help='subject name')
parser.add_argument('--movement',     type=str, help='movement name')
parser.add_argument('--frame_offset', type=int, help='Offset first frame by this number')

parser.set_defaults(useGUI=False)
parser.set_defaults(override_ctrl=False)
parser.set_defaults(drawRef=False)
parser.set_defaults(plot_kin=False)
parser.set_defaults(plot_vel=False)
parser.set_defaults(write_video=False)

cmd_args,_ = parser.parse_known_args()
cmd_args = vars(cmd_args)

defaults = {
        'sim_freq':240,
        'agent_urdf':'humanoid/humanoid.urdf',
        'kps':'1000,100,500,500,400,400,300,500,500,400,400,300',
        'kds':'100,10,50,50,40,40,30,50,50,40,40,30',
        'window_length':0.5,
        'dataset_root':'./datasets',
        'dataset':'h36m_25fps',
        'data_splits':'train,val',
        'subject':'S11',
        'movement':'Walking_1',
        'frame_offset':0,
        'plane_adjust':0,
        'friction':0.9,
        'mass_scale':1.0,
        'floor_pent_reduce':0,
        'lower_feet_to_height':None,
        'match_init_contact':False,
        'rotate_ankles':False,
        'pose_format':'mscoco',
        'loops':1,
        }
#Load configuration file args, if exists
if cmd_args['cfg_file'] is not None:
    with open(cmd_args['cfg_file'], 'r') as f:
        yaml_args = yaml.safe_load(f)
else:
    yaml_args = {}

#Update args from YAML file, cmd line, or default
for (k,v) in cmd_args.items():
    if v is None or v is False:
        if k in yaml_args and k not in ['write_video']: #Do not force write video
            cmd_args[k] = yaml_args[k]
        else:
            cmd_args[k] = defaults.get(k, None)

print('--'*30)
print('Run arguments')
for k in sorted(cmd_args.keys()):
    print('{}: {}'.format(k,cmd_args[k]))
print('--'*30)

sim_freq  = cmd_args['sim_freq']
time_step = 1./sim_freq

useGUI = cmd_args['useGUI']
loops  = cmd_args['loops']
override_ctrl = cmd_args['override_ctrl']
draw_ref = cmd_args['drawRef']
plot_kin = cmd_args['plot_kin']
plot_vel = cmd_args['plot_vel']
plot_force = cmd_args['plot_force']
plot_torque = cmd_args['plot_torque']
save_metrics = cmd_args['save_metrics']
met_group    = cmd_args['met_group_name']
use_presets = False

agent_urdf  = cmd_args['agent_urdf']
pose_format = cmd_args['pose_format']

#Controller gains
kps = [int(item) for item in re.split(' |,', cmd_args['kps'])]
kds = [int(item) for item in re.split(' |,', cmd_args['kds'])]

custom_pose = cmd_args['custom_pose']
dataset_root = cmd_args['dataset_root']
dataset = cmd_args['dataset']
data_splits = [item for item in cmd_args['data_splits'].split(',')]
subj = cmd_args['subject']
mvmt = cmd_args['movement']
plane_adj  = cmd_args['plane_adjust']
friction = cmd_args['friction']
mass_scale = cmd_args['mass_scale']
floor_pent_reduce = cmd_args['floor_pent_reduce']
lowest_height = cmd_args['lower_feet_to_height']
match_init_contact = cmd_args['match_init_contact']
rotate_ankles = cmd_args['rotate_ankles']
const_seg_lens = cmd_args['const_seg_lens']
frame_offset = cmd_args['frame_offset']

#Input trajectory
input_traj = cmd_args['input_traj']
loadTrajectory = False
if not input_traj is None and os.path.isfile(input_traj):
    loadTrajectory = True

#Write video
write_video = cmd_args['write_video']
if cmd_args['video_name'] is None:
    if loadTrajectory:
        if custom_pose is None:
            vid_name = subj+'_'+mvmt+'_'+input_traj.split('/')[-1].split('.')[0]+'.mp4'
        else:
            vid_name = custom_pose+'_'+input_traj.split('/')[-1].split('.')[0]+'.mp4'
    else:
        vid_name = 'reference_'+subj+'_'+mvmt+'.mp4'
else:
    vid_name = cmd_args['video_name']
os.makedirs(cmd_args['vid_dir'], exist_ok=True)
vid_options = options="--mp4=\""+cmd_args['vid_dir']+"/"+vid_name+"\" --mp4fps="+str(sim_freq)

root = 0 #pelvis
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
#Joints that should never touch the ground (may change depending on motion) TODO: Alternatively look at base of support error
fall_indices  = [root, chest, neck, rightHip, rightKnee, rightShoulder,\
                rightElbow, rightWrist, leftHip, leftKnee, leftShoulder, leftElbow, leftWrist]

joint_names   = ['chest', 'neck', 'rightHip', 'rightKnee', 'rightAnkle', 'rightShoulder',\
                'rightElbow', 'leftHip', 'leftKnee', 'leftAnkle', 'leftShoulder', 'leftElbow']
joint_dofs    = [3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1]
joint_forces  = [200., 50., 200., 150., 90., 100., 60., 200., 150., 90., 100., 60.]
maxForces = []

joint_weights = [1, 1, 10, 1, 1, 3, 1, 10, 1, 1, 3, 1] #Weights when computing losses
jws = []

for jd, jf, jw in zip(joint_dofs, joint_forces, joint_weights): 
    maxForces.append([jf] * jd)
    jws.extend([jw]*jd)

kpOrg = [
        0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500, 400,
        400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400, 400, 400,
        400, 400, 300
    ]
kdOrg = [
    0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40, 40,
    40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
]

def main():
    if useGUI:
        if write_video:
            physicsClient = p.connect(p.GUI, options=vid_options)
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        else:
            physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

    else:
        physicsClient = p.connect(p.DIRECT) #p.DIRECT for non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0) #Disable mouse picking
    p.setGravity(0,-9.8,0)

    p.setTimeStep(time_step)
    p.setPhysicsEngineParameter(numSubSteps=2)

    #Cam 54138969: cameraYaw: 160 cameraPitch: -10 cameraDistance: 3.25
    #Cam 58860488: cameraYaw: 270  cameraPitch: -10  cameraDistance: 1.25
    #Cam 60457274: cameraYaw: -20 cameraPitch: -10 cameraDistance: 2.5
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=-30,
                                 cameraPitch=-5,
                                 cameraTargetPosition=[0, 0.75, -1.3])

    y2zOrn = p.getQuaternionFromEuler([(-90*(np.pi/180)), 0, (0*(np.pi/180))]) 
    planeId  = p.loadURDF('data/plane.urdf', [0, plane_adj, 0], baseOrientation=y2zOrn)

    startPos = [0,0.85,0]
    flags    = p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    humanoid = p.loadURDF(agent_urdf,
                          startPos,
                          globalScaling=0.25,
                          useFixedBase=False,
                          flags=flags)

    #Change plane friction to 0.9
    p.changeDynamics(planeId, linkIndex=-1, lateralFriction=friction)

    alpha = 0.0
    #Add friction to all humanoid joints
    p.changeDynamics(humanoid, linkIndex=-1, lateralFriction=friction)
    #p.changeVisualShape(humanoid, -1, rgbaColor=[1,1,1,alpha])
    for j in range(p.getNumJoints(humanoid)):
        p.changeDynamics(humanoid, linkIndex=j, lateralFriction=friction)
        #p.changeVisualShape(humanoid, j, rgbaColor=[1,1,1,alpha])

    #Set damping to zero, for accurate inverse and forward dynamics comparison
    p.changeDynamics(humanoid, linkIndex=-1, linearDamping=0, angularDamping=0)

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

        #print('{} {} joint_index: {}, link_name: {}'.format(j, joint_name, joint_index, link_name))
        #print('ll: {}, ul: {}'.format(joint_ll, joint_ul))
        #print('--'*30)
        joint_mass  = p.getDynamicsInfo(humanoid, j)[0] * scale
        p.changeDynamics(humanoid, j, mass=joint_mass)
        total_mass += joint_mass 

    print('Total mass: {} kg ~ {:.2f} lbs'.format(total_mass, total_mass*2.2))
    reset_body(humanoid)
    controlMode = p.STABLE_PD_CONTROL #POSITION_CONTROL, VELOCITY_CONTROL, STABLE_PD_CONTROL, TORQUE_CONTROL

    targ_base         = []
    targ_base_vel     = []
    targ_base_orn     = []
    targ_base_orn_vel = []
    tpos  = []
    tvels = []
    height_adj = 0

    motion_data = None
    numFrames   = 1
    vid_freq    = 1./25
    vid_time    = (numFrames-1) * vid_freq

    t = 0 #Current simulation time
    cycleCount = 0 #To keep track of loops. Not currently used
    frame     = -1
    frameNext = 0

    if custom_pose in ['jump','walk','backflip','cartwheel','spin']:
        with open(os.path.join(pybullet_data.getDataPath(),'data/motions/humanoid3d_'+custom_pose+'.txt'), 'r') as f:
                motion_data = json.load(f)['Frames']

        numFrames = len(motion_data)
        vid_freq  = motion_data[0][0]
        vid_time  = (numFrames-1) * vid_freq
        num_time_steps = (np.ceil(vid_time/time_step)+1).astype(np.int32)

        frameTime = t - (cycleCount * vid_time)
        frame = int(frameTime/vid_freq)
        frameNext = min(frame + 1, numFrames-1)

        frameFraction = (frameTime - (frame*vid_freq))/vid_freq
        #print('{:.3f}s: f: {:.3f}'.format(t, frame+frameFraction))
        t += time_step

        targetBasePos, targetBaseOrn, target_positions,\
        baseLinVel, baseAngVel, target_vels = get_pose_from_txt(frame, frameNext, frameFraction, motion_data, motion_data[0][0]) 

        targ_base.append(targetBasePos)
        targ_base_orn.append(targetBaseOrn)
        targ_base_vel.append(baseLinVel)
        targ_base_orn_vel.append(baseAngVel)
        tpos.append(target_positions)
        tvels.append(target_vels)

    elif custom_pose == 'h36m_discussion':
        with open(os.path.join('out_data','discussion.json'), 'r') as f:
            motion_data = json.load(f)['Frames']

        numFrames = len(motion_data)
        vid_freq  = motion_data[0][0]
        vid_time  = (numFrames-1) * vid_freq

        t = 0 #simulation time
        cycleCount = 0 #To keep track of loops. Not currently used
        frame = -1
        frameNext = 0

        while frameNext > frame:
            frameTime = t - (cycleCount * vid_time)
            frame     = int(frameTime/vid_freq)
            frameNext = min(frame + 1, numFrames-1)

            frameFraction = (frameTime - (frame*vid_freq))/vid_freq
            t += time_step

            frameData     = motion_data[frame]
            frameDataNext = motion_data[frameNext]
            frame_duration = frameData[0]
            targetBasePos, targetBaseOrn, target_positions,\
                    baseLinVel, baseAngVel, target_vels = get_angles_vels(frameData, frameDataNext, frameFraction, frame_duration)

            targ_base.append(targetBasePos)
            targ_base_orn.append(targetBaseOrn)
            targ_base_vel.append(baseLinVel)
            targ_base_orn_vel.append(baseAngVel)
            tpos.append(target_positions)
            tvels.append(target_vels)
    else:
        pose_data = get_pose_data(dataset_root, dataset, data_splits, subject=subj, movement=mvmt, pose_format=pose_format)
        motion_data      = pose_data['kpts']
        keyFrameDuration = pose_data['keyFrameDuration']
        motion_data = median_filter(motion_data, 15)

        numFrames = min(len(motion_data),100)
        #numFrames = len(motion_data)

        if const_seg_lens:
            seg_lens    = compute_segment_lens(motion_data, pose_format)
            motion_data = constrain_segment_lens(motion_data, seg_lens, pose_format) 

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

        #Try estimating ground plane
        k = int(numFrames*0.05) #% of frames
        _motion_data = np.copy(motion_data[frame_offset:frame_offset+numFrames])

        _md = np.copy(_motion_data[:,lfoot_idxs+rfoot_idxs]).reshape(-1,3)
        _md = _md[_md[:,1].argsort()][:k]

        lfoot_contact = np.ones(len(_motion_data))
        rfoot_contact = np.ones(len(_motion_data))

        floor_height = np.mean(_md, axis=0)[1] #Average of lowest k points
        print('floor height: {}'.format(floor_height))

        #Simple contact estimation baseline (Borrowed from Rempe etal.)
        #Not in contact if:
        # 1) Heels and toes height > 5cm + (half shape of foot shape height)
        height_thresh = 0.05
        lfoot_contact[np.min(_motion_data[:,lfoot_idxs,1],1) > (floor_height + height_thresh)] = 0
        rfoot_contact[np.min(_motion_data[:,rfoot_idxs,1],1) > (floor_height + height_thresh)] = 0

        # 2) Heels and toes distance > 2cm from previous step
        move_thresh = 0.02
        diff = _motion_data[1:,lfoot_idxs+rfoot_idxs] - _motion_data[:-1,lfoot_idxs+rfoot_idxs]
        dist = np.linalg.norm(diff,axis=-1)
        l_move = np.concatenate(([False],dist[:,0]>move_thresh))
        r_move = np.concatenate(([False],dist[:,1]>move_thresh))

        lfoot_contact[l_move] = 0
        rfoot_contact[r_move] = 0

        foot_height = 0.055
        if pose_format == 'mscoco_foot':
            motion_data[:,:,1] += (floor_height + (foot_height/2)) 
        else:
            motion_data[:,:,1] += (floor_height + foot_height) 

        vid_freq  = keyFrameDuration
        vid_time  = (numFrames-1) * vid_freq
        num_time_steps = (np.ceil(vid_time/time_step)+1).astype(np.int32)

        #Assume contact is on floor at frame 0
        frameTime = t - (cycleCount * vid_time)
        frame     = int(frameTime/vid_freq) + frame_offset
        frameNext = min(frame + 1, frame+numFrames-1)

        frameFraction = (frameTime - ((frame-frame_offset)*vid_freq))/vid_freq
        t += time_step

        targetBasePos, targetBaseOrn, target_positions,\
            baseLinVel, baseAngVel, target_vels = get_motion_data(frame, frameNext, frameFraction, motion_data, file_format='json', pose_format=pose_format, rotate_ankles=rotate_ankles)

        targ_base.append(targetBasePos)
        targ_base_orn.append(targetBaseOrn)
        targ_base_vel.append(baseLinVel)
        targ_base_orn_vel.append(baseAngVel)
        tpos.append(target_positions)
        tvels.append(target_vels)

        set_body_pos(humanoid, targ_base[0], targ_base_orn[0], tpos[0], targ_base_vel[0], targ_base_orn_vel[0], tvels[0])
        aabbMinL = p.getAABB(humanoid, leftAnkle)[0]
        aabbMinR = p.getAABB(humanoid, rightAnkle)[0]
        aabbMinFoot = min(aabbMinL[1], aabbMinR[1])
        aabbMaxFoot = max(aabbMinL[1], aabbMinR[1])
        #height_adj = -(aabbMinFoot * floor_pent_reduce)
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

    targ_base = np.array(targ_base)
    targ_base += [0, height_adj, 0]

    gtpos  = tpos
    gtvels = tvels

    joint_quats = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1] #quaternion dof for each joint
    joint_euler = [3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1] #euler dof for each joint
    plot_euler_angles = plot_kin
    plot_euler_vels = plot_vel
    ext_force = None #Optional: Optimized set of external forces
    if loadTrajectory:
        #Load optimized control targets
        print('Loading trajectory: {}'.format(input_traj))
        opt_traj = np.load(input_traj)
        ref_traj = np.array(quaternion_to_euler(tpos))

        #Convert to quaternions
        tpos = []
        for i in range(opt_traj.shape[0]):
            j_idx = 0
            temp  = []
            for j_dof in joint_euler:
                if j_dof == 1:
                    temp.append(opt_traj[i][j_idx:j_idx+j_dof])
                else:
                    temp.append(p.getQuaternionFromEuler(opt_traj[i][j_idx:j_idx+j_dof]))
                j_idx += j_dof
            tpos.append(temp)

        #Compute target velocities 
        tvels = []
        for i in range(len(tpos)):
            j_idx=0
            tpos0 = []
            tpos1 = []
            for j in range(len(tpos[i])):
                tpos0.append(tpos[i][j])
                try:
                    tpos1.append(tpos[i+1][j])
                except IndexError:
                    tpos1.append(tpos[i][j])
            tvels.append(getJointVels(tpos0, tpos1, vid_freq))

        num_time_steps = len(tpos)
        numFrames      = int((num_time_steps/sim_freq)/vid_freq)
        
        #Load external forces if optimized
        ext_force_file = os.path.join('output_external_forces',input_traj.split('/')[-1])
        if os.path.isfile(ext_force_file):
            ext_force = np.load(ext_force_file)
            print('Max force: {}, min force: {}'.format(np.max(ext_force), np.min(ext_force)))

    print(str(numFrames)+' frames')
    vid_sample_ratio  = int(vid_freq/time_step)

    print('{} time steps'.format(num_time_steps))

    contact_args = ['contactFlag',
        'bodyUniqueIdA',
        'bodyUniqueIdB',
        'linkIndexA',
        'linkIndexB',
        'positionOnA',
        'positionOnB',
        'contactNormalOnB',
        'contactDistance',
        'normalForce',
        'lateralFriction1',
        'lateralFrictionDir1',
        'lateralFriction2',
        'lateralFrictionDir2']

    curr_base = []
    curr_base_vel = []
    curr_base_orn = []
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

    fallen_state = []
    balance_state = []

    lankle_contact = []
    rankle_contact = []

    lankle_force = []
    rankle_force = []

    lankle_force_sense = np.empty((num_time_steps, 6))
    rankle_force_sense = np.empty((num_time_steps, 6))
    joint_torques = np.empty((num_time_steps, np.sum(joint_euler))) #Applied joint torques
    joint_react_forces = np.empty((num_time_steps, len(joint_indices), 6)) #Measured joint reaction forces and moments

    #Enable joint reaction force sensors
    p.enableJointForceTorqueSensor(humanoid, leftAnkle)
    p.enableJointForceTorqueSensor(humanoid, rightAnkle)

    p.enableJointForceTorqueSensor(humanoid, root)
    p.enableJointForceTorqueSensor(humanoid, chest)
    p.enableJointForceTorqueSensor(humanoid, neck)
    p.enableJointForceTorqueSensor(humanoid, rightHip)
    p.enableJointForceTorqueSensor(humanoid, rightKnee)
    p.enableJointForceTorqueSensor(humanoid, rightShoulder)
    p.enableJointForceTorqueSensor(humanoid, rightElbow)
    p.enableJointForceTorqueSensor(humanoid, rightWrist)
    p.enableJointForceTorqueSensor(humanoid, leftHip)
    p.enableJointForceTorqueSensor(humanoid, leftKnee)
    p.enableJointForceTorqueSensor(humanoid, leftShoulder)
    p.enableJointForceTorqueSensor(humanoid, leftElbow)
    p.enableJointForceTorqueSensor(humanoid, leftWrist)

    set_body_pos(humanoid, targ_base[0], targ_base_orn[0], gtpos[0], targ_base_vel[0], targ_base_orn_vel[0], gtvels[0])
    set_body_pos(humanoid_kin, targ_base[0], targ_base_orn[0], gtpos[0], targ_base_vel[0], targ_base_orn_vel[0], gtvels[0])
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

    aabbMinL,aabbMaxL = p.getAABB(humanoid, leftAnkle)
    aabbMinR,aabbMaxR = p.getAABB(humanoid, rightAnkle)
    print(aabbMinL)
    print(aabbMinR)
    
    print('---'*30)
    t_i = 0
    try:
        #Running at simulation frequency
        while frameNext > frame:
            '''
            camera_params = p.getDebugVisualizerCamera()
            yaw = camera_params[8]
            pitch = camera_params[9]
            dist = camera_params[10]
            cam_targ = camera_params[11]
            print('Camera yaw: {:.2f}, pitch: {:.2f}, dist: {:.2f}'.format(yaw, pitch, dist))
            print('Camera target: {}'.format(cam_targ))
            print('--'*30)
            '''
            if override_ctrl:
                p.resetBasePositionAndOrientation(humanoid, targ_base[-1], targ_base_orn[-1])
                p.resetBaseVelocity(humanoid, targ_base_vel[-1], targ_base_orn_vel[-1])

            p.setJointMotorControlMultiDofArray(humanoid,
                                                joint_indices,
                                                controlMode,
                                                targetPositions=tpos[t_i],
                                                targetVelocities=tvels[t_i],
                                                positionGains=kps,
                                                velocityGains=kds,
                                                forces=maxForces,
                                                )

            set_body_pos(humanoid_kin, targ_base[-1], targ_base_orn[-1], gtpos[-1], targ_base_vel[-1], targ_base_orn_vel[-1], gtvels[-1])

            basePos, baseOrn = p.getBasePositionAndOrientation(humanoid)
            baseVel, baseOrnVel = p.getBaseVelocity(humanoid)
            #ornErr = np.arccos(np.abs(np.clip(np.dot(np.array([0,0,0,1]), baseOrn),-1,1)))
            #print('basePos: {}, baseOrn: {}, ornErr: {}'.format(np.around(basePos,3), np.around(baseOrn,3), ornErr))

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

            fallen_state.append(has_fallen(p, humanoid, fall_indices))
            balance_state.append(is_balanced(p, humanoid, leftAnkle, rightAnkle, k_comv)) #Use kinematic reference COM velocity
            #print('k_comv: {:.3f}, comv: {:.3f}, hasFallen: {}, hasBalance: {}'.format(np.max(np.abs(k_comv)), np.max(np.abs(comv)), fallen_state[-1], balance_state[-1]))

            #convex_hull_dist = is_balanced(p, humanoid, leftAnkle, rightAnkle, k_comv)[1]

            kin_pose = []
            kin_vel  = []
            kin_pose_eul = []
            kin_vel_eul  = []
            j_idx = 0
            for i,(idx,dof) in enumerate(zip(joint_indices, joint_dofs)):
                js = p.getJointStateMultiDof(humanoid, idx)
                kin_pose.append(js[0])
                kin_vel_eul.extend(js[1])
                if dof == 1:
                    kin_pose_eul.extend(js[0])
                    kin_vel.append(js[1])
                else:
                    kin_pose_eul.extend(p.getEulerFromQuaternion(js[0]))
                    kin_vel.append(p.getQuaternionFromEuler(js[1]))

                joint_torques[t_i,j_idx:j_idx+dof] = js[3]
                joint_react_forces[t_i,i] = js[2]
                j_idx += dof

            curr_kin_pose.append(kin_pose)
            curr_kin_vel.append(kin_vel)
            curr_kin_pose_eul.append(kin_pose_eul)
            curr_kin_vel_eul.append(kin_vel_eul)

            #Grab joint reaction forces (and moments)
            rankle_force_sense[t_i] = p.getJointStateMultiDof(humanoid, 5)[2] #rankle
            lankle_force_sense[t_i] = p.getJointStateMultiDof(humanoid, 11)[2] #lankle

            rcont = p.getContactPoints(humanoid, -1, rightAnkle, -1) 
            lcont = p.getContactPoints(humanoid, -1, leftAnkle, -1) 
            aabbMinL = p.getAABB(humanoid, leftAnkle)[0]
            aabbMinR = p.getAABB(humanoid, rightAnkle)[0]
            lankle_contact.append((aabbMinL[1]<0.0005)*1.0)
            rankle_contact.append((aabbMinR[1]<0.0005)*1.0)
            if len(rcont) > 0:
                cont_normal = rcont[0][9]
                cont_fric1  = rcont[0][10]
                cont_fric2  = rcont[0][12]

                rankle_force.append([cont_fric1, cont_normal, cont_fric2])
            else:
                rankle_force.append(np.array([0,0,0]))

            if len(lcont) > 0:
                cont_normal = lcont[0][9]
                cont_fric1  = lcont[0][10]
                cont_fric2  = lcont[0][12]

                lankle_force.append([cont_fric1, cont_normal, cont_fric2])
            else:
                lankle_force.append(np.array([0,0,0]))


            #Apply External Residual Forces to root node, if optimized
            if ext_force is not None:
                p.applyExternalForce(humanoid, -1, ext_force[t_i], basePos, flags=p.WORLD_FRAME)

            p.stepSimulation()

            t += time_step
            t_i += 1
            frameTime = t - (cycleCount * vid_time)
            frame     = int(frameTime/vid_freq) + frame_offset
            frameNext = min(frame +1, frame+numFrames-1)

            frameFraction = (frameTime - ((frame-frame_offset)*vid_freq))/vid_freq

            if custom_pose in ['jump','walk','backflip','cartwheel','spin']:
                targetBasePos, targetBaseOrn, target_positions,\
                baseLinVel, baseAngVel, target_vels = get_pose_from_txt(frame, frameNext, frameFraction, motion_data, motion_data[0][0])
            elif custom_pose in ['test_walk']:
                targetBasePos, targetBaseOrn, target_positions,\
                        baseLinVel, baseAngVel, target_vels = get_motion_data(frame, frameNext, frameFraction, motion_data, file_format='json', pose_format='h36m_17')
            else:
                targetBasePos, targetBaseOrn, target_positions,\
                baseLinVel, baseAngVel, target_vels = get_motion_data(frame, frameNext, frameFraction, motion_data, file_format='json', pose_format=pose_format, rotate_ankles=rotate_ankles)

            targ_base = np.concatenate((targ_base, (np.array(targetBasePos)+[0,height_adj,0])[None]))
            targ_base_orn.append(targetBaseOrn)
            targ_base_vel.append(baseLinVel)
            targ_base_orn_vel.append(baseAngVel)
            gtpos.append(target_positions)
            gtvels.append(target_vels)

            if write_video:
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
            else:
                time.sleep(time_step)
    except Exception as e:
        print('Error: {}'.format(e))

    p.disconnect()

    curr_base = np.array(curr_base)
    curr_base_orn = np.array(curr_base_orn)
    l_root = np.mean(np.linalg.norm(targ_base[:len(curr_base)] - curr_base, axis=-1))

    curr_com = np.array(curr_com)
    kin_com  = np.array(kin_com)
    l_com    = np.mean(np.linalg.norm(kin_com - curr_com, axis=-1))

    fail_state = np.bitwise_or(fallen_state, np.invert(balance_state))
    pass_rate = 1-(np.sum(fail_state)/len(fail_state))
    frames_until_failure = pass_rate*numFrames
    print('l_root: {:.2f}mm, l_com: {:.2f}mm'.format(l_root*1000, l_com*1000))
    print('Number of frames until failure: {:.2f}'.format(frames_until_failure))

    curr_head = np.array(curr_head)
    kin_head  = np.array(kin_head)

    l_root_orn  = 0
    l_joint     = 0
    l_joint_eul = 0
    l_joint_vel = 0 
    l_joint_vel_eul = 0 

    gt_kin_pose_vel = array_to_joint_list(np.array(euler_to_quaternion(gtvels)), joint_quats)
    temp_orn = []
    for f in range(len(curr_base)): #Iterate over frames
        l_root_orn += np.arccos(np.abs(np.clip(np.dot(targ_base_orn[f], curr_base_orn[f]),-1,1)))
        temp_orn.append(np.arccos(np.abs(np.clip(np.dot(targ_base_orn[f], curr_base_orn[f]),-1,1))))

        for j, (targ_pose, curr_pose) in enumerate(zip(gtpos[f], curr_kin_pose[f])): #Iterate over joints
            if joint_dofs[j] == 1: #Distance for revolute angles is just difference in radians
                l_joint += np.abs(np.array(targ_pose) - np.array(curr_pose)).item()
            else:
                l_joint += np.arccos(np.abs(np.clip(np.array(targ_pose) @ np.array(curr_pose),-1,1)))

        for j, (targ_vel, curr_vel) in enumerate(zip(gt_kin_pose_vel[f], curr_kin_vel[f])):
            if joint_dofs[j] == 1: #Distance for revolute angles is just difference in radians
                l_joint_vel += np.abs(np.array(targ_vel - np.array(curr_vel))).item()
            else:
                l_joint_vel += np.arccos(np.abs(np.clip(np.array(targ_vel) @ np.array(curr_vel),-1,1)))

    targ_pose_eul     = np.array(quaternion_to_euler(gtpos))
    curr_kin_pose_eul = np.array(curr_kin_pose_eul)
    l_joint_eul = np.sum(jws * (targ_pose_eul[:len(curr_kin_pose_eul)] - curr_kin_pose_eul)**2)

    targ_vels_eul = []
    for i1 in gtvels:
        tve = []
        for i2 in i1:
            tve.extend(i2)
        targ_vels_eul.append(tve)
    targ_vels_eul = np.array(targ_vels_eul)
    curr_kin_vel_eul  = np.array(curr_kin_vel_eul)
    l_joint_vel_eul = np.sum(jws * (targ_vels_eul[:len(curr_kin_vel_eul)] - curr_kin_vel_eul)**2)

    print('l_root_orn: {:.4f}, l_joint: {:.4f}, l_vel: {:.4f}'.format(l_root_orn, l_joint_eul, l_joint_vel_eul)) 
    plt.show()

    if save_metrics:
        metric_dir = './output_pybullet_metrics'
        if custom_pose is None:
            metric_dir = os.path.join(metric_dir, subj, mvmt)
        else:
            metric_dir = os.path.join(metric_dir, custom_pose)

        metric_file = os.path.join(metric_dir, met_group+'.json')
        os.makedirs(metric_dir, exist_ok=True)
        metrics = {'KIN_ANG':curr_kin_pose_eul.tolist(),
                   'KIN_VEL':curr_kin_vel_eul.tolist(),
                   'FORCE_ANK_L':lankle_force_sense.tolist(),
                   'FORCE_ANK_R':rankle_force_sense.tolist(),
                   'FORCE_ANK':np.concatenate((lankle_force_sense,rankle_force_sense),axis=-1).tolist(),
                   'CONTACT_FORCE_ANK':np.concatenate((lankle_force, rankle_force),axis=-1).tolist(),
                   'APPL_TORQ':joint_torques.tolist(),
                   'simulation_frequency':sim_freq}
        with open(metric_file, 'w') as f:
            json.dump(metrics, f)

    ref_traj = np.array(quaternion_to_euler(gtpos))
    ref_vel = gtvels
    targ_traj = np.array(quaternion_to_euler(tpos))
    targ_vel = tvels
    j_idx = 0
    if plot_euler_angles: #Plot target and measured euler angles
        plt.figure(figsize=(16,12))
        for i, (j_name, j_dof) in enumerate(zip(joint_names, joint_euler)):
            ref_vals   = ref_traj[:,j_idx:j_idx+j_dof]
            targ_vals   = targ_traj[:,j_idx:j_idx+j_dof]
            joint_vals = curr_kin_pose_eul[:,j_idx:j_idx+j_dof]
            j_idx += j_dof

            if j_dof == 1:
                plt.subplot(5,6,j_idx)
                plt.plot(ref_vals, 'k--',label='Reference')
                plt.plot(targ_vals, 'c-', label='target')
                plt.plot(joint_vals, 'r', label='Measured')
                plt.ylim([-np.pi, np.pi])
                plt.title(j_name)
            else:
                plt.subplot(5,6,j_idx-2)
                plt.plot(ref_vals[:,0], 'k--', label='Reference')
                plt.plot(targ_vals[:,0], 'c-', label='Target')
                plt.plot(joint_vals[:,0], 'r', label='Measured')
                plt.ylim([-np.pi, np.pi])
                plt.title(j_name+' X (roll)')

                plt.subplot(5,6,j_idx-1)
                plt.plot(ref_vals[:,1], 'k--', label='Reference')
                plt.plot(targ_vals[:,1], 'c-', label='Target')
                plt.plot(joint_vals[:,1], 'r', label='Measured')
                plt.ylim([-np.pi, np.pi])
                plt.title(j_name+' Y (pitch)')

                plt.subplot(5,6,j_idx)
                plt.plot(ref_vals[:,2], 'k--', label='Reference')
                plt.plot(targ_vals[:,2], 'c-', label='Target')
                plt.plot(joint_vals[:,2], 'r', label='Measured')
                plt.ylim([-np.pi, np.pi])
                plt.title(j_name+' Z (yaw)')

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    if plot_euler_vels: #Plot target and measured euler joint velocities
        plt.figure(figsize=(16,12))
        plt_idx = 0
        for j_idx, (j_name, j_dof) in enumerate(zip(joint_names, joint_euler)):
            ref_vals   = targ_vels_eul[:,plt_idx:plt_idx+j_dof]
            targ_vals  = np.array([item[j_idx] for item in targ_vel])
            joint_vals = curr_kin_vel_eul[:,plt_idx:plt_idx+j_dof]
            plt_idx += j_dof

            if j_dof == 1:
                plt.subplot(5,6,plt_idx)
                plt.plot(ref_vals, 'k--',label='Reference')
                plt.plot(targ_vals, 'c-',label='Target')
                plt.plot(joint_vals, 'r', label='Measured')
                plt.title(j_name+' velocity')
            else:
                plt.subplot(5,6,plt_idx-2)
                plt.plot(ref_vals[:,0], 'k--', label='Reference')
                plt.plot(targ_vals[:,0], 'c-',label='Target')
                plt.plot(joint_vals[:,0], 'r', label='Measured')
                plt.title(j_name+' X (roll) velocity')

                plt.subplot(5,6,plt_idx-1)
                plt.plot(ref_vals[:,1], 'k--', label='Reference')
                plt.plot(targ_vals[:,1], 'c-',label='Target')
                plt.plot(joint_vals[:,1], 'r', label='Measured')
                plt.title(j_name+' Y (pitch) velocity')

                plt.subplot(5,6,plt_idx)
                plt.plot(ref_vals[:,2], 'k--', label='Reference')
                plt.plot(targ_vals[:,2], 'c-',label='Target')
                plt.plot(joint_vals[:,2], 'r', label='Measured')
                plt.title(j_name+' Z (yaw) velocity')

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    if plot_force:
        from scipy.signal import medfilt
        lankle_force_sense = np.array(lankle_force_sense)
        rankle_force_sense = np.array(rankle_force_sense)

        #Invert axes for force sensors
        lankle_force_sense *= -1
        rankle_force_sense *= -1

        lankle_contact     = np.array(lankle_contact) 
        rankle_contact     = np.array(rankle_contact) 
        lankle_force       = np.array(lankle_force)
        rankle_force       = np.array(rankle_force)

        '''
        plt.figure(figsize=(12,8))
        plt.subplot(1,3,1)
        plt.plot(medfilt(lankle_force_sense[:,0],15), label='force_sense_x left')
        plt.plot(medfilt(rankle_force_sense[:,0],15), label='force_sense_x right')
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(medfilt(lankle_force_sense[:,1],15), label='force_sense_y left')
        plt.plot(medfilt(rankle_force_sense[:,1],15), label='force_sense_y right')
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(medfilt(lankle_force_sense[:,2],15), label='force_sense_z left')
        plt.plot(medfilt(rankle_force_sense[:,2],15), label='force_sense_z right')
        plt.legend()
        '''

        grf_fig = plt.figure(figsize=(16,6))
        plt.rcParams.update({'font.size': 22})
        plt.plot(medfilt(np.sum(lankle_force,-1),49), 'r', linewidth=2.5, label='Force (left)')
        plt.plot(medfilt(np.sum(rankle_force,-1),49), 'b--', linewidth=2.5, label='Force (right)')
        plt.legend(loc='upper right')
        plt.ylim([0, 500])
        plt.ylabel('N')
        plt.xlabel('Time (s)')

        plt.tight_layout()

        #grf_fig.savefig(os.path.join('output_plots',subj+'_'+mvmt+'_'+input_traj.split('/')[-1].split('.')[0]+'.png'))

        _lfoot_contact = zoom(lfoot_contact, len(lankle_contact)/len(lfoot_contact)) > 0.5
        _rfoot_contact = zoom(rfoot_contact, len(rankle_contact)/len(rfoot_contact)) > 0.5

        contact_fig = plt.figure(figsize=(16,6))
        plt.subplot(2,1,1)
        plt.plot(_lfoot_contact, 'r', label='gt_contact (left)',linewidth=2.5)
        plt.plot(_rfoot_contact, 'b--', label='gt_contact (right)', linewidth=2.5)
        plt.ylim([0,1.5])
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.subplot(2,1,2)
        plt.plot(lankle_contact, 'r', label='measured contact (left)',linewidth=2.5)
        plt.plot(rankle_contact, 'b--', label='measured contact (right)', linewidth=2.5)
        plt.ylim([0,1.5])
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')

        plt.tight_layout()

        #contact_fig.savefig(os.path.join('output_plots',subj+'_'+mvmt+'_'+pose_format+'_contact.png'))

        plt.show()

    if plot_torque: #Plot applied joint torques at each controllable joint
        '''
        plt.figure(figsize=(16,12))
        j_idx = 0
        for i, (j_name, j_dof) in enumerate(zip(joint_names, joint_euler)):
            vals = joint_torques[:,j_idx:j_idx+j_dof]
            j_idx += j_dof

            if j_dof == 1:
                plt.subplot(5,6,j_idx)
                plt.plot(vals, label='Applied torque')
                plt.ylim([-200, 200])
                plt.title(j_name)
            else:
                plt.subplot(5,6,j_idx-2)
                plt.plot(vals[:,0], label='Applied torque')
                plt.ylim([-200, 200])
                plt.title(j_name+' X (roll)')

                plt.subplot(5,6,j_idx-1)
                plt.plot(vals[:,1], label='Applied torque')
                plt.title(j_name+' Y (pitch)')

                plt.subplot(5,6,j_idx)
                plt.plot(vals[:,2], label='Applied torque')
                plt.ylim([-200, 200])
                plt.title(j_name+' Z (yaw)')

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        '''

        plt.figure(figsize=(16,12))
        j_idx = 1
        for i, j_name in enumerate(joint_names):
            vals = joint_react_forces[:,i]

            plt.subplot(6,6,j_idx)
            plt.plot(vals[:,0], label='Measured force')
            plt.ylim([-1500, 1500])
            plt.title(j_name+' Fx')

            plt.subplot(6,6,j_idx+1)
            plt.plot(vals[:,1], label='Measured force')
            plt.ylim([-1500, 1500])
            plt.title(j_name+' Fy')

            plt.subplot(6,6,j_idx+2)
            plt.plot(vals[:,2], label='Measured force')
            plt.ylim([-1500, 1500])
            plt.title(j_name+' Fz')

            j_idx += 3
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.suptitle('Measured Forces')

        plt.figure(figsize=(16,12))
        j_idx = 1
        for i, j_name in enumerate(joint_names):
            vals = joint_react_forces[:,i]

            plt.subplot(6,6,j_idx)
            plt.plot(vals[:,3], label='Measured moment', c='r')
            plt.ylim([-500, 500])
            plt.title(j_name+' Mx')

            plt.subplot(6,6,j_idx+1)
            plt.plot(vals[:,4], label='Measured moment', c='r')
            plt.ylim([-500, 500])
            plt.title(j_name+' My')

            plt.subplot(6,6,j_idx+2)
            plt.plot(vals[:,5], label='Measured moment', c='r')
            plt.ylim([-500, 500])
            plt.title(j_name+' Mz')

            j_idx += 3
        plt.legend()
        plt.suptitle('Measured Moments')
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
