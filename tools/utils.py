#Many functions adapted directly from PyBullet DeepMimic implementation
import os
import sys
DIR = os.getcwd()
sys.path.append(DIR)

import json
import pybullet as p
import numpy as np

from scipy import interpolate

from tools.inverse_kinematics import coord_to_rot, coord_seq_to_rot_seq 

pelvis = 0 #root/base
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

def reset_body(humanoid, offset=[0,0,0]):
    #Reset base and joint positions
    p.resetBasePositionAndOrientation(humanoid, (np.array([0,0.77,0]) + np.array(offset)).tolist(), [0,0,0,1])
    p.resetJointStateMultiDof(humanoid, chest, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, neck, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, rightHip, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, rightKnee, [0], [0])
    p.resetJointStateMultiDof(humanoid, rightAnkle, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, rightShoulder, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, rightElbow, [0], [0])
    p.resetJointStateMultiDof(humanoid, leftHip, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, leftKnee, [0], [0])
    p.resetJointStateMultiDof(humanoid, leftAnkle, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, leftShoulder, [0, 0, 0, 1], [0, 0, 0, 1])
    p.resetJointStateMultiDof(humanoid, leftElbow, [0], [0])

def set_body_pos(humanoid, basePos, baseOrn, target_positions, baseLinVel=None, baseAngVel=None, target_vels=None, offset=[0,0,0]):
    p.resetBasePositionAndOrientation(humanoid, (np.array(basePos) + np.array(offset)).tolist(), baseOrn)

    p.resetJointStateMultiDof(humanoid, chest, target_positions[0], target_vels[0])
    p.resetJointStateMultiDof(humanoid, neck, target_positions[1], target_vels[1])
    p.resetJointStateMultiDof(humanoid, rightHip, target_positions[2], target_vels[2])
    p.resetJointStateMultiDof(humanoid, rightKnee, target_positions[3], target_vels[3])
    p.resetJointStateMultiDof(humanoid, rightAnkle, target_positions[4], target_vels[4])
    p.resetJointStateMultiDof(humanoid, rightShoulder, target_positions[5], target_vels[5])
    p.resetJointStateMultiDof(humanoid, rightElbow, target_positions[6], target_vels[6])
    p.resetJointStateMultiDof(humanoid, leftHip, target_positions[7], target_vels[7])
    p.resetJointStateMultiDof(humanoid, leftKnee, target_positions[8], target_vels[8])
    p.resetJointStateMultiDof(humanoid, leftAnkle, target_positions[9], target_vels[9])
    p.resetJointStateMultiDof(humanoid, leftShoulder, target_positions[10], target_vels[10])
    p.resetJointStateMultiDof(humanoid, leftElbow, target_positions[11], target_vels[11])

    if baseLinVel is not None:
        p.resetBaseVelocity(humanoid, baseLinVel, baseAngVel)

def get_base_and_joint_angles(frame_num, pose):
    rot_pose = coord_to_rot(frameNum=frame_num, frame=pose[frame_num], frame_duration=0.001, pose_format=pose_format)

    basePos = [rot_pose[1], rot_pose[2], rot_pose[3]]
    baseOrn = [rot_pose[5], rot_pose[6], rot_pose[7], rot_pose[4]]

    chestRot         = [rot_pose[9], rot_pose[10], rot_pose[11], rot_pose[8]]
    neckRot          = [rot_pose[13], rot_pose[14], rot_pose[15], rot_pose[12]]
    rightHipRot      = [rot_pose[17], rot_pose[18], rot_pose[19], rot_pose[16]]
    rightKneeRot     = [rot_pose[20]]
    rightAnkleRot    = [rot_pose[22], rot_pose[23], rot_pose[24], rot_pose[21]]
    rightShoulderRot = [rot_pose[26], rot_pose[27], rot_pose[28], rot_pose[25]]
    rightElbowRot    = [rot_pose[29]]
    leftHipRot       = [rot_pose[31], rot_pose[32], rot_pose[33], rot_pose[30]]
    leftKneeRot      = [rot_pose[34]]
    leftAnkleRot     = [rot_pose[36], rot_pose[37], rot_pose[38], rot_pose[35]]
    leftShoulderRot  = [rot_pose[40], rot_pose[41], rot_pose[42], rot_pose[39]]
    leftElbowRot     = [rot_pose[43]]

    joint_angles = [chestRot, neckRot, rightHipRot, rightKneeRot, rightAnkleRot, rightShoulderRot,
                  rightElbowRot, leftHipRot, leftKneeRot, leftAnkleRot, leftShoulderRot, leftElbowRot]

    return basePos, baseOrn, joint_angles

def get_pose_from_txt(frame, frameNext, frameFraction, motion_data, keyFrameDuration, return_list=False):
    frameData = motion_data[frame]
    frameDataNext = motion_data[frameNext]

    return get_angles_vels(frameData, frameDataNext, frameFraction, keyFrameDuration, return_list)

def get_motion_data(frame, frameNext, frameFraction, motion_data, file_format, return_list=False, pose_format=None, rotate_ankles=False):
    if file_format == 'json':
        keyFrameDuration = 1/50
        frameData = coord_to_rot(frameNum=frame, frame=motion_data[frame], frame_duration=keyFrameDuration, pose_format=pose_format, rotate_ankles=rotate_ankles)
        frameDataNext = coord_to_rot(frameNum=frameNext, frame=motion_data[frameNext], frame_duration=keyFrameDuration, pose_format=pose_format, rotate_ankles=rotate_ankles)
    else:
        frameData = motion_data['Frames'][frame]
        frameDataNext = motion_data['Frames'][frameNext]
        keyFrameDuration = keyFrameDuration

    return get_angles_vels(frameData, frameDataNext, frameFraction, keyFrameDuration, return_list)

def get_angles_vels(frameData, frameDataNext, frameFraction, keyFrameDuration, return_list=False):
    basePos1Start = [frameData[1], frameData[2], frameData[3]]
    basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    basePos = [
      basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
      basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
      basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]
    baseLinVel = ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
    baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
    baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
    baseOrn = p.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    baseAngVel = ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration)

    chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
    chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
    chestRot = p.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
    chestVel = ComputeAngVel(chestRotStart, chestRotEnd, keyFrameDuration)

    neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
    neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
    neckRot = p.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
    neckVel = ComputeAngVel(neckRotStart, neckRotEnd, keyFrameDuration)

    rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
    rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
    rightHipRot = p.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)
    rightHipVel = ComputeAngVel(rightHipRotStart, rightHipRotEnd, keyFrameDuration)

    rightKneeRotStart = [frameData[20]]
    rightKneeRotEnd = [frameDataNext[20]]
    rightKneeRot = [
      rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
    ]
    rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

    rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
    rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
    rightAnkleRot = p.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)
    rightAnkleVel = ComputeAngVel(rightAnkleRotStart, rightAnkleRotEnd,
            keyFrameDuration)

    rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
    rightShoulderRotEnd = [
      frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
    ]
    rightShoulderRot = p.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd,
                                          frameFraction)
    rightShoulderVel = ComputeAngVel(rightShoulderRotStart, rightShoulderRotEnd,
            keyFrameDuration)

    rightElbowRotStart = [frameData[29]]
    rightElbowRotEnd = [frameDataNext[29]]
    rightElbowRot = [
      rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
    ]
    rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]

    leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
    leftHipRotEnd = [frameDataNext[31], frameDataNext[32], frameDataNext[33], frameDataNext[30]]
    leftHipRot = p.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)
    leftHipVel = ComputeAngVel(leftHipRotStart, leftHipRotEnd, keyFrameDuration)

    leftKneeRotStart = [frameData[34]]
    leftKneeRotEnd = [frameDataNext[34]]
    leftKneeRot = [leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])]
    leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

    leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
    leftAnkleRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[35]]
    leftAnkleRot = p.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)
    leftAnkleVel = ComputeAngVel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration)

    leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]
    leftShoulderRotEnd = [frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[39]]
    leftShoulderRot = p.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)
    leftShoulderVel = ComputeAngVel(leftShoulderRotStart, leftShoulderRotEnd,
            keyFrameDuration)

    leftElbowRotStart = [frameData[43]]
    leftElbowRotEnd = [frameDataNext[43]]
    leftElbowRot = [
      leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
    ]
    leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

    if return_list: #return as list of DOFs instead of list of tuples
        joint_angles = list(chestRot+neckRot+rightHipRot+tuple(rightKneeRot)+rightAnkleRot+rightShoulderRot+\
                      tuple(rightElbowRot)+leftHipRot+tuple(leftKneeRot)+leftAnkleRot+leftShoulderRot+tuple(leftElbowRot))

        joint_vels = chestVel+neckVel+rightHipVel+rightKneeVel+rightAnkleVel+rightShoulderVel+\
                    rightElbowVel+leftHipVel+leftKneeVel+leftAnkleVel+leftShoulderVel+leftElbowVel
    else:
        joint_angles = [chestRot, neckRot, rightHipRot, rightKneeRot, rightAnkleRot, rightShoulderRot,
                      rightElbowRot, leftHipRot, leftKneeRot, leftAnkleRot, leftShoulderRot, leftElbowRot]

        joint_vels = [chestVel, neckVel, rightHipVel, rightKneeVel, rightAnkleVel, rightShoulderVel,
                    rightElbowVel, leftHipVel, leftKneeVel, leftAnkleVel, leftShoulderVel, leftElbowVel]
    return basePos, baseOrn, joint_angles, baseLinVel, baseAngVel, joint_vels

def getJointVels(jointAngStart, jointAngEnd, keyFrameDuration=1/50):
    chestRotStart, neckRotStart, rightHipRotStart, rightKneeRotStart, rightAnkleRotStart, rightShoulderRotStart,\
            rightElbowRotStart, leftHipRotStart, leftKneeRotStart, leftAnkleRotStart, leftShoulderRotStart, leftElbowRotStart = jointAngStart

    chestRotEnd, neckRotEnd, rightHipRotEnd, rightKneeRotEnd, rightAnkleRotEnd, rightShoulderRotEnd,\
            rightElbowRotEnd, leftHipRotEnd, leftKneeRotEnd, leftAnkleRotEnd, leftShoulderRotEnd, leftElbowRotEnd = jointAngEnd

    chestVel = ComputeAngVel(chestRotStart, chestRotEnd, keyFrameDuration)
    neckVel = ComputeAngVel(neckRotStart, neckRotEnd, keyFrameDuration)
    rightHipVel = ComputeAngVel(rightHipRotStart, rightHipRotEnd, keyFrameDuration)
    rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]
    rightAnkleVel = ComputeAngVel(rightAnkleRotStart, rightAnkleRotEnd,keyFrameDuration)
    rightShoulderVel = ComputeAngVel(rightShoulderRotStart, rightShoulderRotEnd,keyFrameDuration)
    rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]
    leftHipVel = ComputeAngVel(leftHipRotStart, leftHipRotEnd, keyFrameDuration)
    leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]
    leftAnkleVel = ComputeAngVel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration)
    leftShoulderVel = ComputeAngVel(leftShoulderRotStart, leftShoulderRotEnd,keyFrameDuration)
    leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

    joint_vels = [chestVel, neckVel, rightHipVel, rightKneeVel, rightAnkleVel, rightShoulderVel,
                rightElbowVel, leftHipVel, leftKneeVel, leftAnkleVel, leftShoulderVel, leftElbowVel]

    return joint_vels

def ComputeAngVel(ornStart, ornEnd, deltaTime):
    dorn = p.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = p.getAxisAngleFromQuaternion(dorn)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
              (axis[2] * angle) / deltaTime]
    return angVel

def ComputeLinVel(posStart, posEnd, deltaTime):
    vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
           (posEnd[2] - posStart[2]) / deltaTime]
    return vel

def computeCOMposVel(pb, uid):
    """Compute center-of-mass position and velocity."""
    num_joints = 15
    jointIndices = range(num_joints)
    link_states = pb.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
    link_pos = np.array([s[0] for s in link_states])
    link_vel = np.array([s[-2] for s in link_states])
    tot_mass = 0.
    masses = []
    for j in jointIndices:
      mass_, *_ = pb.getDynamicsInfo(uid, j)
      masses.append(mass_)
      tot_mass += mass_
    masses = np.asarray(masses)[:, None]
    com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
    com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
    return com_pos, com_vel

def has_fallen(pb, uid, joint_indices):
    #check if any non-allowed body part (from joint_indices)hits the ground
    terminates = False
    pts = pb.getContactPoints()
    for p in pts:
      part = -1
      #ignore self-collision
      if (p[1] == p[2]):
        continue
      if (p[1] == uid):
        part = p[3]
      if (p[2] == uid):
        part = p[4]
      if (part >= 0 and part in joint_indices):
        #print("terminating part:", part)
        terminates = True

    return terminates


def draw_ground_truth(coord_seq, frame, lifeTime=1.0, shift=[0.0, 0.0, 0.0], pose_format='mscoco'):
    if pose_format == 'hand':
        #Hand format, 0-indexed
        links = [[0,1], [1,2], [2,3], [3,4],
             [0,5], [5,6], [6,7], [7,8],
             [0,9], [9,10], [10,11], [11,12],
             [0,13], [13,14],[14,15], [15,16],
             [0,17], [17,18], [18,19], [19,20]] 

        idx_0=0
    else:
        #MSCOCO pose, 1-indexed
        links = [[16,14],[14,12],[17,15],[15,13],
                         [12,13],[6,12],[7,13],[6,7],
                         [6,8],[7,9],[8,10],[9,11],[2,3],
                         [1,2],[1,3],[2,4],[3,5],[1,6],[1,7]]
        idx_0=1

    joint = coord_seq[frame]
    shift = np.array(shift)
    for idx, (i,j) in enumerate(links):
        p.addUserDebugLine(lineFromXYZ=joint[i-idx_0] + shift,
                         lineToXYZ=joint[j-idx_0] + shift,
                         lineColorRGB=(0, 0, 255),
                         lineWidth=1,
                         lifeTime=lifeTime)

#Fit to B-Spline and return interpolated trajectories
#Perform each interpolation independently
def get_bspline(x, indices, num_points):
    intpl_points = []
    tcks = []

    for dof in range(x.shape[-1]):
        targets = x[:,dof]
        new_points = np.linspace(0,1,num_points)

        try:
            tck,u = interpolate.splprep([indices, targets], k=3, s=0)
            smpl, out = interpolate.splev(new_points, tck)
        except ValueError:
            #All zero inputs, B-spline function fails
            out = [np.zeros(new_points.shape)]*dof

        intpl_points.append(out)
        tcks.append(tck)

    intpl_points = np.array(intpl_points).transpose()

    return intpl_points, tcks 

#Convert from euler joints to quaternion joints
def euler_to_quaternion(x, dofs=None):
    tpos = []
    if dofs == None: #Joint list format
        for t in range(len(x)):
            temp = []
            for item in x[t]:
                if len(item) == 1:
                    temp.extend(item)
                else:
                    temp.extend(p.getQuaternionFromEuler(item))
            tpos.append(temp)
    else: #NUmpy array format
        for t in range(len(x)):
            j_idx = 0 
            temp = []
            for dof in dofs:
                if dof == 1:
                    temp.append(x[t][j_idx:j_idx+dof])
                else:
                    temp.append(p.getQuaternionFromEuler(x[t][j_idx:j_idx+dof]))
                j_idx += dof 
            tpos.append(temp)

    return tpos

#Convert from quaternion joints to euler joints
def quaternion_to_euler(x, dofs=None):
    tpos = []
    if dofs == None: #Joint list format
        for t in range(len(x)):
            temp = []
            for item in x[t]:
                if len(item) == 1:
                    temp.extend(item)
                else:
                    temp.extend(p.getEulerFromQuaternion(item))
            tpos.append(temp)
    else: #Numpy array format
        for t in range(len(x)):
            j_idx = 0 
            temp = []
            for dof in dofs:
                if dof == 1:
                    temp.append(x[t][j_idx:j_idx+dof])
                else:
                    temp.append(p.getEulerFromQuaternion(x[t][j_idx:j_idx+dof]))
                j_idx += dof 
            tpos.append(temp)

    return tpos

def array_to_joint_list(x, dofs):
    out = []
    for t in range(x.shape[0]):
        j_idx = 0
        temp = []
        for dof in dofs:
            if dof == 1:
                temp.append(tuple(x[t][j_idx:j_idx+dof]))
            else:
                temp.append(tuple(x[t][j_idx:j_idx+dof]))
            j_idx += dof
        out.append(temp)
    return out

#remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size = 3): 
    import copy
    from scipy.signal import medfilt
    filtered = copy.deepcopy(kpts)

    #apply median filter to get rid of poor keypoints estimations
    for j in range(kpts.shape[1]):
        xs = kpts[:,j,0]
        ys = kpts[:,j,1]
        zs = kpts[:,j,2]

        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)

        filtered[:,j] = np.stack([xs, ys, zs], axis = -1) 

    return filtered

#Estimate segment lengths from joint/marker positions
#Poses expected shape: Time x Joints x Dim
def compute_segment_lens(poses, layout='mscoco'):
    l_shin    = None
    r_shin    = None
    l_femur   = None
    r_femur   = None
    l_forearm = None
    r_forearm = None
    l_humerus = None
    r_humerus = None

    l_clavicle = None
    r_clavicle = None
    l_flank = None
    r_flank = None

    l_foot_h = None
    r_foot_h = None
    l_foot_b = None
    r_foot_b = None
    l_foot_s = None
    r_foot_s = None

    seg_lens = {}
    if layout.startswith('mscoco'):
        nose      = 0 
        left_eye  = 1 
        right_eye = 2 
        left_ear  = 3 
        right_ear = 4 
        #neck
        #thorax
        left_shoulder  = 5 
        right_shoulder = 6 
        left_elbow     = 7 
        right_elbow    = 8 
        left_wrist     = 9 
        right_wrist    = 10
        #pelvis
        left_hip        = 11
        right_hip       = 12
        left_knee       = 13
        right_knee      = 14
        left_ankle      = 15
        right_ankle     = 16

        if layout == 'mscoco_foot':
            left_heel       = 19
            right_heel      = 22
            left_big_toe    = 17
            right_big_toe   = 20
            left_small_toe  = 18
            right_small_toe = 21
    else:
        pass

    l_shin    = np.mean(np.linalg.norm(poses[:,left_ankle] - poses[:,left_knee], axis=-1))
    r_shin    = np.mean(np.linalg.norm(poses[:,right_ankle] - poses[:,right_knee], axis=-1))

    l_femur   = np.mean(np.linalg.norm(poses[:,left_knee] - poses[:,left_hip], axis=-1))
    r_femur   = np.mean(np.linalg.norm(poses[:,right_knee] - poses[:,right_hip], axis=-1))

    l_forearm = np.mean(np.linalg.norm(poses[:,left_wrist] - poses[:,left_elbow], axis=-1))
    r_forearm = np.mean(np.linalg.norm(poses[:,right_wrist] - poses[:,right_elbow], axis=-1))

    l_humerus = np.mean(np.linalg.norm(poses[:,left_elbow] - poses[:,left_shoulder], axis=-1))
    r_humerus = np.mean(np.linalg.norm(poses[:,right_elbow] - poses[:,right_shoulder], axis=-1))

    shoulder_width = np.mean(np.linalg.norm(poses[:,right_shoulder] - poses[:,left_shoulder], axis=-1))
    l_clavicle = shoulder_width/2
    r_clavicle = shoulder_width/2

    pelvis = np.mean(np.linalg.norm(poses[:,right_hip] - poses[:,left_hip], axis=-1))
    r_flank = np.mean(np.linalg.norm(poses[:,right_hip] - poses[:,right_shoulder], axis=-1))
    l_flank = np.mean(np.linalg.norm(poses[:,left_hip] - poses[:,left_shoulder], axis=-1))

    if layout in ['mscoco_foot']:
        l_foot_h = np.mean(np.linalg.norm(poses[:,left_ankle] - poses[:,left_heel], axis=-1))
        r_foot_h = np.mean(np.linalg.norm(poses[:,right_ankle] - poses[:,right_heel], axis=-1))

        l_foot_b = np.mean(np.linalg.norm(poses[:,left_ankle] - poses[:,left_big_toe], axis=-1))
        r_foot_b = np.mean(np.linalg.norm(poses[:,right_ankle] - poses[:,right_big_toe], axis=-1)) 

        l_foot_s = np.mean(np.linalg.norm(poses[:,left_ankle] - poses[:,left_small_toe], axis=-1))
        r_foot_s = np.mean(np.linalg.norm(poses[:,right_ankle] - poses[:,right_small_toe], axis=-1))

    seg_lens['l_shin'] = l_shin 
    seg_lens['r_shin'] = r_shin
    seg_lens['l_femur'] = l_femur
    seg_lens['r_femur'] = r_femur
    seg_lens['l_forearm'] = l_forearm
    seg_lens['r_forearm'] = r_forearm
    seg_lens['l_humerus'] = l_humerus
    seg_lens['r_humerus'] = r_humerus
    seg_lens['l_clavicle'] = l_clavicle
    seg_lens['r_clavicle'] = r_clavicle
    seg_lens['pelvis'] = pelvis
    seg_lens['l_flank'] = l_flank
    seg_lens['r_flank'] = r_flank
    seg_lens['l_foot_h'] = l_foot_h
    seg_lens['r_foot_h'] = r_foot_h
    seg_lens['l_foot_b'] = l_foot_b
    seg_lens['r_foot_b'] = r_foot_b
    seg_lens['l_foot_s'] = l_foot_s
    seg_lens['r_foot_s'] = r_foot_s

    return seg_lens

#Constrain the segment lengths of poses to be within tolerance% of segment_lengths 
def constrain_segment_lens(poses, segment_lengths, layout='mscoco', tolerance=0.0):
    #Start from right hip joint and traverse outwards

    if layout.startswith('mscoco'):
        nose      = 0
        left_eye  = 1
        right_eye = 2
        left_ear  = 3
        right_ear = 4
        #neck
        #thorax
        left_shoulder  = 5
        right_shoulder = 6
        left_elbow     = 7
        right_elbow    = 8
        left_wrist     = 9
        right_wrist    = 10
        #pelvis
        left_hip        = 11
        right_hip       = 12
        left_knee       = 13
        right_knee      = 14
        left_ankle      = 15
        right_ankle     = 16

        if layout == 'mscoco_foot':
            left_heel       = 19
            right_heel      = 22
            left_big_toe    = 17
            right_big_toe   = 20
            left_small_toe  = 18
            right_small_toe = 21
    else:
        left_heel       = 19
        right_heel      = 22
        left_big_toe    = 17
        right_big_toe   = 20
        left_small_toe  = 18
        right_small_toe = 21

    #Pelvis
    poses[:,left_hip] = normalize_segment(segment_lengths['pelvis'], poses[:,right_hip], poses[:,left_hip])

    #Right Flank
    poses[:,right_shoulder] = normalize_segment(segment_lengths['r_flank'], poses[:,right_hip], poses[:,right_shoulder])

    #Right Humerus
    poses[:,right_elbow] = normalize_segment(segment_lengths['r_humerus'], poses[:,right_shoulder], poses[:,right_elbow])

    #Right Forearm
    poses[:,right_wrist] = normalize_segment(segment_lengths['r_forearm'], poses[:,right_elbow], poses[:,right_wrist])

    #Right Femur
    poses[:,right_knee] = normalize_segment(segment_lengths['r_femur'], poses[:,right_hip], poses[:,right_knee])

    #Right Shin
    poses[:,right_ankle] = normalize_segment(segment_lengths['r_shin'], poses[:,right_knee], poses[:,right_ankle])

    #Left Flank
    poses[:,left_shoulder] = normalize_segment(segment_lengths['l_flank'], poses[:,left_hip], poses[:,left_shoulder])

    #Left Humerus
    poses[:,left_elbow] = normalize_segment(segment_lengths['l_humerus'], poses[:,left_shoulder], poses[:,left_elbow])

    #Left Forearm
    poses[:,left_wrist] = normalize_segment(segment_lengths['l_forearm'], poses[:,left_elbow], poses[:,left_wrist])

    #Left Femur
    poses[:,left_knee] = normalize_segment(segment_lengths['l_femur'], poses[:,left_hip], poses[:,left_knee])

    #Left Shin
    poses[:,left_ankle] = normalize_segment(segment_lengths['l_shin'], poses[:,left_knee], poses[:,left_ankle])

    if layout in ['mscoco_foot']:
        #Right Heel
        poses[:,right_heel] = normalize_segment(segment_lengths['r_foot_h'], poses[:,right_ankle], poses[:,right_heel])

        #Right Big Toe
        poses[:,right_big_toe] = normalize_segment(segment_lengths['r_foot_b'], poses[:,right_ankle], poses[:,right_big_toe])

        #Right Small Toe
        poses[:,right_small_toe] = normalize_segment(segment_lengths['r_foot_s'], poses[:,right_ankle], poses[:,right_small_toe])

        #Left Heel
        poses[:,left_heel] = normalize_segment(segment_lengths['l_foot_h'], poses[:,left_ankle], poses[:,left_heel])

        #Left Big Toe
        poses[:,left_big_toe] = normalize_segment(segment_lengths['l_foot_b'], poses[:,left_ankle], poses[:,left_big_toe])

        #Left Small Toe
        poses[:,left_small_toe] = normalize_segment(segment_lengths['l_foot_s'], poses[:,left_ankle], poses[:,left_small_toe])

    return poses

def normalize_segment(seg_len, joint_a, joint_b):
    diff = joint_a - joint_b
    new_vec = (diff/np.linalg.norm(diff, axis=-1)[:,None]) * seg_len

    return joint_a - new_vec
