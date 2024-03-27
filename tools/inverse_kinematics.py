#Manual inverse kinematics
#Adopted from: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/inverse_kinematics.py

import numpy as np
from math import *
import pickle
import json
from tools.transformation import *

from pyquaternion import Quaternion

def get_angle(vec1, vec2):
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return acos(cos_theta)

def get_quaternion(ox, oy, oz, x, y, z):
    #Given transformed axis in x-y-z order return a quaternion
    ox /= np.linalg.norm(ox)
    oy /= np.linalg.norm(oy)
    oz /= np.linalg.norm(oz)

    set1 = np.vstack((ox, oy, oz))

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    set2 = np.vstack((x, y, z))

    rot_mat = superimposition_matrix(set1, set2, scale=False, usesvd=True)
    rot_qua = quaternion_from_matrix(rot_mat)

    return rot_qua, rot_mat[:3,:3]

# 3D coord to deepmimic rotations
def coord_to_rot(frameNum, frame, frame_duration, pose_format='mscoco', rotate_ankles=False):
    eps = 0.001
    axis_rotate_rate = 0.3

    frame = np.array(frame)
    includeFoot = False #Includes toe keypoints

    if pose_format == 'mocap_47':
        #NOTE: Surface markers may require additional adjustments
        head_top = frame[46]
        pelvis   = (frame[2]+frame[23]+frame[18]+frame[39])/4 #(Root) Mid-point hips and posterior joints
        neck     = (frame[0]+frame[44])/2 #T10 and clavicle
        thorax   = frame[43]
        t1       = frame[44]

        nose = None
        leye = None
        reye = None
        lear = None
        rear = None

        lshould = frame[1]
        rshould = frame[22]
        lelbow  = (frame[5]+frame[12])/2 #Between lateral and medial
        relbow  = (frame[26]+frame[33])/2 #Between lateral and medial
        lwrist  = (frame[10]+frame[17])/2 #Between lateral and medial
        rwrist  = (frame[31]+frame[38])/2 #Between lateral and medial
        #lelbow  = frame[5]
        #relbow  = frame[26]
        #lwrist  = frame[10]
        #rwrist  = frame[31]

        lhip   = frame[2]
        rhip   = frame[23]
        lknee  = frame[8] #lateral only
        rknee  = frame[29] #lateral only
        lankle = frame[4] #lateral only
        rankle = frame[25] #lateral only

        #Fix hips to closer to center of mass, rather than surface
        lhip[2] = pelvis[2]
        rhip[2] = pelvis[2]

        lbigtoe   = frame[16]
        lsmalltoe = frame[9]
        lheel     = frame[3]
        rbigtoe   = frame[37]
        rsmalltoe = frame[30]
        rheel     = frame[24]

        includeFoot = True

    elif pose_format == 'h36m_17':
        head_top = frame[10]
        pelvis   = frame[0] 
        nose     = frame[9]
        neck     = frame[8]

        thorax = frame[7] #or spine. Halfway between pelvis and neck
        t1 = nose

        lshould = frame[11]
        rshould = frame[14]
        lelbow  = frame[12]
        relbow  = frame[15]
        lwrist  = frame[13]
        rwrist  = frame[16]

        lhip   = frame[4]
        rhip   = frame[1]
        lknee  = frame[5]
        rknee  = frame[2]
        lankle = frame[6]
        rankle = frame[3]

        includeFoot = False

    elif pose_format == 'physcap':
        head_top = frame[0]
        pelvis   = (frame[2] + frame[6])/2
        neck     = frame[1]

        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?
        t1 = thorax

        lshould = frame[10]
        rshould = frame[13]
        lelbow  = frame[11]
        relbow  = frame[14]
        lwrist  = frame[12]
        rwrist  = frame[15]

        lhip   = frame[2]
        rhip   = frame[6]
        lknee  = frame[3]
        rknee  = frame[7]
        lankle = frame[4] 
        rankle = frame[8]

        includeFoot = False

        #Index 5: Left Toe
        #Index 9: Right Toe

    else:
        head_top = None
        pelvis = (frame[12]+frame[11])/2 #(Root) Mid-point between hips
        neck   = (frame[5]+frame[6])/2 #Mid-point between shoulders 
        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?

        nose = frame[0]
        leye = frame[1]
        reye = frame[2]
        lear = frame[3]
        rear = frame[4]

        lshould = frame[5]
        rshould = frame[6]
        lelbow  = frame[7]
        relbow  = frame[8]
        lwrist  = frame[9]
        rwrist  = frame[10]

        lhip   = frame[11]
        rhip   = frame[12]
        lknee  = frame[13]
        rknee  = frame[14]
        lankle = frame[15]
        rankle = frame[16]

    if pose_format == 'mscoco_foot':
        lbigtoe   = frame[17]
        lsmalltoe = frame[18]
        lheel     = frame[19]
        rbigtoe   = frame[20]
        rsmalltoe = frame[21]
        rheel     = frame[22]

        includeFoot = True

    tmp = [[] for i in range(15)]
    # duration of frame in seconds (1D),
    tmp[0] = [frame_duration]
    # root position (3D),
    tmp[1] = pelvis
    # root rotation (4D),
    root_y = (thorax - pelvis)
    root_z = (rhip - pelvis)
    root_x = np.cross(root_y, root_z)

    x = np.array([1.0, 0, 0])
    y = np.array([0, 1.0, 0])
    z = np.array([0, 0, 1.0])

    rot_qua,rot_mat = get_quaternion(root_x, root_y, root_z, x, y, z)
    tmp[2] = list(rot_qua)

    # chest rotation (4D),
    chest_y = (thorax - pelvis)
    chest_z = (rshould - neck)
    chest_x = np.cross(chest_y, chest_z)
    rot_qua,rot_mat = get_quaternion(chest_x, chest_y, chest_z, root_x, root_y, root_z)
    tmp[3] = list(rot_qua)

    # neck rotation (4D),
    if head_top is not None:
        neck_y = (head_top - neck)
        neck_z = np.cross(head_top - t1, neck - t1)
    else:
        eye_mid = (leye+reye)/2 #Only for rotation measures
        neck_y = (neck - pelvis)
        neck_z = (rshould - neck)
        #neck_z = np.cross(eye_mid-nose, neck-thorax)
    neck_x = np.cross(neck_y, neck_z)
    rot_qua,rot_mat = get_quaternion(neck_x, neck_y, neck_z, chest_x, chest_y, chest_z)
    tmp[4] = list(rot_qua)

    # right hip rotation (4D),
    r_hip_y = (rhip - rknee)
    #r_hip_z = np.cross(rhip - rknee, rankle - rknee)
    r_hip_z = (rhip - pelvis)
    r_hip_x = np.cross(r_hip_y, r_hip_z)
    rot_qua,rot_mat = get_quaternion(r_hip_x, r_hip_y, r_hip_z, root_x, root_y, root_z)
    tmp[5] = list(rot_qua)

    # right knee rotation (1D),
    vec1 = rhip - rknee 
    vec2 = rankle - rknee
    angle1 = get_angle(vec1, vec2)
    tmp[6] = [angle1 - np.pi]

    # right ankle rotation (4D),
    if includeFoot:
        r_ankle_x = 0.5*(rbigtoe+rsmalltoe) - rheel
        r_ankle_z = rsmalltoe - rbigtoe
        r_ankle_y = np.cross(r_ankle_z, r_ankle_x)

        #Obtain the rotation matrix of the knee, to relate to the ankle
        r_knee_y = rknee - rankle
        r_knee_z = (rhip - pelvis)
        r_knee_x = np.cross(r_knee_y, r_knee_z)

        #Manually defined standard basis. Will break if humanoid is rotated
        _x = [0.0, 0.0, -1.0]
        _y = [0.0, 1.0, 0.0]
        _z = [1.0, 0.0, 0.0]

        #rot_qua, rot_mat = get_quaternion(_x,_y,_z,r_knee_x,r_knee_y,r_knee_z)
        rot_qua,rot_mat = get_quaternion(r_ankle_x, r_ankle_y, r_ankle_z, r_knee_x,r_knee_y,r_knee_z)

        if rotate_ankles:
            temp,_ = get_quaternion(_x,_y,_z,r_ankle_x, r_ankle_y, r_ankle_z)
            #roughly the quaternion rotation from the local basis to the "standard" basis for the initial standing position
            #because the joints in the foot are not in-line, we will apply this rotation to make it the standard basis
            #so then any subsequent transformations should be relative to this position
            local_to_standard = np.array([0.94458129, -0.0030325, 0.32599256, -0.03854677])
            rot_qua = quaternion_multiply(rot_qua,local_to_standard)
        tmp[7] = list(rot_qua)
    else:
        tmp[7] = [1, 0, 0, 0]

    #  right shoulder rotation (4D),
    r_shou_y = (rshould - relbow)
    r_shou_z = np.cross(rwrist - relbow, rshould - relbow)
    r_shou_x = np.cross(r_shou_y, r_shou_z)
    rot_qua,rot_mat = get_quaternion(r_shou_x, r_shou_y, r_shou_z, chest_x, chest_y, chest_z)
    tmp[8] = list(rot_qua)

    # right elbow rotation (1D),
    vec1 = rshould - relbow
    vec2 = rwrist - relbow
    angle1 = get_angle(vec1, vec2)
    tmp[9] = [pi - angle1]

    # left hip rotation (4D),
    l_hip_y = (lhip - lknee)
    #l_hip_z = np.cross(lhip - lknee, lankle - lknee)
    l_hip_z = (pelvis - lhip)
    l_hip_x = np.cross(l_hip_y, l_hip_z)
    rot_qua,rot_mat = get_quaternion(l_hip_x, l_hip_y, l_hip_z, root_x, root_y, root_z)
    tmp[10] = list(rot_qua)

    # left knee rotation (1D),
    vec1 = lhip - lknee
    vec2 = lankle - lknee
    angle1 = get_angle(vec1, vec2)
    tmp[11] = [angle1 - np.pi]

    # left ankle rotation (4D),
    if includeFoot:
        l_ankle_x = 0.5*(lbigtoe+lsmalltoe) - lheel
        l_ankle_z = lbigtoe - lsmalltoe
        l_ankle_y = np.cross(l_ankle_z, l_ankle_x)

        #Obtain the rotation matrix of the knee, to relate to the ankle
        l_knee_y = lknee - lankle
        l_knee_z = (pelvis - lhip)
        l_knee_x = np.cross(l_knee_y, l_knee_z)

        #rot_qua,rot_mat = get_quaternion(_x,_y,_z,l_knee_x,l_knee_y,l_knee_z)
        rot_qua,rot_mat = get_quaternion(l_ankle_x, l_ankle_y, l_ankle_z, l_knee_x, l_knee_y, l_knee_z)
        
        if rotate_ankles:
            temp,_ = get_quaternion(_x,_y,_z,l_ankle_x,l_ankle_y,l_ankle_z)
            local_to_standard = np.array([0.97905839, 0.05950645,-0.18780428, 0.05131483])
            #Apply transformation so initial local basis is perceived as "standard"
            rot_qua = quaternion_multiply(rot_qua,local_to_standard)
        tmp[12] = list(rot_qua)
    else:
        tmp[12] = [1, 0, 0, 0]

    # left shoulder rotation (4D),
    l_shou_y = (lshould - lelbow)
    l_shou_z = np.cross(lwrist - lelbow, lshould - lelbow)
    l_shou_x = np.cross(l_shou_y, l_shou_z)
    rot_qua,rot_mat = get_quaternion(l_shou_x, l_shou_y, l_shou_z, chest_x, chest_y, chest_z)
    tmp[13] = list(rot_qua)

    # left elbow rotation (1D)
    vec1 = lshould - lelbow
    vec2 = lwrist - lelbow
    angle1 = get_angle(vec1, vec2)
    tmp[14] = [pi - angle1]

    ret = []
    for i in tmp:
        ret += list(i)
    return np.array(ret)

def coord_seq_to_rot_seq(coord_seq, frame_duration, pose_format):
    ret = []
    for i in range(len(coord_seq)):
        tmp = coord_to_rot(i, coord_seq[i], frame_duration, pose_format)
        ret.append(list(tmp))
    return ret
