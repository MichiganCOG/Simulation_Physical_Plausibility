#Anything dealing with keypoints

import numpy as np

from math import *
from tools.transformation import *
#from pyquaternion import Quaternion

def h36m_to_mscoco(h36m_kpts):
    T,K,C = h36m_kpts.shape 
    mscoco_kpts = -1*np.ones((T,17,C))

    #Does not have eyes and ear points
    h36m_mscoco = { 
    14:0, # nose
    -1:1, # left_eye
    -1:2, # right_eye
    -1:3, # left_ear
    -1:4, # right_ear
    17:5, # left_shoulder
    25:6, # right_shoulder
    18:7, # left_elbow
    26:8, # right_elbow
    19:9, # left_wrist
    27:10, # right_wrist
    6:11, # left_hip
    1:12, # right_hip
    7:13, # left_knee
    2:14, # right_knee
    8:15, # left_ankle
    3:16, # right_ankle
    }   

    for k,v in h36m_mscoco.items():
        if k != -1 and v != -1: 
            mscoco_kpts[:,v] = h36m_kpts[:,k]

    return mscoco_kpts

#Determine the best 3D joint locations to reproduce the 2D joint observations
#Expected shape: joints (N x J x C)
def triangulate(joints, K, R, T, iterations=100):
    #Determine the number of views
    num_views = len(joints)

    '''
    #Determine if any of the joints were just missed
    masks = []
    for i in range(len(joints)):
      masks.append(np.logical_and(joints[i][:,0] < 0, joints[i][:,1] < 0))
    masks = np.asarray(masks)
    masks = np.any(masks, axis=0)
    '''

    #Invert R and T
    Rinv = [r for r in R]
    Tinv = [t for t in T]
    for i in range(len(R)):
        Rinv[i] = np.linalg.inv(R[i])
        Tinv[i] = -Rinv[i]@T[i]

    #Solve triangulation for each point
    joints_3d = []
    for i in range(len(joints[0])):
        #Create placeholders for each point of origin and direction
        origins = np.zeros((0, 3)) 
        directions = np.zeros((0, 3)) 
        #Iterate through each set of joint locations
        for j in range(len(joints)):
            if joints[j][i][0] < 0 or joints[j][i][0] < 0: #Don't triangulate masked out keypoints
                continue
            #Get a line in world coordinates, add to the list of lines
            origin = Tinv[j]
            #Determine the direction of the line (in image space)
            u = (joints[j][i][0] - K[j][0,2])/K[j][0,0]
            v = (joints[j][i][1] - K[j][1,2])/K[j][1,1]
            w = 1.0 
            d = np.array([u,v,w])
            #Normalize
            d = d / np.linalg.norm(d)
            #Convert line into world coordinate frame
            d = np.mat(Rinv[j])*np.mat(d.reshape((3,1)))
            directions = np.concatenate((directions, d.reshape(1,3)), axis=0)
            origins = np.concatenate((origins, origin.reshape(1,3)), axis=0)
   
        #Use the set of lines to determine the nearest 3D point to each
        joints_3d.append(find_nearest_point(origins, directions))
  
    #Correct for missed keypoints
    joints_3d = np.asarray(joints_3d)
    #joints_3d[masks] = np.full((3,), -1e5)

    return joints_3d

def find_nearest_point(a, d):
    m = np.zeros((3,3))
    b = np.zeros((3,1))
    for i in range(d.shape[0]):
        d2 = d[i].dot(d[i].reshape((3,1)))[0,0]
        da = d[i].dot(a[i].reshape((3,1)))[0,0]
        for ii in range(3):
            for jj in range(3):
                m[ii][jj] += d[i,ii] * d[i,jj]
            m[ii][ii] -= d2
            b[ii,0] += d[i,ii] * da - a[i,ii] * d2
    try:
        p = np.linalg.inv(m).dot(b)
    except np.linalg.LinAlgError as err:
        p = np.zeros((3,1), dtype=np.float32)
    return np.squeeze(p)

#Estimate segment lengths from joint/marker positions
#Poses expected shape: Time x Joints x Dim
#DEPRECATED
def est_segment_lens(poses, layout='mscoco'):
    seg_lens = {}

    if layout == 'mscoco':
        l_shin    = np.mean(np.sqrt(np.sum((poses[:,15] - poses[:,13])**2, axis=-1)))
        r_shin    = np.mean(np.sqrt(np.sum((poses[:,16] - poses[:,14])**2, axis=-1)))

        l_femur   = np.mean(np.sqrt(np.sum((poses[:,13] - poses[:,11])**2, axis=-1)))
        r_femur   = np.mean(np.sqrt(np.sum((poses[:,14] - poses[:,12])**2, axis=-1)))

        l_forearm = np.mean(np.sqrt(np.sum((poses[:,9] - poses[:,7])**2, axis=-1)))
        r_forearm = np.mean(np.sqrt(np.sum((poses[:,10] - poses[:,8])**2, axis=-1)))

        l_humerus = np.mean(np.sqrt(np.sum((poses[:,7] - poses[:,5])**2, axis=-1)))
        r_humerus = np.mean(np.sqrt(np.sum((poses[:,8] - poses[:,6])**2, axis=-1)))

        shoulder_width = np.mean(np.sqrt(np.sum((poses[:,6] - poses[:,5])**2, axis=-1)))
        l_clavicle = shoulder_width/2
        r_clavicle = shoulder_width/2

        pelvis = np.mean(np.sqrt(np.sum((poses[:,12] - poses[:,11])**2, axis=-1)))

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

    elif layout == 'fullbody':
        l_shin    = np.mean(np.sqrt(np.sum((poses[:,3] - poses[:,2])**2, axis=-1)))
        r_shin    = np.mean(np.sqrt(np.sum((poses[:,7] - poses[:,6])**2, axis=-1)))

        l_femur   = np.mean(np.sqrt(np.sum((poses[:,2] - poses[:,1])**2, axis=-1)))
        r_femur   = np.mean(np.sqrt(np.sum((poses[:,6] - poses[:,5])**2, axis=-1)))

        l_forearm = np.mean(np.sqrt(np.sum((poses[:,15] - poses[:,14])**2, axis=-1)))
        r_forearm = np.mean(np.sqrt(np.sum((poses[:,19] - poses[:,18])**2, axis=-1)))

        l_humerus = np.mean(np.sqrt(np.sum((poses[:,14] - poses[:,13])**2, axis=-1)))
        r_humerus = np.mean(np.sqrt(np.sum((poses[:,18] - poses[:,17])**2, axis=-1)))

        pelvis = np.mean(np.sqrt(np.sum((poses[:,5] - poses[:,1])**2, axis=-1)))

        seg_lens['l_shin'] = l_shin
        seg_lens['r_shin'] = r_shin
        seg_lens['l_femur'] = l_femur
        seg_lens['r_femur'] = r_femur
        seg_lens['l_forearm'] = l_forearm
        seg_lens['r_forearm'] = r_forearm
        seg_lens['l_humerus'] = l_humerus
        seg_lens['r_humerus'] = r_humerus
        seg_lens['pelvis'] = pelvis

    return seg_lens

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

#Compute center of mass based on assumed body weight percentages
#Assuming mscoco_foot format
def compute_CoM(kpts, total_mass=1.0):
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
    left_heel       = 19
    right_heel      = 22
    left_big_toe    = 17
    right_big_toe   = 20
    left_small_toe  = 18
    right_small_toe = 21

    #Segments weights as percentage of body weight (source: Harless via Ramachandran)
    head_neck = total_mass*(7.6/100)
    torso     = total_mass*(44.2/100)
    upper_arm = total_mass*(3.2/100)
    lower_arm = total_mass*(1.7/100)
    hand      = total_mass*(0.9/100)
    thigh     = total_mass*(11.9/100)
    calf      = total_mass*(4.6/100)
    foot      = total_mass*(2.0/100)

    head_cr  = np.mean(kpts[:5],axis=0)
    torso_cr = np.mean(kpts[[left_shoulder,right_shoulder,left_hip,right_hip]],axis=0)
    l_up_arm_cr = np.mean(kpts[[left_shoulder, left_elbow]],axis=0)
    r_up_arm_cr = np.mean(kpts[[right_shoulder, right_elbow]], axis=0)
    l_lo_arm_cr = np.mean(kpts[[left_elbow, left_wrist]],axis=0)
    r_lo_arm_cr = np.mean(kpts[[right_elbow, right_wrist]], axis=0)
    l_hand_cr   = kpts[left_wrist]
    r_hand_cr   = kpts[right_wrist]
    l_thigh_cr  = np.mean(kpts[[left_hip, left_knee]], axis=0)
    r_thigh_cr  = np.mean(kpts[[right_hip, right_knee]], axis=0)
    l_calf_cr  = np.mean(kpts[[left_knee, left_ankle]], axis=0)
    r_calf_cr  = np.mean(kpts[[right_knee, right_ankle]], axis=0)
    try:
        l_foot_cr  = np.mean(kpts[[left_heel, left_big_toe, left_small_toe]], axis=0)
        r_foot_cr  = np.mean(kpts[[right_heel, right_big_toe, right_small_toe]], axis=0)
    except IndexError: #If foot keypoints don't exist
        l_foot_cr = l_calf_cr
        r_foot_cr = r_calf_cr

    weights = [head_neck, torso, upper_arm, upper_arm, lower_arm, lower_arm, hand, hand, thigh, thigh, calf, calf, foot, foot]
    centers = np.stack([head_cr, torso_cr, l_up_arm_cr, r_up_arm_cr, l_lo_arm_cr, r_lo_arm_cr, l_hand_cr, r_hand_cr,\
            l_thigh_cr, r_thigh_cr, l_calf_cr, r_calf_cr, l_foot_cr, r_foot_cr])

    return np.average(centers, axis=0, weights=weights)

#remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size = 3): 
    import copy
    from scipy.signal import medfilt
    filtered = copy.deepcopy(kpts)

    #apply median filter to get rid of poor keypoints estimations
    for j in range(kpts.shape[1]):
        xs = kpts[:,j,0]
        ys = kpts[:,j,1]

        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)

        if kpts.shape[-1] == 3:
            zs = kpts[:,j,2]
            zs = medfilt(zs, window_size)
            filtered[:,j] = np.stack([xs, ys, zs], axis = -1) 
        else:
            filtered[:,j] = np.stack([xs, ys], axis = -1) 

    return filtered

#Nimble Physics functions
def get_world_pos(skel, skel_keys):
    skel_pose = []
    skel_coords = skel.getJointWorldPositionsMap()

    for k in skel_keys:
        skel_pose.append(skel_coords[k])
    skel_pose = np.array(skel_pose)

    return skel_pose

def fit_to_sequence(skel, skel_joints, kpt_positions):
    out_joint_angles = []
    next_vel = []

    init_positions = skel.getPositions()
    #coords = [get_world_pos(skel, skel_keys)]
    for i, kpts in enumerate(kpt_positions):
        err = skel.fitJointsToWorldPositions(skel_joints, kpts.flatten())
        out_joint_angles.append(skel.getPositions())

    skel.setPositions(init_positions)
    return np.array(out_joint_angles)

#Adopted from: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/inverse_kinematics.py
#Retrieve joint angles directly from keypoints
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
def get_joint_rotations(pose, layout='mscoco'):
    eps = 0.001
    axis_rotate_rate = 0.3

    pose = np.array(pose, dtype=np.float32)
    includeFoot = False #Includes toe keypoints

    head_top = None
    pelvis = (pose[12]+pose[11])/2 #(Root) Mid-point between hips
    neck   = (pose[5]+pose[6])/2 #Mid-point between shoulders 
    thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?

    nose = pose[0]
    leye = pose[1]
    reye = pose[2]
    lear = pose[3]
    rear = pose[4]

    lshould = pose[5]
    rshould = pose[6]
    lelbow  = pose[7]
    relbow  = pose[8]
    lwrist  = pose[9]
    rwrist  = pose[10]

    lhip   = pose[11]
    rhip   = pose[12]
    lknee  = pose[13]
    rknee  = pose[14]
    lankle = pose[15]
    rankle = pose[16]

    if layout == 'mscoco_foot':
        lbigtoe   = pose[17]
        lsmalltoe = pose[18]
        lheel     = pose[19]
        rbigtoe   = pose[20]
        rsmalltoe = pose[21]
        rheel     = pose[22]

        includeFoot = True

    tmp = [[] for i in range(15)]
    # duration of frame in seconds (1D),
    tmp[0] = [1/50.]
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
    return np.array(ret), tmp

def get_pelvis(pose, layout='mscoco'):
    pelvis = (pose[12]+pose[11])/2 #(Root) Mid-point between hips
    return pelvis

def get_root_orn(pose, layout='mscoco'):
    pose = np.array(pose, dtype=np.float32)

    pelvis = (pose[12]+pose[11])/2 #(Root) Mid-point between hips
    neck   = (pose[5]+pose[6])/2 #Mid-point between shoulders 
    thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?

    lhip   = pose[11]
    rhip   = pose[12]
    lknee  = pose[13]
    rknee  = pose[14]

    # root rotation (4D),
    root_y = (thorax - pelvis)
    root_z = (rhip - pelvis)
    root_x = np.cross(root_y, root_z)

    x = np.array([1.0, 0, 0])
    y = np.array([0, 1.0, 0])
    z = np.array([0, 0, 1.0])

    rot_qua,rot_mat = get_quaternion(root_x, root_y, root_z, x, y, z)

    return rot_qua
