import numpy as np
import matplotlib.pyplot as plt

import pybullet_data

import math
from scipy.spatial import Delaunay, ConvexHull

#Return kinematic reference humanoid. Not affected by gravity or friction
def get_kin_ref_humanoid(pb, agent_urdf, startPos, flags, isVisible):
    #Kinematics reference model
    uid = pb.loadURDF(agent_urdf,
                          startPos,
                          globalScaling=0.25,
                          useFixedBase=False,
                          flags=flags)

    #Above changes to kinematics reference model + make transparent, disable collisions
    alpha = 0
    if isVisible:
        alpha = 0.4
    pb.changeDynamics(uid, linkIndex=-1, linearDamping=0, angularDamping=0)
    pb.changeVisualShape(uid, -1, rgbaColor=[1,1,1,alpha])
    pb.setCollisionFilterGroupMask(uid, -1, collisionFilterGroup=0, collisionFilterMask=0)

    for j in range(pb.getNumJoints(uid)):
        pb.changeVisualShape(uid, j, rgbaColor=[1,1,1,alpha])
        pb.setCollisionFilterGroupMask(uid, j, collisionFilterGroup=0, collisionFilterMask=0)

    return uid

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

def is_balanced(pb, uid, leftAnkle, rightAnkle, ref_comv):
    '''
    Returns true if pose is both stationary and balanced

    ref_comv: The center-of-mass of the reference trajectory b/c it is assumed to never be 'falling'. 
                In this case, the center-of-mass velocity will be high but the pose should be stationary.
    '''
    aabbMinL,aabbMaxL = pb.getAABB(uid, leftAnkle)
    aabbMinR,aabbMaxR = pb.getAABB(uid, rightAnkle)

    foot_L = np.stack((aabbMinL, aabbMaxL))
    foot_R = np.stack((aabbMinR, aabbMaxR))
    convex_points = np.concatenate((foot_L, foot_R))[:,[0,2]]
    centroid = np.mean(convex_points, axis=0) #center of convex hull

    com, comv = computeCOMposVel(pb, uid)
    hull = Delaunay(convex_points)

    #centroid = np.mean(convex_points[hull.simplices], axis=0)
    #vectors = convex_points[hull.simplices] - centroid
    #expansion_factor = 1.1  # Adjust this factor as needed
    #expanded_points = centroid + vectors * expansion_factor
    #expanded_hull = Delaunay(expanded_points)

    #expand region around convex hull
    stretchCoef = 0.10
    convh = ConvexHull(convex_points)
    _convex_points = bufferPoints(convex_points[convh.vertices], stretchCoef, n=len(convex_points))
    hull = Delaunay(_convex_points)

    comv_thresh = 0.250
    #Only valid if agent is stationary, else assume pose is "balanced"
    if np.max(np.abs(ref_comv)) > comv_thresh:
        return True

    #Return true if COM (gravity projected) lies within convex hull
    is_balanced = hull.find_simplex([com[0], com[2]]) >= 0 

    if False and not is_balanced:
        plt.figure() 
        #plt.triplot(convex_points[:,0], convex_points[:,1], hull.simplices)
        #plt.plot(convex_points[:,0], convex_points[:,1], 'o')
        plt.triplot(_convex_points[:,0], _convex_points[:,1], hull.simplices)
        plt.scatter(_convex_points[:,0], _convex_points[:,1], color='r')
        plt.scatter(com[0],com[2],s=50,c='m')
        plt.scatter(centroid[0], centroid[1], s=100, c='b')
        plt.show()

    return is_balanced

def centroid_dist(pb, uid, leftAnkle, rightAnkle, ref_comv):
    '''
    Quick computation between CoG distance and centroid of foot keypoints
    Foot keypoints can also be viewed as a proxy of the base of support
    '''
    aabbMinL,aabbMaxL = pb.getAABB(uid, leftAnkle)
    aabbMinR,aabbMaxR = pb.getAABB(uid, rightAnkle)

    foot_L = np.stack((aabbMinL, aabbMaxL))
    foot_R = np.stack((aabbMinR, aabbMaxR))
    convex_points = np.concatenate((foot_L, foot_R))[:,[0,2]]
    centroid = np.mean(convex_points, axis=0) #center of convex hull

    com, comv = computeCOMposVel(pb, uid)
    dist = np.linalg.norm(centroid - com[[0,2]]) #distance between centroid of convex hull and COG

    if False:
        plt.figure() 
        plt.scatter(centroid[0], centroid[1], s=100, c='b')
        plt.scatter(com[0],com[2],s=50,c='m')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.show()

    return dist

def PointsInCircum(eachPoint,r,n=100):
    return [(eachPoint[0] + math.cos(2*math.pi/n*x)*r,eachPoint[1] + math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]


def bufferPoints (inPoints, stretchCoef, n):
    newPoints = []
    for eachPoint in inPoints:
        newPoints += PointsInCircum(eachPoint, stretchCoef, n)
    newPoints = np.array(newPoints)
    newBuffer = ConvexHull (newPoints)

    return newPoints[newBuffer.vertices]
