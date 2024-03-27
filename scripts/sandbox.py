import argparse
import os
import sys
DIR = os.getcwd()
sys.path.append(DIR)

import json
import pybullet as p
import time
import pybullet_data

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

time_step = 1./200

parser = argparse.ArgumentParser()

parser.add_argument('--useGUI', action='store_true', help='visualize with GUI')

parser.add_argument('--write_video', action='store_true', help='Write out video')
parser.add_argument('--out_video_name', default='temp.mp4', type=str, help='Output video name')

parser.set_defaults(useGUI=False)
parser.set_defaults(write_video=False)

args = parser.parse_args()

useGUI = args.useGUI
use_presets = False


#Write video
write_video = args.write_video
vid_name    = args.out_video_name
vid_options = options="--mp4=\"./out_videos/"+vid_name+"\" --mp4fps=200"

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13

joint_indices = [chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder,\
                rightElbow, leftHip, leftKnee, leftAnkle, leftShoulder, leftElbow]
joint_names   = ['chest', 'neck', 'rightHip', 'rightKnee', 'rightAnkle', 'rightShoulder',\
                'rightElbow', 'leftHip', 'leftKnee', 'leftAnkle', 'leftShoulder', 'leftElbow']
joint_dofs    = [3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1]
joint_forces  = [200., 50., 200., 150., 90., 100., 60., 200., 150., 90., 100., 60.]
maxForces = []

for jd, jf in zip(joint_dofs, joint_forces): 
    maxForces.append([jf] * jd)

def main():
    if useGUI:
        if write_video:
            physicsClient = p.connect(p.GUI, options=vid_options)
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        else:
            physicsClient = p.connect(p.GUI)

    else:
        if write_video:
            physicsClient = p.connect(p.DIRECT, options=vid_options) #p.DIRECT for non-graphical version
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        else:
            physicsClient = p.connect(p.DIRECT) #p.DIRECT for non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0) #Disable mouse picking
    p.setGravity(0,-9.8,0)

    p.resetDebugVisualizerCamera(cameraDistance=1.25,
                                 cameraYaw=-0,
                                 cameraPitch=-5,
                                 cameraTargetPosition=[0, 0.35, 0])

    y2zOrn = p.getQuaternionFromEuler([-1.57, 0, 0]) 
    planeId  = p.loadURDF('data/plane.urdf', [0, -0.1, 0], baseOrientation=y2zOrn)

    startPos = [0,1,0]
    flags    = p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION
    humanoid = p.loadURDF('humanoid/humanoid.urdf',
                          startPos,
                          globalScaling=0.25,
                          useFixedBase=True,
                          flags=flags)
    
    startPos = [1,0,0]

    #Change plane friction to 0.9
    p.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)

    #Assuming this is the dictionary
    joint_types = {0:'joint_revolute',
                   1:'joint_prismatic',    
                   2:'joint_spherical',    
                   3:'joint_planar',    
                   4:'joint_fixed'}

        
    p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)

    jointIds = {}

    for obj in [humanoid]:
        numJoints = p.getNumJoints(obj)
        print(str(numJoints)+' joints')
        total_mass = 0
        for j in range(numJoints):
            joint_info = p.getJointInfo(obj, j)

            joint_index = joint_info[0]
            joint_name  = joint_info[1]
            joint_type  = joint_types[joint_info[2]]
            joint_frict = joint_info[7]
            joint_ll    = joint_info[8]
            joint_ul    = joint_info[9]
            link_name   = joint_info[12]

            joint_mass  = p.getDynamicsInfo(obj, j)[0]
            total_mass += joint_mass 

            print('{} - {} - {} - ll: {} - ul: {} - link: {}, {} kg'.format(joint_index, joint_name, joint_type, joint_ll, joint_ul, link_name, joint_mass))
            
            #Add debug controls
            p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)
            if (joint_info[2] == p.JOINT_PRISMATIC or joint_info[2] == p.JOINT_REVOLUTE):
                paramId = p.addUserDebugParameter(joint_name.decode('utf-8'),joint_ll,joint_ul,0)
                jointIds[j] = paramId
            elif joint_info[2] == p.JOINT_SPHERICAL:
                if joint_ll > 0:
                    paramIdX = p.addUserDebugParameter(joint_name.decode('utf-8')+'X',-1*joint_ll,joint_ll,0)
                else:
                    paramIdX = p.addUserDebugParameter(joint_name.decode('utf-8')+'X',-3.14,3.14,0)
                if joint_ul > 0:
                    paramIdY = p.addUserDebugParameter(joint_name.decode('utf-8')+'Y',-1*joint_ul,joint_ul,0)
                else:
                    paramIdY = p.addUserDebugParameter(joint_name.decode('utf-8')+'Y',-3.14,3.14,0)
                paramIdZ = p.addUserDebugParameter(joint_name.decode('utf-8')+'Z',-3.14,3.14,0)
                jointIds[j] = [paramIdX, paramIdY, paramIdZ]

        print('Total mass: {}'.format(total_mass))
        print('--'*30)

    controlMode = p.POSITION_CONTROL

    #Enable force sensors
    #p.enableJointForceTorqueSensor(humanoid, leftAnkle)
    #p.enableJointForceTorqueSensor(humanoid, rightAnkle)
    
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

    link_state_args = ['linkWorldPosition',
    'linkWorldOrientation',
    'localInertialFramePosition',
    'localInertialFrameOrientation',
    'worldLinkFramePosition',
    'worldLinkFrameOrientation']

    def drawAABB(aabbMin, aabbMax):
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 0, 0]) 
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 1, 0]) 
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 1]) 

        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMax[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMin[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1]) 

    aabb = p.getAABB(humanoid, leftAnkle)
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    print(aabb)
    drawAABB(aabbMin, aabbMax)
    p.addUserDebugPoints([aabbMin, aabbMax],[[0,0,255],[0,0,255]],5) 

    while True:

        for i in jointIds.keys():
            if type(jointIds[i]) == list:
                cx,cy,cz = jointIds[i]
                tx = p.readUserDebugParameter(cx)
                ty = p.readUserDebugParameter(cy)
                tz = p.readUserDebugParameter(cz)

                targetPos = p.getQuaternionFromEuler([tx,ty,tz])
                p.setJointMotorControlMultiDof(
                                      humanoid,
                                      i,
                                      controlMode,
                                      targetPos,
                                      force=[5*240]*3
                                      )
            else:
                c = jointIds[i]
                targetPos = p.readUserDebugParameter(c)
                p.setJointMotorControl2(
                                      humanoid,
                                      i,
                                      controlMode,
                                      targetPos,
                                      force=5*240,
                                      )
        link_states = p.getLinkStates(humanoid, joint_indices)

        p.stepSimulation()
        if write_video:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        else:
            time.sleep(time_step)

    p.disconnect()

if __name__ == "__main__":
    main()
