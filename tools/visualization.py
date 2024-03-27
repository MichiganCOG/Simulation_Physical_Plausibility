import sys
sys.path.append('./')

import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from tools.keypoints import compute_CoM, median_filter

from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

#MSCOCO skeleton: layout='mscoco'
mscoco_links = [[16,14],[14,12],[17,15],[15,13],
                 [12,13],[6,12], [7,13],[6,7],
                 [6,8],[7,9],[8,10],[9,11],[2,3],
                 [1,2],[1,3],[2,4],[3,5],[1,6],[1,7]]

#MSCOCO skeleton: layout='mscoco_foot' Includes markers for toes and heel
#1-indexed
mscoco_foot_links = [[16,14],[14,12],[17,15],[15,13],
                 [12,13],[6,12], [7,13],[6,7],
                 [6,8],[7,9],[8,10],[9,11],[2,3],
                 [1,2],[1,3],[2,4],[3,5],[1,6],[1,7],
                 [16,20],[18,20],[19,20],
                 [17,23],[21,23],[22,23]]
#mocap pose
#0-indexed
mocap_links = [[46,0],[0,1],[0,22],[1,21],[22,42],
               [21,5],[5,6],[6,10],[10,7], #lateral
               [0,43],[43,2],[2,20],[43,23],[23,41],
               [20,8],[8,19],[19,4],[4,9],#lateral
               [41,29],[29,40],[40,25],[25,30], #lateral
               [42,26],[26,27],[27,31],[31,28], #lateral
               #[5,12],[12,13],[13,17],[17,14],[10,17], #medial
               #[26,33],[33,34],[34,38],[38,35],[31,38], #medial
               #[15,8],[15,11],[11,16],[4,11],[9,16], #medial
               #[29,36],[36,32],[32,37],[25,32],[30,37], #medial
               #[18,20],[2,18],[23,29],[44,45], #posterior
               ]
 
#Human3.6m joints skeleton: layout='h36m'. 1-indexed values
h36m_links = [[11,10],[7,8],
              [7,1], [1,13],[13,14],[9,10],
              [1,2],[2,3],[4,5],[5,6],
              [15,16], [15,14], [18,19], [26,27], [26,2],
              [14,18], [14,26], [27,28], [18,7],
              [3,4], [8,9], [28,30], [28,31],
              [19,20], [20,22], [22,23]
             ]
            #Redundant: 31-32, 28-29, 23-24, 20-21, 14-17-25, 1-12

#Human3.6m joints skeleton: layout='h36m_17'. 1-indexed values
h36m_17_links = [[1,0],[2,1],[3,2],[4,0],[5,4],
                 [6,5],[7,0],[8,7],[9,8],[10,9],
                 [11,8],[12,11],[13,12],[14,8],
                 [15,14],[16,15]
                ]
#(H36M 17)
#0: Pelvis 
#1: Right Hip
#2: Right Knee
#3: Right Ankle
#4: Left Hip
#5: Left Knee
#6: Left Ankle
#7: Spine
#8: Neck
#9: Nose
#10: Head (top)
#11: Left Shoulder
#12: Left Elbow 
#13: Left Wrist
#14: Right Shoulder
#15: Right Elbow
#16: Right Wrist

#Joints from the SMPL skeleton. layout='smpl'. 0-indexed
smpl_links = [[0,1],[1,2],[0,2],
              [1,4],[4,7],[7,10],[2,5],[5,8],[8,11],
              [16,18],[18,20],[17,19],[19,21]
              ] 

hand_links = [[0,1], [1,2], [2,3], [3,4],
             [0,5], [5,6], [6,7], [7,8],
             [0,9], [9,10], [10,11], [11,12],
             [0,13], [13,14],[14,15], [15,16],
             [0,17], [17,18], [18,19], [19,20]] 

#Joints from PyBullet simulator (from neural physcap)
pybullet_links = [[0,1], [1,10], [10,11], [11,12],
                  [2,6],
                  [1,13], [13,14], [14,15],
                  [10,2], [2,3], [3,4], [4,5],
                  [13,6], [6,7], [7,8], [8,9],
                  ]

#(PyBullet)
#0: Head (Neck)
#1: Neck (Chest)
#2: Left Hip
#3: Left Knee
#4: Left Heel
#5: Left Toe
#6: Right Hip
#7: Right Knee
#8: Right Heel
#9: Right Toe
#10: Left Shoulder
#11: Left Elbow
#12: Left Wrist
#13: Right Shoulder
#14: Right Elbow
#15: Right Wrist

kpt_marker_size = 20
line_thickness  = 2.0
quivl = None
quivr = None

#TODO: Combine update_video function between draw single pose and draw multipose

#Create video showing predicted pose along with GT and predicted forces
def draw_pose(cam, subject, movement, poses, grf=None, save_path=None, frame_width=808, frame_height=608, fps=24, layout='mscoco', joint=None, input_frame_dir=None, downsample_by=1, frame_offset=0):
    if layout == 'mocap':
        neighbor_links = mocap_links
        num_joints = 47
        idx_0 = 0
        la_idx = 4
        ra_idx = 25
        draw_base_of_support = False
    elif layout == 'h36m':
        neighbor_links = h36m_links
        num_joints = 32
        idx_0 = 1
        #Redundant to remove: 30-31, 27-28, 22-23, 19-20, 13-16-24, 0-11,
        #Extra joint to remove: 4,5,9,10,21,22,29,30
        draw_base_of_support = False
    elif layout == 'h36m_17':
        neighbor_links = h36m_17_links
        num_joints = 17
        idx_0 = 0
        draw_base_of_support = False
    elif layout == 'hand':
        neighbor_links = hand_links
        num_joints = 21
        idx_0 = 0
    elif layout == 'mscoco_foot':
        neighbor_links = mscoco_foot_links
        num_joints = 23
        idx_0 = 1
        la_idx = 19
        ra_idx = 22
        draw_base_of_support = False
    elif layout == 'smpl':
        neighbor_links = smpl_links
        num_joints = 24
        idx_0 = 0
        la_idx = 7
        la_idx = 8
        draw_base_of_support = False
    elif layout == 'pybullet':
        neighbor_links = pybullet_links
        num_joints = 16
        idx_0 = 0
        draw_base_of_support = False
    else:
        neighbor_links = mscoco_links
        num_joints = 17
        idx_0 = 1
        la_idx = 15
        ra_idx = 16
        draw_base_of_support = False

    fig = plt.figure(1, figsize=(16,14))
    try:
        plt.suptitle('{} - {} - {}'.format(cam.replace('_',' '), subject.replace('_',' '), movement.replace('_',' ')))
    except:
        plt.suptitle('{} - {} - {}'.format(cam, subject, movement))

    use_3d = True if poses.shape[-1]==3 else False
    lines  = []
    points = None
    points2 = None
    frm_count = None
    com_plot = None
    image = None
    base_of_supp = [] #Base-of-support. Convex hull of contact points (test with feet for now)

    joint_idxs = []

    if use_3d:
        ax = plt.subplot(111, projection='3d')

        if grf is not None:
            grf = np.copy(grf)
            #Swap Y-Z axes for visualization
            grf[:,[1,2,4,5]] = grf[:,[2,1,5,4]]
            global quivl, quivr
            lgrf = grf[0,3:]
            lgrf_norm = np.linalg.norm(lgrf) + np.finfo(float).eps
            rgrf = grf[0,:3]
            rgrf_norm = np.linalg.norm(rgrf) + np.finfo(float).eps

            quivl = ax.quiver(*poses[0,la_idx], *(lgrf/lgrf_norm), color='b', length=lgrf_norm)
            quivr = ax.quiver(*poses[0,ra_idx], *(rgrf/rgrf_norm), color='r', length=rgrf_norm)

            '''
            fig_grf = plt.figure(2)
            spec = fig_grf.add_gridspec(2,3)
            _axes = [fig_grf.add_subplot(spec[0,0]),
                     fig_grf.add_subplot(spec[0,1]),
                     fig_grf.add_subplot(spec[0,2]),
                     fig_grf.add_subplot(spec[1,0]),
                     fig_grf.add_subplot(spec[1,1]),
                     fig_grf.add_subplot(spec[1,2])]

            labels = ['Fx1', 'Fy1', 'Fz1', 'Fx2', 'Fy2', 'Fz2']
            colors = ['r', 'g', 'b', 'r--', 'g--', 'b--']
            for idx, lbl in enumerate(labels):
                _ax = _axes[idx]
                _ax.plot(np.arange(len(grf)), grf[:,idx], colors[idx], label=lbl) 
                _ax.set_ylim([-2.5,17])
                _ax.legend()
            '''

        #180-degree rotation of subject
        deg = 180
        rotate = R.from_euler('y', deg, degrees=True)
        poses  = np.array([rotate.apply(item) for item in poses]).astype(np.float32)

        #Mirror around vertical for visualization (left/right swapped?)
        poses[...,0] *= -1

        #Swap Y-Z axes for visualization
        poses[:,:,[1,2]] = poses[:,:,[2,1]]

        ax.set_xlim([-2000, 2000])
        ax.set_zlim([0, 2500])
        ax.set_ylim([-1600, 2000])

        #ax.set_xlim([-2, 2])
        #ax.set_zlim([-2, 2])
        #ax.set_ylim([-2, 2])

        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')

        #FOR VISUALIZATION
        #ax.set_xlim([-1000, 1000])
        #ax.set_zlim([0, 1500])
        #ax.set_ylim([-2000, 2000])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=10, azim=-130)

        #ax.view_init(elev=25, azim=None)

        ch  = 3
        for idx in range(len(neighbor_links)):
            lines.append(ax.plot([], [], [], color='black', linewidth=line_thickness)[0])

        if joint is not None:
            lines.append(ax.plot([], [], [], color='magenta')[0])
        else:
            points = ax.scatter(poses[0,:,0], poses[0,:,1], poses[0,:,2], s=kpt_marker_size, color='black')

            if layout == 'mocap':
                pelvis = (poses[:,2]+poses[:,23]+poses[:,18]+poses[:,39])/4
                neck   = (poses[:,0]+poses[:,44])/2
                added = np.stack((pelvis, neck))
                #points2 = ax.scatter(added[:,0,0], added[:,0,1], added[:,0,2], color='red')

                poses[:,2,2]  = pelvis[:,2]
                poses[:,23,2] = pelvis[:,2]

        if draw_base_of_support:
            floor_height = np.min(poses[0], axis=0)[2] - 5.0
            com = compute_CoM(poses[0])
            com_plot = ax.plot(com[0], com[1], floor_height, '*', c='r') #Plot center-of-mass but project it onto the ground

            convex_points = poses[0,15:,:2]
            convex_hull = ConvexHull(convex_points)
            for simplex in convex_hull.simplices:
                base_of_supp.append(ax.plot(convex_points[simplex, 0], convex_points[simplex, 1], [floor_height, floor_height], 'c', linewidth=2.0))
    else:
        if grf is None:
            ax = plt.subplot(111)
        else:
            ax = fig.add_subplot(spec[0,:2])

        #Swap Y-Z axes for visualization
        #poses[:,:,[0,1]] = poses[:,:,[1,0]]

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        ax.set_xlim([0,frame_width])
        ax.set_ylim([frame_height,0]) #invert y-axis, so origin at top-left
        #ax.axis('off')

        #For 2D viewpoints, overlay pose on video frames (if provided)
        if input_frame_dir is not None:
            stride = int(downsample_by)
            frame_paths = sorted(glob.glob(os.path.join(input_frame_dir,'*.png')))
            frame_paths = frame_paths[frame_offset::stride]
            background = plt.imread(frame_paths[0]) #frame_width x frame_height x 3
            image = ax.imshow(background)

        ch  = 2
        for idx in range(len(neighbor_links)):
            lines.append(ax.plot([], [], color='yellow', linewidth=line_thickness)[0])

        if joint is not None:
            lines.append(ax.plot([], [], [], color='magenta')[0])
        else:
            points = ax.scatter(poses[0,:,0], poses[0,:,1], s=kpt_marker_size, color='black')

        #frm_count = ax.text(50,50,0)

    def update_video(frame):
        #frm_count.set_text(str(frame))
        pose = poses[frame]

        if image is not None:
           background = plt.imread(frame_paths[frame])
           image.set_data(background)

        global quivl,quivr
        if quivl is not None:
            lgrf = grf[frame,3:]
            lgrf_norm = np.linalg.norm(lgrf) + np.finfo(float).eps
            rgrf = grf[frame,:3]
            rgrf_norm = np.linalg.norm(rgrf) + np.finfo(float).eps

            quivl.remove()
            quivr.remove()
            quivl = ax.quiver(*pose[la_idx],*(lgrf/lgrf_norm), color='b', length=lgrf_norm)
            quivr = ax.quiver(*pose[ra_idx],*(rgrf/rgrf_norm), color='r', length=rgrf_norm)

        if draw_base_of_support:
            #Remove previous lines
            for _ in range(len(base_of_supp)):
                base_of_supp.pop()[0].remove()
            convex_points = poses[frame,15:,:2]
            convex_hull = ConvexHull(convex_points)
            for idx, simplex in enumerate(convex_hull.simplices):
                base_of_supp.append(ax.plot(convex_points[simplex, 0], convex_points[simplex, 1], [floor_height, floor_height], 'c', linewidth=2.0))

        for idx, (i,j) in enumerate(neighbor_links):
            if -1 in [pose[i-idx_0,0], pose[i-idx_0,1], pose[j-idx_0,0], pose[j-idx_0,1]]:
                continue

            lines[idx].set_xdata(np.array([pose[i-idx_0,0],pose[j-idx_0,0]]))
            lines[idx].set_ydata(np.array([pose[i-idx_0,1],pose[j-idx_0,1]]))
            if use_3d:
                lines[idx].set_3d_properties(np.array([pose[i-idx_0,2],pose[j-idx_0,2]]), zdir='z')

        if joint is not None:
            lines[-1].set_xdata(poses[:frame,joint,0])
            lines[-1].set_ydata(poses[:frame,joint,1])
            if use_3d:
                lines[-1].set_3d_properties(poses[:frame,joint,2])
        else:
            if use_3d:
                points._offsets3d = (pose[:,0], pose[:,1], pose[:,2])
                points._sizes3d = np.ones(num_joints) * kpt_marker_size

                if draw_base_of_support:
                    com = compute_CoM(pose)
                    com_plot[0].set_data(com[0], com[1])
                    com_plot[0].set_3d_properties(floor_height)

                #if layout == 'mocap':
                #    points2._offsets3d = (added[:,frame,0], added[:,frame,1], added[:,frame,2])
                #    points2._sizes3d = np.ones(len(added)) * 15
            else:
                points.set_offsets(pose) #update keypoints position
                points.set_sizes(np.ones(num_joints) * kpt_marker_size) #update keypoints sizes

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, len(poses)), interval=1000/fps, repeat=False)

    if not save_path:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={})

        out_vid = os.path.join(save_path, cam+'_'+subject+'_'+movement+'.mp4')
        anim.save(out_vid)
        print('Saved video to: {}'.format(out_vid))
        
        plt.close()

#Create video of several poses for comparison (expecting list of 2 for now)
quivl1 = None
quivr1 = None
quivl2 = None
quivr2 = None
def draw_multi_pose(cam, subject, movement, all_poses, all_grfs=None, titles=None, save_path=None, fps=24, layout='mscoco'):
    if layout == 'mocap':
        neighbor_links = mocap_links
        num_joints = 47
        la_idx = 4
        ra_idx = 25
    elif layout =='h36m':
        neighbor_links = h36m_links
        num_joints = 32
        idx_0 = 0
        #Redundant to remove: 30-31, 27-28, 22-23, 19-20, 13-16-24, 0-11,
        #Extra joint to remove: 4,5,9,10,21,22,29,30
    elif layout == 'hand':
        neighbor_links = hand_links
        num_joints = 21
        idx_0 = 0
    elif layout == 'mscoco_foot':
        neighbor_links = mscoco_foot_links
        num_joints = 23
        idx_0 = 1
        la_idx = 19
        ra_idx = 22
    else:
        neighbor_links = mscoco_links
        num_joints = 17
        idx_0 = 1
        la_idx = 15
        ra_idx = 16

    fig = plt.figure(1, figsize=(16,14))
    try:
        plt.suptitle('{} - {} - {}'.format(cam.replace('_',' '), subject.replace('_',' '), movement.replace('_',' ')))
    except:
        plt.suptitle('{} - {} - {}'.format(cam, subject, movement))

    #use_3d = True if all_poses[0].shape[-1]==3 else False

    lines  = [[], [], []]
    points = [None, None, None]
    frm_count = None
    if all_grfs is not None:
        assert(len(all_poses) == len(all_grfs))
        global quivl1, quivr1, quivl2, quivr2

    person_axes = [plt.subplot(121, projection='3d'),
                   plt.subplot(122, projection='3d')]

    grfs = []
    for i, ax in enumerate(person_axes):
        if all_grfs is not None:
            grfs.append(np.copy(all_grfs[i]))

            #Swap Y-Z axes for visualization
            grfs[-1][:,[1,2,4,5]] = grfs[-1][:,[2,1,5,4]]

            lgrf = grfs[-1][0,3:]
            lgrf_norm = np.linalg.norm(lgrf) + np.finfo(float).eps
            rgrf = grfs[-1][0,:3]
            rgrf_norm = np.linalg.norm(rgrf) + np.finfo(float).eps

            if i == 0:
                quivl1 = ax.quiver(*all_poses[i][0,la_idx], *(lgrf/lgrf_norm), color='b', length=lgrf_norm)
                quivr1 = ax.quiver(*all_poses[i][0,ra_idx], *(rgrf/rgrf_norm), color='r', length=rgrf_norm)
            elif i == 1:
                quivl2 = ax.quiver(*all_poses[i][0,la_idx], *(lgrf/lgrf_norm), color='b', length=lgrf_norm)
                quivr2 = ax.quiver(*all_poses[i][0,ra_idx], *(rgrf/rgrf_norm), color='r', length=rgrf_norm)

        #180-degree rotation of subject
        deg = 180
        rotate = R.from_euler('y', deg, degrees=True)
        all_poses[i]  = np.array([rotate.apply(item) for item in all_poses[i]]).astype(np.float32)

        #Mirror around verticalfor visualization (left/right swapped?)
        all_poses[i][...,0] *= -1

        #Swap Y-Z axes for visualization
        all_poses[i][:,:,[1,2]] = all_poses[i][:,:,[2,1]]

        ax.set_xlim([-2000, 2000])
        ax.set_zlim([0, 2500])
        ax.set_ylim([-1600, 2000])

        #ax.set_xlim([-1, 1])
        #ax.set_zlim([-1, 1])
        #ax.set_ylim([-1, 1])

        #FOR VISUALIZATION
        #ax.set_xlim([-500, 500])
        #ax.set_zlim([0, 1800])
        #ax.set_ylim([0, 1800])

        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_zticklabels([])

        #ax.view_init(elev=0, azim=-90)
        ax.view_init(elev=25, azim=None)

        if titles is not None:
            ax.set_title(titles[i])

        if i==0:
            clr = 'black'
        else:
            clr = 'blue'
        ch  = 3
        for idx in range(len(neighbor_links)):
            lines[i].append(ax.plot([], [], [], color=clr, linewidth=line_thickness)[0])

        points[i] = ax.scatter(all_poses[i][0,:,0], all_poses[i][0,:,1], all_poses[i][0,:,2], color=clr)
    #frm_count = person_axes[0].text(-2000,0,1500,0)

    def update_video(frame):
        #frm_count.set_text(str(frame))
        for p_idx in range(2):
            pose = all_poses[p_idx][frame]
            points[p_idx]._offsets3d = (pose[:,0], pose[:,1], pose[:,2])
            points[p_idx]._sizes3d = np.ones(num_joints) * kpt_marker_size

            if p_idx == 0:
                global quivl1, quivr1
                if quivl1 is not None:
                    lgrf = grfs[p_idx][frame,3:]
                    lgrf_norm = np.linalg.norm(lgrf) + np.finfo(float).eps
                    rgrf = grfs[p_idx][frame,:3]
                    rgrf_norm = np.linalg.norm(rgrf) + np.finfo(float).eps

                    quivl1.remove()
                    quivr1.remove()
                    quivl1 = person_axes[p_idx].quiver(*pose[la_idx],*(lgrf/lgrf_norm), color='b', length=lgrf_norm)
                    quivr1 = person_axes[p_idx].quiver(*pose[ra_idx],*(rgrf/rgrf_norm), color='r', length=rgrf_norm)
            elif p_idx == 1:
                global quivl2, quivr2
                if quivl2 is not None:
                    lgrf = grfs[p_idx][frame,3:]
                    lgrf_norm = np.linalg.norm(lgrf) + np.finfo(float).eps
                    rgrf = grfs[p_idx][frame,:3]
                    rgrf_norm = np.linalg.norm(rgrf) + np.finfo(float).eps

                    quivl2.remove()
                    quivr2.remove()
                    quivl2 = person_axes[p_idx].quiver(*pose[la_idx],*(lgrf/lgrf_norm), color='b', length=lgrf_norm)
                    quivr2 = person_axes[p_idx].quiver(*pose[ra_idx],*(rgrf/rgrf_norm), color='r', length=rgrf_norm)

            for idx, (i,j) in enumerate(neighbor_links):
                if -1 in [pose[i-idx_0,0], pose[i-idx_0,1], pose[j-idx_0,0], pose[j-idx_0,1]]:
                    continue

                lines[p_idx][idx].set_xdata(np.array([pose[i-idx_0,0],pose[j-idx_0,0]]))
                lines[p_idx][idx].set_ydata(np.array([pose[i-idx_0,1],pose[j-idx_0,1]]))
                lines[p_idx][idx].set_3d_properties(np.array([pose[i-idx_0,2],pose[j-idx_0,2]]), zdir='z')
        #return ln

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, len(all_poses[0])), interval=1000/fps, repeat=False)

    if not save_path:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={})

        out_vid = os.path.join(save_path, cam+'_'+subject+'_'+movement+'.mp4')
        anim.save(out_vid)
        print('Saved video to: {}'.format(out_vid))
        plt.close()
        
if __name__ == "__main__":
    width  = 1000
    height = 1002

    #dataset = 'h36m_25fps'
    dataset = 'h36m'
    split   = 'val'
    samples = [
		['S9','Walking_1'],
            ]
    for subj, mvmt in samples:
        #json_file = './data/pybullet/h36m/val/S11_SittingDown_1.npy'
        json_file = './data/pybullet/'+dataset+'/'+split+'/'+subj+'_'+mvmt+'.npy'

        data = np.load(json_file, allow_pickle=True).item()

        kpts_3d = data['kpts_3d'][:400]
        #kpts_3d[...,0] *= -1
        kpts_mc = data['kpts_mc']
        grf     = data['grf']

        if grf is not None:
            total_frames = len(kpts_3d)
            time = grf['time']
            indices = np.linspace(0, len(time)-1, total_frames).astype(np.int32)

            fx1 = np.array(grf['ground_force1_vx'])[indices]
            fy1 = np.array(grf['ground_force1_vy'])[indices]
            fz1 = np.array(grf['ground_force1_vz'])[indices]
            fx2 = np.array(grf['ground_force2_vx'])[indices]
            fy2 = np.array(grf['ground_force2_vy'])[indices]
            fz2 = np.array(grf['ground_force2_vz'])[indices]
            grf_sampled = np.stack((fx1,fy1,fz1,fx2,fy2,fz2), axis=1)

        #deg = 180
        deg = 0
        rotate = R.from_euler('y', deg, degrees=True)
        kpts_3d = median_filter(np.array([rotate.apply(item) for item in kpts_3d]).astype(np.float32),15)

        from tools.keypoints import compute_segment_lens, constrain_segment_lens
        seg_lens = compute_segment_lens(kpts_3d, 'mscoco_foot')

        _kpts_3d = constrain_segment_lens(np.copy(kpts_3d), seg_lens, 'mscoco_foot')

        '''
        #Invert translation (fix horizontal positions to center)
        kpts_3d[:,:,0] -= kpts_3d[:,16,0,None]
        kpts_3d[:,:,2] -= kpts_3d[:,16,2,None]

        #Invert rotations (fix to original orientation)
        for i in range(1,len(kpts_3d)):
            est_rot, rssd = R.align_vectors(kpts_3d[0], kpts_3d[i])
            kpts_3d[i] = est_rot.apply(kpts_3d[i]).astype(np.float32)
        '''

        draw_pose('3d', subj, mvmt, kpts_3d, layout='mscoco_foot', fps=50, save_path='./runtime_visualization/forcepose_gt')
        #draw_pose('main_figure', subj, mvmt, kpts_3d, layout='mscoco_foot')
        #draw_multi_pose('test_3d', subj, mvmt, [kpts_3d, _kpts_3d], layout='mscoco_foot', fps=50)
