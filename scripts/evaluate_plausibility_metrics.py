#Evaluate other plausibility metrics to compare with PyBullet
#Ex: Footskate (FS%) and Ground Penetration

import os
import sys
import glob
DIR = os.getcwd()
sys.path.append(DIR)

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

from tools.utils import median_filter
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',      default='h36m_25fps', type=str, help='dataset name')
parser.add_argument('--data_splits',  default='val', type=str, help='Splits from the dataset')
parser.add_argument('--frame_offset', default=200, type=int, help='Offset first frame by this number')
parser.add_argument('--num_frames',   default=100, type=int, help='Total number of frames')

args,_ = parser.parse_known_args()

subj_mvmt = [
  'S9,Directions_1', 
  'S9,Discussion_1',
  'S9,Greeting_1', 
  'S9,Posing_1', 
  'S9,Purchases_1', 
  'S9,Photo_1', #Replaced from TakingPhoto 1
  'S9,Waiting_1', 
  'S9,WalkDog_1', 
  'S9,WalkTogether_1', 
  'S9,Walking_1', 
  'S11,Directions_1', 
  'S11,Discussion_1', 
  'S11,Greeting_2', #Replaced from Greeting 1
  'S11,Posing_1', 
  'S11,Purchases_1', 
  'S11,Photo_1',  #Replaced from TakingPhoto 1
  'S11,Waiting_1', 
  'S11,WalkDog_1', 
  'S11,WalkTogether_1', 
  'S11,Walking_1', 
 ]

def main(args):
    dataset      = args.dataset
    splits       = [item for item in args.data_splits.split(',')]

    frame_offset = args.frame_offset
    num_frames   = args.num_frames

    for split in splits:
        print(dataset+':'+split)
        all_kpts = {}
        all_fs   = []
        all_gp   = []
        all_mpjpe = []
        all_mpjpe_g = []
        for item in subj_mvmt:
            subj,mvmt = item.split(',')
            #print('{} - {}'.format(subj, mvmt))

            #Grab ground truth 3D keypoint markers
            data_path    = os.path.join('/z/home/natlouis/PoseFormer/saved_outputs/world_space','gt_cpn_ft_'+subj+'_'+mvmt+'_cam3.npy')
            gt_kpts_3d = np.load(data_path)
            gt_kpts_3d[:,:,[1,2]] = gt_kpts_3d[:,:,[2,1]]
            gt_kpts_3d[...,0] *= -1

            #Downsample to 25 fps
            gt_kpts_3d = gt_kpts_3d[::2]
            gt_kpts_3d = gt_kpts_3d[frame_offset:frame_offset+num_frames]

            ground_height = estimate_ground_height(gt_kpts_3d, foot_idxs=[3,6])
            lfoot_contact, rfoot_contact = estimate_ground_contacts(gt_kpts_3d, ground_height, lfoot_idxs=[6], rfoot_idxs=[3])

            '''
            plt.figure()
            plt.subplot(2,1,1)
            plt.title(subj+','+mvmt+': contact')
            plt.plot(lfoot_contact, label='gt left')
            plt.subplot(2,1,2)
            plt.plot(rfoot_contact, label='gt right')
            plt.legend()
            plt.tight_layout()
            '''

            if dataset == 'poseformer':
                dataset_root = '/z/home/natlouis/PoseFormer/saved_outputs/world_space'
                data_path    = os.path.join(dataset_root,'cpn_ft_'+subj+'_'+mvmt+'_cam3.npy')

                kpts_3d = median_filter(np.load(data_path), 15)
                kpts_3d[:,:,[1,2]] = kpts_3d[:,:,[2,1]]
                kpts_3d[...,0] *= -1

                #Downsample to 25 fps
                kpts_3d = kpts_3d[::2]
                kpts_3d = kpts_3d[frame_offset:frame_offset+num_frames]

                kpts_3d /= 1000

                ground_height = estimate_ground_height(kpts_3d, foot_idxs=[3,6])
                #lfoot_contact, rfoot_contact = estimate_ground_contacts(kpts_3d, ground_height, lfoot_idxs=[6], rfoot_idxs=[3])
                fs = compute_fs(kpts_3d, lfoot_contact, rfoot_contact, lfoot_idx=6, rfoot_idx=3)
                gp = compute_gp(kpts_3d, ground_height)
                mpjpe_g = compute_mpjpe_g(kpts_3d, gt_kpts_3d)
                mpjpe = compute_mpjpe(kpts_3d, gt_kpts_3d)

            elif dataset == 'gt_poseformer':
                dataset_root = '/z/home/natlouis/PoseFormer/saved_outputs/world_space'
                data_path    = os.path.join(dataset_root,'gt_cpn_ft_'+subj+'_'+mvmt+'_cam3.npy')

                kpts_3d = np.load(data_path)
                kpts_3d[:,:,[1,2]] = kpts_3d[:,:,[2,1]]
                kpts_3d[...,0] *= -1

                #Downsample to 25 fps
                kpts_3d = kpts_3d[::2]
                kpts_3d = kpts_3d[frame_offset:frame_offset+num_frames]

                ground_height = estimate_ground_height(kpts_3d, foot_idxs=[3,6])
                #lfoot_contact, rfoot_contact = estimate_ground_contacts(kpts_3d, ground_height, lfoot_idxs=[6], rfoot_idxs=[3])
                fs = compute_fs(kpts_3d, lfoot_contact, rfoot_contact, lfoot_idx=6, rfoot_idx=3)
                gp = compute_gp(kpts_3d, ground_height)
                mpjpe_g = compute_mpjpe_g(kpts_3d, gt_kpts_3d)
                mpjpe = compute_mpjpe(kpts_3d, gt_kpts_3d)

            elif dataset == 'neural_physcap':
                dataset_root = '/z/home/natlouis/Neural_Physcap_Demo/results/phys_results/h36m'
                data_path    = os.path.join(dataset_root,subj+'_'+mvmt+'.60457274','p_3D_dyn_world.npy')

                kpts_3d = median_filter(np.load(data_path), 15)
                kpts_3d = np.reshape(kpts_3d, (kpts_3d.shape[0],-1,3))
                kpts_3d[:,:,[1,2]] = kpts_3d[:,:,[2,1]]
                kpts_3d[...,0] *= -1

                t_offset=int(10/2) #The authors use a temporal window of length 10, so the first 10 frames are discarded. (Divide 2 for downsampling)

                #Downsample to 25 fps
                kpts_3d = kpts_3d[::2]
                kpts_3d = kpts_3d[frame_offset-t_offset:(frame_offset-t_offset)+num_frames]

                kpts_3d /= 1000

                ground_height = estimate_ground_height(kpts_3d, foot_idxs=[5,9])
                #lfoot_contact, rfoot_contact = estimate_ground_contacts(kpts_3d, ground_height, lfoot_idxs=[4], rfoot_idxs=[8])
                fs = compute_fs(kpts_3d, lfoot_contact, rfoot_contact, lfoot_idx=[4,5], rfoot_idx=[8,9])
                gp = compute_gp(kpts_3d, ground_height)
                mpjpe_g = compute_mpjpe_g(kpts_3d, gt_kpts_3d, 'physcap')
                mpjpe = compute_mpjpe(kpts_3d, gt_kpts_3d, 'physcap')

            elif dataset == 'mscoco': #No heel and toe keypoints
                dataset_root = '/z/home/natlouis/video_grf_pred/data/pybullet/'
                pseudo_gt_dataset = 'h36m_25fps'
                pseudo_gt = load_pseudo_gt(dataset_root, pseudo_gt_dataset, split, subj, mvmt, frame_offset, num_frames)

                kpts_3d = np.copy(pseudo_gt)

                ground_height = estimate_ground_height(kpts_3d, foot_idxs=[15,16])

                #lfoot_contact, rfoot_contact = estimate_ground_contacts(kpts_3d, ground_height, lfoot_idxs=[15], rfoot_idxs=[16])
                fs = compute_fs(kpts_3d, lfoot_contact, rfoot_contact, lfoot_idx=15, rfoot_idx=16)
                gp = compute_gp(kpts_3d[:,:16], ground_height) #Down to ankles only 
                mpjpe_g = compute_mpjpe_g(kpts_3d, gt_kpts_3d, 'mscoco')
                mpjpe = compute_mpjpe(kpts_3d, gt_kpts_3d, 'mscoco')

            else:
                dataset_root = '/z/home/natlouis/video_grf_pred/data/pybullet/'
                pseudo_gt_dataset = 'h36m_25fps'
                pseudo_gt = load_pseudo_gt(dataset_root, pseudo_gt_dataset, split, subj, mvmt, frame_offset, num_frames)

                kpts_3d = np.copy(pseudo_gt)

                ground_height = estimate_ground_height(kpts_3d)
                #lfoot_contact, rfoot_contact = estimate_ground_contacts(kpts_3d, ground_height)
                fs = compute_fs(kpts_3d, lfoot_contact, rfoot_contact, lfoot_idx=[17,18,19], rfoot_idx=[20,21,22])
                gp = compute_gp(kpts_3d, ground_height) #Down to ankles only 
                mpjpe_g = compute_mpjpe_g(kpts_3d, gt_kpts_3d, 'mscoco_foot')
                mpjpe = compute_mpjpe(kpts_3d, gt_kpts_3d, 'mscoco_foot')

                '''
                plt.subplot(2,1,1)
                plt.plot(_lfoot_contact, label='pred left')
                plt.legend()
                plt.subplot(2,1,2)
                plt.plot(_rfoot_contact, label='pred right')
                plt.legend()
                plt.tight_layout()
                '''
                
            plt.show()
            print('{}: FS: {:.2f} GP: {:.2f}mm, mpjpe-G: {:.2f}mm, mpjpe: {:.2f}mm'.format(item, fs, gp*1000, mpjpe_g*1000, mpjpe*1000))
            all_fs.append(fs)
            all_gp.append(gp)
            all_mpjpe_g.append(mpjpe_g)
            all_mpjpe.append(mpjpe)
            all_kpts[item] = kpts_3d
        print('Footskate: {:.2f}%, GP: {:.2f}mm, mpjpe-G: {:.2f}mm, mpjpe: {:.2f}mm'.format(np.mean(all_fs), np.mean(all_gp)*1000, np.mean(all_mpjpe_g)*1000, np.mean(all_mpjpe)*1000))

def load_pseudo_gt(dataset_root, dataset, split, subj, mvmt, frame_offset=0, num_frames=-1):
    data_path    = os.path.join(dataset_root,dataset,split,subj+'_'+mvmt+'.npy')
    data = np.load(data_path, allow_pickle=True).item()
    
    kpts_3d = median_filter(data['kpts_3d'], 15)

    #Rotate for consistency with previous runs
    deg = 180
    rotate = R.from_euler('y', deg, degrees=True)
    kpts_3d = np.array([rotate.apply(item) for item in kpts_3d]).astype(np.float32)

    kpts_3d /= 1000
    kpts_3d = kpts_3d[frame_offset:frame_offset+num_frames]

    return kpts_3d

def estimate_ground_height(kpts_3d, foot_idxs=[17,18,19,20,21,22]):
    #Assuming horizontal ground plane
    #Assuming mscoco_foot format
    k = int(len(kpts_3d)*0.05) #% of frames

    _kpts = np.copy(kpts_3d[:,foot_idxs]).reshape(-1,3)
    _kpts = _kpts[_kpts[:,1].argsort()][:k]

    ground_height = np.mean(_kpts, axis=0)[1] #Average of lowest k points

    return ground_height

def estimate_ground_contacts(kpts_3d, ground_height, lfoot_idxs=[17,18,19], rfoot_idxs=[20,21,22]):
    lfoot_contact = np.ones(len(kpts_3d))
    rfoot_contact = np.ones(len(kpts_3d))

    #Simple contact estimation baseline (Borrowed from Rempe etal.)
    #Not in contact if:
    # 1) Heels and Toes height > xcm
    height_thresh = 0.05
    #height_thresh = 0.10
    lfoot_contact[np.min(kpts_3d[:,lfoot_idxs,1],1) > (ground_height + height_thresh)] = 0
    rfoot_contact[np.min(kpts_3d[:,rfoot_idxs,1],1) > (ground_height + height_thresh)] = 0

    # 2) Heels or Toes distance > xcm from previous step
    #diff = kpts_3d[1:,15:17] - kpts_3d[:-1,15:17]
    #dist = np.linalg.norm(diff,axis=-1)

    diff = kpts_3d[1:,lfoot_idxs+rfoot_idxs] - kpts_3d[:-1,lfoot_idxs+rfoot_idxs]
    dist = np.linalg.norm(diff,axis=-1)
    #move_thresh = 0.002
    #l_move = np.concatenate(([False],np.logical_or(dist[:,0]>move_thresh, dist[:,1]>move_thresh, dist[:,2]>move_thresh)))
    #r_move = np.concatenate(([False],np.logical_or(dist[:,3]>move_thresh, dist[:,4]>move_thresh, dist[:,5]>move_thresh)))
    move_thresh = 0.020
    l_move = np.concatenate(([False],dist[:,0]>move_thresh))
    r_move = np.concatenate(([False],dist[:,1]>move_thresh))

    lfoot_contact[l_move] = 0
    rfoot_contact[r_move] = 0

    return lfoot_contact, rfoot_contact

#Footskate
def compute_fs(kpts_3d, gt_lcontact, gt_rcontact, lfoot_idx=15, rfoot_idx=16):
    lankle = kpts_3d[:,lfoot_idx]
    rankle = kpts_3d[:,rfoot_idx]

    l_dist = np.linalg.norm(lankle[1:] - lankle[:-1],axis=-1)
    r_dist = np.linalg.norm(rankle[1:] - rankle[:-1],axis=-1)

    move_thresh = 0.020

    if l_dist.ndim > 1:
        l_move = np.concatenate(([False],np.any(l_dist>move_thresh, axis=-1)))
    else:
        l_move = np.concatenate(([False],l_dist>move_thresh))
    if r_dist.ndim > 1:
        r_move = np.concatenate(([False],np.any(r_dist>move_thresh, axis=-1)))
    else:
        r_move = np.concatenate(([False],r_dist>move_thresh))

    #Ankle moves more than move_thresh AND there is ground truth foot contact
    l_fs = np.logical_and(l_move, gt_lcontact)
    r_fs = np.logical_and(r_move, gt_rcontact)

    '''
    plt.subplot(2,1,1)
    plt.plot(gt_lcontact, label='left contact')
    plt.plot(l_move, label='left moving')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(gt_rcontact, label='right contact')
    plt.plot(r_move, label='right moving')
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''

    #footskate percentage
    fs = np.mean(np.logical_or(l_fs, r_fs))*100.

    return fs

#Ground penetration
def compute_gp(kpts_3d, ground_height):

    #Grab all joints below ground plane
    joint_y_vals = np.copy(kpts_3d)[:,:,1]
    below_ground = np.clip(joint_y_vals - ground_height,-10000,0)

    #Average distance to ground for joints below ground
    idxs  = np.nonzero(below_ground)
    if len(idxs[0]) > 1:
        gp = np.abs(np.mean(below_ground[idxs]))
    else:
        gp = 0

    return gp

from tools.visualization import draw_pose

def compute_mpjpe(predicted, target, pose_format='h36m_17'):
    """ 
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape[0] == target.shape[0]

    #Translate all pelvis to center
    target    -= target[:,None,0]
    if pose_format == 'h36m_17':
        predicted -= predicted[:,None,0]
        return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))
    elif pose_format == 'physcap':
        new_predicted = np.zeros_like(target)

        head_top = predicted[:,0]
        neck     = predicted[:,1]
        lshould = predicted[:,10]
        rshould = predicted[:,13]
        lelbow  = predicted[:,11]
        relbow  = predicted[:,14]
        lwrist  = predicted[:,12]
        rwrist  = predicted[:,15]
        lhip   = predicted[:,2]
        rhip   = predicted[:,6]
        lknee  = predicted[:,3]
        rknee  = predicted[:,7]
        lankle = predicted[:,4]
        rankle = predicted[:,8]

        pelvis   = (predicted[:,2] + predicted[:,6])/2
        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?

        new_predicted[:,0] = pelvis
        new_predicted[:,1] = rhip   
        new_predicted[:,2] = rknee  
        new_predicted[:,3] = rankle 
        new_predicted[:,4] = lhip   
        new_predicted[:,5] = lknee  
        new_predicted[:,6] = lankle 
        new_predicted[:,7] = thorax
        new_predicted[:,8] = neck     
        #new_predicted[:,9] = nose     
        new_predicted[:,10] = head_top
        new_predicted[:,11] = lshould 
        new_predicted[:,12] = lelbow  
        new_predicted[:,13] = lwrist  
        new_predicted[:,14] = rshould 
        new_predicted[:,15] = relbow  
        new_predicted[:,16] = rwrist  

        new_predicted -= new_predicted[:,None,0]

        #Zero-out nose
        target[:,9] *= 0
        new_predicted[:,9] *= 0

        return np.mean(np.linalg.norm(new_predicted - target, axis=len(target.shape)-1))

    elif pose_format in ['mscoco', 'mscoco_foot']:
        new_predicted = np.zeros_like(target)

        nose = predicted[:,0]
        leye = predicted[:,1]
        reye = predicted[:,2]
        lear = predicted[:,3]
        rear = predicted[:,4]
        lshould = predicted[:,5]
        rshould = predicted[:,6]
        lelbow  = predicted[:,7]
        relbow  = predicted[:,8]
        lwrist  = predicted[:,9]
        rwrist  = predicted[:,10]
        lhip   = predicted[:,11]
        rhip   = predicted[:,12]
        lknee  = predicted[:,13]
        rknee  = predicted[:,14]
        lankle = predicted[:,15]
        rankle = predicted[:,16]

        pelvis = (predicted[:,12]+predicted[:,11])/2 #(Root) Mid-point between hips
        neck   = (predicted[:,5]+predicted[:,6])/2 #Mid-point between shoulders
        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck.

        new_predicted[:,0] = pelvis
        new_predicted[:,1] = rhip   
        new_predicted[:,2] = rknee  
        new_predicted[:,3] = rankle 
        new_predicted[:,4] = lhip   
        new_predicted[:,5] = lknee  
        new_predicted[:,6] = lankle 
        new_predicted[:,7] = thorax
        new_predicted[:,8] = neck     
        new_predicted[:,9] = nose     
        #new_predicted[:,10] = 0
        new_predicted[:,11] = lshould 
        new_predicted[:,12] = lelbow  
        new_predicted[:,13] = lwrist  
        new_predicted[:,14] = rshould 
        new_predicted[:,15] = relbow  
        new_predicted[:,16] = rwrist  

        new_predicted -= new_predicted[:,None,0]

        #Zero-out head_top
        target[:,10] *= 0
        new_predicted[:,10] *=0

        return np.mean(np.linalg.norm(new_predicted - target, axis=len(target.shape)-1))
    else:
        return 0

def compute_mpjpe_g(predicted, target, pose_format='h36m_17'):
    """ 
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape[0] == target.shape[0]

    if pose_format == 'h36m_17':
        return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))
    elif pose_format == 'physcap':
        new_predicted = np.zeros_like(target)

        head_top = predicted[:,0]
        neck     = predicted[:,1]
        lshould = predicted[:,10]
        rshould = predicted[:,13]
        lelbow  = predicted[:,11]
        relbow  = predicted[:,14]
        lwrist  = predicted[:,12]
        rwrist  = predicted[:,15]
        lhip   = predicted[:,2]
        rhip   = predicted[:,6]
        lknee  = predicted[:,3]
        rknee  = predicted[:,7]
        lankle = predicted[:,4]
        rankle = predicted[:,8]

        pelvis   = (predicted[:,2] + predicted[:,6])/2
        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck. But maybe 1/3 from neck?

        new_predicted[:,0] = pelvis
        new_predicted[:,1] = rhip   
        new_predicted[:,2] = rknee  
        new_predicted[:,3] = rankle 
        new_predicted[:,4] = lhip   
        new_predicted[:,5] = lknee  
        new_predicted[:,6] = lankle 
        new_predicted[:,7] = thorax
        new_predicted[:,8] = neck     
        #new_predicted[:,9] = nose     
        new_predicted[:,10] = head_top
        new_predicted[:,11] = lshould 
        new_predicted[:,12] = lelbow  
        new_predicted[:,13] = lwrist  
        new_predicted[:,14] = rshould 
        new_predicted[:,15] = relbow  
        new_predicted[:,16] = rwrist  

        #Zero-out nose
        target[:,9] *= 0

        return np.mean(np.linalg.norm(new_predicted - target, axis=len(target.shape)-1))

    elif pose_format in ['mscoco', 'mscoco_foot']:
        new_predicted = np.zeros_like(target)

        nose = predicted[:,0]
        leye = predicted[:,1]
        reye = predicted[:,2]
        lear = predicted[:,3]
        rear = predicted[:,4]
        lshould = predicted[:,5]
        rshould = predicted[:,6]
        lelbow  = predicted[:,7]
        relbow  = predicted[:,8]
        lwrist  = predicted[:,9]
        rwrist  = predicted[:,10]
        lhip   = predicted[:,11]
        rhip   = predicted[:,12]
        lknee  = predicted[:,13]
        rknee  = predicted[:,14]
        lankle = predicted[:,15]
        rankle = predicted[:,16]

        pelvis = (predicted[:,12]+predicted[:,11])/2 #(Root) Mid-point between hips
        neck   = (predicted[:,5]+predicted[:,6])/2 #Mid-point between shoulders
        thorax = (pelvis+neck)/2       #Estimating between pelvis and neck.

        new_predicted[:,0] = pelvis
        new_predicted[:,1] = rhip   
        new_predicted[:,2] = rknee  
        new_predicted[:,3] = rankle 
        new_predicted[:,4] = lhip   
        new_predicted[:,5] = lknee  
        new_predicted[:,6] = lankle 
        new_predicted[:,7] = thorax
        new_predicted[:,8] = neck     
        new_predicted[:,9] = nose     
        #new_predicted[:,10] = 0
        new_predicted[:,11] = lshould 
        new_predicted[:,12] = lelbow  
        new_predicted[:,13] = lwrist  
        new_predicted[:,14] = rshould 
        new_predicted[:,15] = relbow  
        new_predicted[:,16] = rwrist  

        #Zero-out head_top
        target[:,10] *= 0

        return np.mean(np.linalg.norm(new_predicted - target, axis=len(target.shape)-1))
    else:
        return 0

if __name__ == "__main__":
    main(args)
