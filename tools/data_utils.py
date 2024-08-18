import os
import glob
import json
import numpy as np

from scipy.spatial.transform import Rotation as R

def get_pose_data(dataset_root, dataset, splits, **kwargs):
    pose_format = kwargs['pose_format']

    subj = kwargs['subject']
    mvmt = kwargs['movement']
    
    data = None
    for split in splits:
        filename = os.path.join(dataset_root,dataset,split,subj+'_'+mvmt+'.npy')
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=True).item()

    kpts_3d = data['kpts_3d']
    kpts_mc = data['kpts_mc']
    grf     = data['grf']
    keyFrameDuration = data.get('keyFrameDuration', 1./50)

    #Rotate for consistency with previous runs
    #NOTE: Ideally remove this or rotate to align with PyBullet +X
    deg = 180
    rotate = R.from_euler('y', deg, degrees=True)
    kpts_3d = np.array([rotate.apply(item) for item in kpts_3d]).astype(np.float32)

    if pose_format == 'mocap_47':
        pelvis   = (kpts_mc[0,2]+kpts_mc[0,23]+kpts_mc[0,18]+kpts_mc[0,39])/4
        kpts_mc -= pelvis
        kpts_mc /= 1000
        return {'kpts':kpts_mc, 'keyFrameDuration':keyFrameDuration}
    else:
        pelvis = (kpts_3d[0,12]+kpts_3d[0,11])/2 #(Root) Mid-point between hips
        kpts_3d -= pelvis
        kpts_3d /= 1000

        return {'kpts':kpts_3d, 'keyFrameDuration':keyFrameDuration}
