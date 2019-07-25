import numpy as np
# from mayavi import mlab
import time
# import sys
import torch.utils.data as data
# import glob
import os
import os.path
from data import utils,config
# import config
from data.box_overlaps import bbox_overlaps
import models.surfacenet
from torch.autograd import Variable
import torch
# from utils import KDAnchorCenter, KDAnchorCenterShow,load_kitti_label
# import pickle
# import PreProcess
# # import utils
# from utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
# import Data.PreProcess
# from Data import PreProcess

class KittiData(data.Dataset):

    def __init__(self, cfg, root='./KITTI', set='train',type='velodyne_train', TargetLoad = True, TargetWrite = True):
        super(KittiData, self).__init__()
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        self.label_path = os.path.join(self.data_path, "label_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.anchors = cfg.anchors.reshape(-1,7) # (w, h, 2, 7) -> ( (w x h x 2), 7)
        self.layer_div = cfg.div
        self.FrontView_W = cfg.FV_W
        self.FrontView_H = cfg.FV_H
        self.FrontView_shape = (int(self.FrontView_H / self.layer_div / 2), int(self.FrontView_W / 2))
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold
        self.anchors_per_position = cfg.anchors_per_position
        self.TargetLoad = TargetLoad
        self.TargetWrite = TargetWrite
        

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

    def CalculateTarget(self, gt_box3d_corner, lidar_center, point_cloud_xyz, anchor_num = 2):
        lidar_num = lidar_center.shape[0]
        

        pos_equal_one = np.zeros((*self.FrontView_shape, anchor_num))
        neg_equal_one = np.zeros((*self.FrontView_shape, anchor_num))
        targets = np.zeros((*self.FrontView_shape, anchor_num * 7)) # x,y,z,w,h,l,theta

        # gt_box3d_xyzhwlr = utils.hwlxyzr_to_xyzhwlr(gt_box3d_hwlxyzr)
        gt_xyzhwlr = utils.box3d_corner_to_center_batch(gt_box3d_corner)

        # anchors_corner = utils.anchors_center_to_corner(self.anchors)

        # 128*16->64*8
        lidar_center = utils.ChangeSizeLidarDiv2(lidar_center, self.FrontView_shape[1], self.FrontView_shape[0])
        # lidar_box_center: x,y,z,h,w,l,r
        lidar_box_center = utils.lidar_center_to_3dbox_center_surface(lidar_center)

        # d 
        self.anchors = lidar_box_center

        anchors_d = np.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2)

        anchors_corner = utils.anchors_center_to_corner(lidar_box_center)

        anchors_standup_2d = utils.corner_to_standup_box2d_batch(anchors_corner)

        gt_standup_2d = utils.corner_to_standup_box2d_batch(gt_box3d_corner)

        # print('8888888888888888888888',anchors_corner.shape,gt_box3d_corner.shape)        

        aa = np.array([])

        iou2d = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32), # anchors_standup_2d ： ((lidarNum * anchornum), 4)
            np.ascontiguousarray(gt_standup_2d).astype(np.float32), # gt_standup_2d ( GT ) ： (N, 4) 
        )
        # print(iou2d.shape)
        #iou2d = (lidarNum * anchornum, N)  N is GT_Num
        # print('iou {} {} {} {}'.format(type(iou2d), iou2d.dtype, iou2d.shape, iou2d))
        # print('111111111111111-1---1--1-{}'.format(iou.shape))

        # 3D IOU
        # iou = utils.caculate_3d_iou(iou2d, gt_xyzhwlr, lidar_box_center)  #3D IOU
        iou = iou2d
        # print('0000000000000000000')
        # print(lidar_box_center.shape,lidar_center.shape)
        # print(iou.shape)
        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0]) # [0, 1, ……， N-1]
        mask = iou.T[id_highest_gt, id_highest] > 0 #[N * 1] 
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask] 
        
        # print('iiiiiiiiiiiiiiiiiiiiiiiiiiiou',iou.shape)
        # print(self.anchors)
        # print(lidar_box_center.shape)
        # print(self.anchors.shape)
        # print(gt_xyzhwlr.shape)

        # find anchor iou > cfg.XXX_POS_IOU
        # print(iou.shape)
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold) # iou ((lidarNum * anchornum), N)
        # print('kkkkkkkkkkkk',id_pos,id_pos_gt)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold, axis=1) == iou.shape[1])[0]

        # id_pos = np.concatenate([id_pos, id_highest])
        # id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)  #去重复
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.FrontView_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # print(index_x)
        # print(index_y)
        # ATTENTION: index_z should be np.array

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]           # det x
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]           # det y
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]     # 暂时用一样的anchor
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.FrontView_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*self.FrontView_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        # print(targets)
        # np.set_printoptions(threshold=np.inf)
        # print(anchors_d)
        # np.set_printoptions(threshold=np.inf)
        anchors2D = anchors_corner
        Anchor_box = lidar_box_center
        return pos_equal_one, neg_equal_one, targets, lidar_box_center, anchors2D, Anchor_box

    # def PrintTest(self):
    #     # with open(os.path.join(self.label_dir, '%s.txt' % set)) as f:
    #     #     self.file_list = f.read().splitlines()
    #     # print(os.path.join(self.files_dirs[0], 'training'))
    #     print(len(self.file_list))
    #     return 0

    def __getitem__(self, i):
#        anno_files = sorted(glob.glob(os.path.join(self.anno_dirs[index], '*.bin')))
        # point_cloud = np.fromfile(self.files_dirs[index], dtype=np.float32).reshape(-1, 4)

        getitem_timestart = time.time()
        # self.file_list[i] = str('000009')
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'

        #load data
        calib = utils.load_kitti_calib(calib_file)
        Tr = calib['Tr_velo2cam']
        gt_box3d_corner = utils.load_kitti_label_corner(label_file, Tr)
        # print(gt_box3d_corner)
        # gt_xyzhwlr = utils.box3d_corner_to_center_batch(gt_box3d_corner)
        # gt_hwlxyzr = utils.load_kitti_label_center(label_file)       # h,w,l,x,y,z,ry
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        #cut lidar
        point_cloud_xyz = lidar[:, 0:3]
        index = np.where(point_cloud_xyz[:, 0] > 0) #and np.where(point_cloud_xyz[:, 2] > -3) and np.where(point_cloud_xyz[:, 2] < -1.5) and np.where(point_cloud_xyz[:, 2] < 30)
        point_cloud_xyz_crop = point_cloud_xyz[index[0], :]
        lidar = lidar[index[0], :]
        # point_cloud_xyz = point_cloud_xyz_crop
        # KDTree_center_xyz = utils.KDAnchorCenter(point_cloud_xyz, 1024)
        # print('KDTree_center_xyz:{}'.format(KDTree_center_xyz))

        #label encoding to pos&neg&regin map

        # start_time = time.time()

        #load or caculate Targets
        if self.TargetLoad:
        # if False:
            Frustum_Voxel, Frustum_Voxel_num = utils.readTarget(self.file_list[i], self.FrontView_W)

        else:
            t0 = time.time()
            Frustum_Voxel, Frustum_Voxel_num= utils.lidar_to_FrontView_Frustum_Voxel(lidar, layer_num = self.FrontView_W)
            #output h*w*pts*xyz
            Frustum_Voxel, Frustum_Voxel_num = utils.Frustum_Voxel_ADD(Frustum_Voxel, Frustum_Voxel_num, self.layer_div)
            # print('++++++++++++++++++++++++++++++++++{}'.format(Frustum_Voxel.shape)) 
            # print(Frustum_Voxel_num.shape)
            # print('Frustum_Voxel:{}'.format(Frustum_Voxel.shape))
            # if self.TargetWrite:
            if self.TargetWrite:
                utils.writeTarget(Frustum_Voxel, Frustum_Voxel_num, self.file_list[i], self.FrontView_W)
            
            # print('TargetLoad = ',self.TargetLoad)

        t1 = time.time()
        lidar_center = utils.Frustum_Voxel_PesuKeyPoint(Frustum_Voxel, Frustum_Voxel_num, layer_div = 1)  #layer_div为1！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        t2 = time.time()
        pos_equal_one, neg_equal_one, targets, anchors_corner, anchors2D, Anchor_box= self.CalculateTarget(gt_box3d_corner, lidar_center, point_cloud_xyz_crop)


        pos_equal_one = np.transpose(pos_equal_one, (1, 0, 2))
        neg_equal_one = np.transpose(neg_equal_one, (1, 0, 2))
        targets = np.transpose(targets, (1, 0, 2))

        return pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num, lidar_center, point_cloud_xyz_crop, gt_box3d_corner, Tr, anchors_corner, anchors2D, Anchor_box

    def __len__(self):
        return len(self.file_list)
    

if __name__== '__main__':
    
    dataset = KittiData(cfg = config.config, root = '/home/ct/KITTI', TargetLoad = False)
    # print(test.__getitem__(1)[1][1][0])
    # test.PrintTest()
    for i in range(dataset.__len__()):
        pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num, lidar_center, point_cloud_xyz = dataset.__getitem__(i)
        

        # print('time:{}'.format(end-start))
        # print('gt_box3d_center:{}'.format(gt_box3d_center))
        # print('gt_xyzhwlr{}'.format(gt_xyzhwlr))
        # print(utils.hwlxyzr_to_xyzhwlr(gt_hwlxyzr))
        # print(gt_xyzhwlr)
        # point_cloud_xyz = lidar[:, 0:3]

        # start = time.time()
        # utils.KDAnchorCenter(point_cloud_xyz, 8)
        # end = time.time()

        # print('time:{}'.format(end-start))
        # print(KDAnchorCenter(point_cloud_xyz, 6))
        if i>-1:
            break

