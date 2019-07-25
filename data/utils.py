from numpy import *
import numpy as np
from mayavi import mlab
import time
import cv2
import sys
import matplotlib.pyplot as plt
import kdtree
from data.config import config as cfg
import torch
import scipy.io
import json
from ._ext import nms
# NodeList = []
# ChilList = []

def Show(PonitCloudrData, anchorcenter = []):
    x = PonitCloudrData[:, 0]
    y = PonitCloudrData[:, 1]
    z = PonitCloudrData[:, 2]
    r = PonitCloudrData[:, 3]
    # mlab.points3d(x, y, z, r, scale_factor=.05, scale_mode='vector')
    # # r = np.ones(z.shape)

    xc = anchorcenter[:, 0]
    yc = anchorcenter[:, 1]
    zc = anchorcenter[:, 2]
    # mlab.points3d(xc, yc, zc, scale_factor=.15, scale_mode='vector')
    # mlab.show()


def LinemedianBlur(img):
    simg = img
    for i in range(img.shape[0]):
        for j in range(1, img.shape[1]-1):
            for k in range(3):
                # if(simg[i,j] == 0 and simg[i, j-1]>0 and simg[i, j+1]>0):
                #     img[i,j] = int((img[i,j-1]+img[i,j+1])/2)
                img[i, j] = int(simg[i, j - 1] + simg[i, j + 1] + simg[i, j]-min(simg[i, j - 1] , simg[i, j + 1] , simg[i, j])-max(simg[i, j - 1] , simg[i, j + 1] , simg[i, j]))
    return img


def preTraverse(root):
	if root.height() == 0:
		return
	print(root.data)
	preTraverse(root.left)
	preTraverse(root.right)


def findKDtreeNode(root, height):
	if root.height() == 0:
		return 
	if root.height() == height:
		NodeList.append(root)
	findKDtreeNode(root.left, height)
	findKDtreeNode(root.right, height)


def findChilData(root):
	if root.height() == 0:
		return 
	ChilList.append(root.data)
	findChilData(root.left)
	findChilData(root.right)


def KDAnchorCenterShow(point_cloud_xyz, AchorCenterNum = 1024):
    ptlist = []
    global ChilList, NodeList
    NodeList = []
    ChilList = []
    ChilListALL = []
    NChilArrALL = []

    for pt in point_cloud_xyz:
        ptlist.append(pt)
    tree = kdtree.create(ptlist, dimensions=3)
    KDtreeNodeHeight = int(np.log2(2 ** tree.height() / AchorCenterNum)) 
    findKDtreeNode(tree, KDtreeNodeHeight)  # output NodeList      height=9,NodeList_Num = 256   height=8,NodeList_Num = 512  
    for Node in NodeList:
        findChilData(Node) # output ChilList
        ChilListALL.append(np.array(ChilList))
        ChilList = []
    ALL = np.zeros((point_cloud_xyz.shape[0], 4))
    for i,cl in enumerate(ChilListALL):
        length = cl.shape[0]
        ALL[i*length:(i+1)*length, :3] = cl
        ALL[i*length:(i+1)*length, 3] = i % 10
    print(len(NodeList))
    AchorCenter = np.zeros((len(NodeList),3))
    print(AchorCenter.shape)
    for i,Node in enumerate(NodeList):
        AchorCenter[i,:] = Node.data
    Show(ALL,AchorCenter)


def KDAnchorCenter(point_cloud_xyz, AchorCenterNum = 1024):
    global NodeList
    NodeList = []
    ptlist = []
    NChilArrALL = []

    start1 = time.time()
    for pt in point_cloud_xyz:
        ptlist.append(pt)
    tree = kdtree.create(ptlist, dimensions=3)
    start2 = time.time()
    # print(tree.height())
    KDtreeNodeHeight = int(np.log2(2 ** tree.height() / AchorCenterNum)) 
    print('kdheight:{}'.format(KDtreeNodeHeight))
    findKDtreeNode(tree, KDtreeNodeHeight)
    # if tree.height() == 17:    #2^16 = 65536~13172 = 2^17
    #     findKDtreeNode(tree, KDtreeNodeHeight)  # output NodeList      height=9,NodeList_Num = 256   height=8,NodeList_Num = 512  
    # else:
    #     findKDtreeNode(tree, KDtreeNodeHeight+1)
    end2 = time.time()
    print('timw1:{}'.format(start2-start1))
    print('timw2:{}'.format(end2-start2))
    AchorCenter = np.zeros((len(NodeList),3))
    for i,Node in enumerate(NodeList):
        AchorCenter[i,:] = Node.data
    # AchorCenterNum = AchorCenter.shape[0]
    # print('KDTree center num: {}'.format(AchorCenterNum))
    return AchorCenter

def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


# def load_kitti_label_center(label_file):
#     # h,w,l,x,y,z,ry
#     with open(label_file,'r') as f:
#         lines = f.readlines()

#     gt_boxes3d_center = []

#     num_obj = len(lines)

#     for j in range(num_obj):
#         obj = lines[j].strip().split(' ')

#         obj_class = obj[0].strip()
#         if obj_class not in cfg.class_list:
#             continue

#         # box3d_corner = box3d_centrt_to_corner(obj[8:])
#         box3d_center = [float(i) for i in obj[8:]]
#         # print(box3d_center)

#         gt_boxes3d_center.append(box3d_center)

#     gt_boxes3d_center = np.array(gt_boxes3d_center).reshape(-1,7)
#     # print('seesee:{}'.format(gt_boxes3d_center))
#     return gt_boxes3d_center


def load_kitti_label_corner(label_file, Tr):

    with open(label_file,'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    num_obj = len(lines)

    for j in range(num_obj):
        obj = lines[j].strip().split(' ')

        obj_class = obj[0].strip()
        if obj_class not in cfg.class_list:
            continue
        # box3d_corner = center_to_corner_box3d(np.array(obj[8:]), Tr)
        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)
        # print('ssssssssseeeeeeeeeeeeeeeeeeeeeeeeeeeee',box3d_corner)
        gt_boxes3d_corner.append(box3d_corner)
        
    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1,8,3)

    return gt_boxes3d_corner


def KDTree_center_to_3dbox_center(KDTree_center_xyz, anchornum = 2):
    #input KDTree x y z                             (KDTreeCenterNum, 3)      offten KDTreeCenterNum = 1024
    #output 3d anchor for anchor center(from KDTree) (KDTreeCenterNum * anchornum, 7)
    print('KDTree_center_xyz: {}'.format(KDTree_center_xyz.shape))
    x = KDTree_center_xyz[:, 0]
    y = KDTree_center_xyz[:, 1]
    z = KDTree_center_xyz[:, 2]

    cx = np.tile(x[..., np.newaxis], anchornum)
    cy = np.tile(y[..., np.newaxis], anchornum)
    cz = np.tile(z[..., np.newaxis], anchornum)
    w = np.ones_like(cx) * 1.6
    l = np.ones_like(cx) * 3.9
    h = np.ones_like(cx) * 1.56
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi/2

    anchor = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
    ancoor7 = anchor.reshape(-1,7)

    # print('anchors:{}'.format(anchors.shape))
    # print(anchors[0, 0, 0:7])
    # print(anchors[0, 1, 0:7])
    # print(anchors[1, 0, 0:7])
    # print(anchors[10, 0, 0:7])
    # print(anchors[11, 0, 0:7])
    # print('++++++')
    return ancoor7

def lidar_center_to_3dbox_center(lidar_center_xyz, anchornum = 2):
    #input KDTree x y z                             (KDTreeCenterNum, 3)      offten KDTreeCenterNum = 1024
    #output 3d anchor for anchor center(from KDTree) (KDTreeCenterNum * anchornum, 7)
    # print('lidar_center_xyz: {}'.format(lidar_center_xyz.shape))
    x = lidar_center_xyz[..., 0]
    y = lidar_center_xyz[..., 1]
    z = lidar_center_xyz[..., 2]

    cx = np.tile(x[..., np.newaxis], anchornum)
    cy = np.tile(y[..., np.newaxis], anchornum)
    cz = np.tile(z[..., np.newaxis], anchornum)
    w = np.ones_like(cx) * 1.6
    l = np.ones_like(cx) * 3.9
    h = np.ones_like(cx) * 1.56
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi/2

    anchor = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
    ancoor7 = anchor.reshape(-1,7)
    # print('anchor shape: {}'.format(ancoor7.shape))
    # print('cccccccccccccccccccccccccccccccccccccome')
    return ancoor7

def lidar_center_to_3dbox_center_surface(lidar_center_xyz, anchornum = 2, mode = 'train'):
    w_len = 1.6
    l_len = 3.9
    h_len = 1.56
    x = lidar_center_xyz[..., 0]
    y = lidar_center_xyz[..., 1]
    #地面大概-１.５
    z = np.ones_like(x) * (-1.5) + h_len / 2
    # z = lidar_center_xyz[..., 2]
    if mode != 'train':
        print('lidar_center_xyz{}'.format(lidar_center_xyz.shape))

    # #中心变换
    # length = l_len / 2
    # k = y / x
    # x = x + sign(x) * length / (1 + k**2)**0.5
    # y = k * x


    cx = np.tile(x[..., np.newaxis], anchornum)
    cy = np.tile(y[..., np.newaxis], anchornum)
    cz = np.tile(z[..., np.newaxis], anchornum)
    w = np.ones_like(cx) * w_len
    l = np.ones_like(cx) * l_len
    h = np.ones_like(cx) * h_len
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi/2

    anchor = np.stack([cx, cy, cz, h, w, l, r], axis=-1)
    # print('=-=-=-=-=-=-=-=-=-=-=-{}'.format(cx.shape))
    
    # print('-------------{}'.format(anchor.shape))
    # print('anchor shape: {}'.format(ancoor7.shape))
    # print('kkkkkkkkkkkkkkkkkkkkk{}'.format(anchor.shape))
    if mode == 'train':
        anchor = np.transpose(anchor, (1, 0, 2, 3))
        ancoor7 = anchor.reshape(-1,7)
        return ancoor7
    else:
        anchor = np.transpose(anchor, (0, 2, 1, 3, 4))
        print(anchor.shape)
        return anchor

def corner_to_standup_box2d_batch(boxes_corner): # ( (CenterNum * anchornum) x 4 x 2 )
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)
    return standup_boxes2d

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


# def center_to_corner_box3d(boxes_center, coordinate='lidar'):
#     # (7) -> (8, 3)
#     # N = boxes_center.shape[0]
#     ret = np.zeros((8, 3), dtype=np.float32)

#     box = [float(i) for i in boxes_center]
#     # print('jjjjjjjjjjjj{}'.format(box))
#     # box = boxes_center
#     translation = box[0:3]
#     size = box[3:6]
#     rotation = [0, 0, box[6]]

#     h, w, l = float(size[0]), float(size[1]), float(size[2])
#     print(translation)
#     print(size)
#     print(rotation)
#     trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
#         [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], 
#         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 
#         [0, 0, 0, 0, h, h, h, h]])

#     # re-create 3D bounding box in velodyne coordinate system
#     yaw = rotation[2]
#     rotMat = np.array([
#         [np.cos(yaw), -np.sin(yaw), 0.0],
#         [np.sin(yaw), np.cos(yaw), 0.0],
#         [0.0, 0.0, 1.0]])
#     cornerPosInVelo = np.dot(rotMat, trackletBox) + \
#         np.tile(translation, (8, 1)).T
#     box3d = cornerPosInVelo.transpose()

#     return box3d


def center_to_corner_box3d(box3d, xyzhwlr=False):

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    #True is xyzhwlr
    if xyzhwlr == True:
        tx,ty,tz,h,w,l,ry = [float(i) for i in box3d]
    else:
        h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]

    t_lidar = tx,ty,tz

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)

def box3d_center_to_corner(box3d):

    # def ry_to_rz(ry):
    #     angle = -ry - np.pi / 2

    #     if angle >= np.pi:
    #         angle -= np.pi
    #     if angle < -np.pi:
    #         angle = 2*np.pi + angle

    #     return angle
    def ry_normalization(ry):# 只有一个方向, -pi~pi -> 0~pi
        if ry < 0:
            ry += np.pi
            # ry+=0
        return ry

    # h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
    # h = box3d[:,0]
    # w = box3d[:,1]
    # l = box3d[:,2]
    # tx = box3d[:,3]
    # ty = box3d[:,4]
    # tz = box3d[:,5]
    # ry = box3d[:,6]
    t_lidar = np.zeros([1,3])
    t_lidar = tx, ty, tz
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    # [0, 0, 0, 0, h, h, h, h]])
                    [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # rz = ry_to_rz(ry)
    rz = ry_normalization(ry)
    # rz = ry

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    # velo_box = Box

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def box3d_corner_to_center_batch(box3d_corner):
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))

    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)


def anchors_center_to_corner(anchors): # 2D birdview ((w x h x 2), 7) -> ((w x h x 2), 4, 2)
    N = anchors.shape[0]
    # print('N is {} {}'.format(N, anchors.shape))
    anchor_corner = np.zeros((N, 4, 2))
    for i in range(N):
        anchor = anchors[i] # (7)
        translation = anchor[0:3] # (x,y,z)
        h, w, l = anchor[3:6]
        rz = anchor[-1]
        Box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2]])
        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(translation[:2], (4, 1)).T
        box2d = cornerPosInVelo.transpose()
        anchor_corner[i] = box2d
    return anchor_corner


def hwlxyzr_to_xyzhwlr(box3d_hwlxyzr):

    box3d_xyzhwlr = np.zeros(box3d_hwlxyzr.shape)
    box3d_xyzhwlr[:,0:3] = box3d_hwlxyzr[:,3:6]
    box3d_xyzhwlr[:,3:6] = box3d_hwlxyzr[:,0:3]
    box3d_xyzhwlr[:,6] = box3d_hwlxyzr[:,6]
    # box3d_xyzhwlr = np.hstack((box3d_hwlxyzr[:,3:6],box3d_hwlxyzr[:,0:3],box3d_hwlxyzr[:,6]))

    return box3d_xyzhwlr 

def caculate_3d_iou(iou2d, gt_box3d_center, lidar):
    # print('iou2d{}'.format(type(iou2d)))
    # print('lidar shape {}'.format(lidar.shape))
    iou3d = np.array(iou2d)
    iou_pos_lidar, iou_pos_gt = np.where(iou2d > 0)
    for i in range(len(iou_pos_lidar)):
        lidar_z = lidar[iou_pos_lidar[i], 2]
        lidar_h = lidar[iou_pos_lidar[i], 3]
        gt_z = gt_box3d_center[iou_pos_gt[i], 2]
        gt_h = gt_box3d_center[iou_pos_gt[i], 3]
        max_z = np.max([(lidar_z + lidar_h/2), (gt_z + gt_h/2)])
        min_z = np.min([(lidar_z - lidar_h/2), (gt_z - gt_h/2)])
        iou3d_z = np.max([(lidar_h + gt_h - (max_z - min_z)), 0])
        iou3d[iou_pos_lidar[i], iou_pos_gt[i]] = iou3d_z * iou2d[iou_pos_lidar[i], iou_pos_gt[i]]
    return iou3d


# def lidar_to_FrontView_Frustum_Voxel_show(lidar, layer_num = 1024):
#     point_num = lidar.shape[0]
#     # print('First pt:{}'.format(lidar[0]))
#     LB = [[] for i in range(64)]
#     PtoN = 0
#     k = 1
#     LB[0] = 0
#     img = mat(zeros((64, layer_num)))
#     Frustum_Voxel = [[[] for j in range(layer_num) ] for i in range(64)]
#     #find first point
#     for pts in range(point_num-1): 
#         #because y1= 0.967,find y1>0
#         if(lidar[pts, 1] > 0 and lidar[pts + 1, 1] <= 0):
#             PtoN = 1
#         if(lidar[pts, 1] < 0 and lidar[pts + 1, 1] >= 0 and PtoN == 1):
#             PtoN = 0
#             LB[k] = pts
#             k = k + 1

#     #每一层变换到中心位置的图
#     NPonitCloudLayerData = []
#     for layer in range(64):
#         layer_begin = LB[layer]
#         if (layer == 63):
#             layer_end = point_num

#         else:
#             layer_end = LB[layer+1]-1

#         LayerData = []
#         LayerData = lidar[layer_begin:layer_end, :] #/ 30
#         # np.set_printoptions(threshold = np.inf)
#         # print('x:{}    y:{}'.format(lidar[layer_begin:layer_end, 0], lidar[layer_begin:layer_end, 1]))
#         NPonitCloudLayerData.append(LayerData)
#         for i in range(len(LayerData)):
#             x = NPonitCloudLayerData[layer][i][0]
#             y = NPonitCloudLayerData[layer][i][1]
#             z = NPonitCloudLayerData[layer][i][2]
#             # theata = int((np.arctan(x / y) / np.pi * layer_num))
#             # if(y > 0):
#             #     if(x > 0):
#             #         theata = int((np.arctan(y / x) / (2 * np.pi) * layer_num))
#             #     else:  
#             #         theata = int(((np.arctan(y / x) + np.pi) / (2 * np.pi)   * layer_num))
#             # else:
#             #     if(x > 0):
#             #         theata = int(((np.arctan(y / x) + 2 * np.pi)/ (2 * np.pi)  * layer_num))
#             #     else:  
#             #         theata = int(((np.arctan(y / x) + np.pi) / (2 * np.pi ) * layer_num))
#             theata = int(((np.arctan2(y , x) + (sign(-y)+ 1) * np.pi) / (2 * np.pi) * layer_num))
#             # if(layer == 0):
#             #     print('+++++++++++++{}'.format(theata))
#             # if(theata > layer_num-2):
#             #     theata = layer_num - 1
#             # print('0000000000000{}...{}'.format(x,y))
            
#             len_r = np.sqrt(x * x + y * y)
#             point = [x, y, z]
#             Frustum_Voxel[layer][theata].append(point)
#             # print(len(Frustum_Voxel[layer][theata]))
#             if(len_r > 40):
#                 len_r = 40
#             len_r = int(len_r * 255 / 40)
#             img[layer, theata] = len_r
    

#     cv2.imwrite("1.jpg", img)
#     img3 = cv2.imread("1.jpg", 0)
#     cv2.imshow("source1", img3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return Frustum_Voxel

def lidar_to_FrontView_Frustum_Voxel(lidar, layer_num = 1024):
    
    max_pts_num = int(4096 / layer_num)
    point_num = lidar.shape[0]
    LB = np.zeros(64).astype(int64)
    # LB = [[] for i in range(64)]
    PtoN = 0
    k = 1
    LB[0] = 0
    img = mat(zeros((64, layer_num)))
    Frustum_Voxel = np.zeros((64,layer_num,max_pts_num,3))
    Frustum_Voxel_num = np.zeros((64,layer_num))
    #find first point
    for pts in range(point_num-1): 
        #because y1= 0.967,find y1>0
        if(lidar[pts, 1] > 0 and lidar[pts + 1, 1] <= 0):
            PtoN = 1
            layer_begin = pts
        if(lidar[pts, 1] < 0 and lidar[pts + 1, 1] >= 0 and PtoN == 1):
            PtoN = 0
            LB[k] = pts
            k = k + 1
    
    #每一层变换到中心位置的图
    # print('-----------------{}'.format(LB))
    # start_time = time.time()
    for layer in range(64):
        layer_begin = LB[layer]
        if (layer == 63):
            layer_end = point_num

        else:
            layer_end = LB[layer+1]-1

        LayerData = []
        # print('+++++++++++++++++++++{}   {}'.format(layer_begin,layer_end))
        LayerData = lidar[layer_begin:layer_end, :3] #/ 30
        x = lidar[layer_begin:layer_end, 0]
        y = lidar[layer_begin:layer_end, 1]
        theata = ((np.arctan2(y , x) + (sign(-y)+ 1) * np.pi) / (2 * np.pi) * layer_num).astype(np.int32) - 1
        # print(theata)
        # lay = range(64)
        voxel_coords, inv_ind, voxel_counts = np.unique(theata, axis=0, \
                                                  return_inverse=True, return_counts=True)
        # if(layer == 0):
        #     print(voxel_coords)
        #     print(voxel_coords.shape)
        #     print(theata.shape)

        # print('+++++++++++++{}'.format(inv_ind))
        # voxel_layer_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((max_pts_num,3), dtype=np.float32)
            pts = LayerData[inv_ind == i]
            if voxel_counts[i] > max_pts_num:
                pts = pts[:max_pts_num, :]
                voxel_counts[i] = max_pts_num
            # print(pts.shape)
            # augment the points
            voxel[:pts.shape[0], :] = pts
            # print('voxel{}'.format(voxel))
            Frustum_Voxel[layer, voxel_coords[i], :, : ] = voxel
            Frustum_Voxel_num[layer, voxel_coords[i]] = pts.shape[0]
        # for i in range(64):
        #     for j in range(layer_num):
        #         if Frustum_Voxel_num[i,j] != 0:
        #             img[i, j] = 1
        #         else:
        #             img[i, j] = 0
    # print('time111:{}'.format(time.time()-start_time))
    # cv2.imshow("source1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # np.set_printoptions(threshold = np.inf)
    # print(Frustum_Voxel_num)
    
    # Frustum_Voxel = np.transpose(Frustum_Voxel, (1, 0, 2, 3))
    # Frustum_Voxel_num = np.transpose(Frustum_Voxel_num, (1, 0))
    return Frustum_Voxel, Frustum_Voxel_num


def Frustum_Voxel_KeyPoint(Frustum_Voxel):
    # print('++++++++++{}   {}'.format(len(Frustum_Voxel),len(Frustum_Voxel[0])))
    lidar_center = np.zeros((Frustum_Voxel.shape[0],Frustum_Voxel.shape[1], Frustum_Voxel.shape[3]))
    # print(lidar_center.shape)
    for i in range(len(Frustum_Voxel)):
        for j in range(len(Frustum_Voxel[i])):
            pts_num = len(Frustum_Voxel[i][j])
            if pts_num > 0:
                x_all = 0
                y_all = 0
                z_all = 0
                for k in range(pts_num):
                    x_all = Frustum_Voxel[i][j][k][0] + x_all
                    y_all = Frustum_Voxel[i][j][k][1] + y_all
                    z_all = Frustum_Voxel[i][j][k][2] + z_all
                x = x_all / pts_num
                y = y_all / pts_num
                z = z_all / pts_num
                lidar_center[i,j,0] = x
                lidar_center[i,j,1] = y
                lidar_center[i,j,2] = z
    return lidar_center


def Frustum_Voxel_PesuKeyPoint(Frustum_Voxel, Frustum_Voxel_num, layer_div = 1):
    lidar_center = np.zeros((Frustum_Voxel.shape[0], Frustum_Voxel.shape[1], Frustum_Voxel.shape[3]))
    lidar_center = Frustum_Voxel[:, :, 0, :]

    # lidar_center = np.zeros(Frustum_Voxel.shape[0], Frustum_Voxel.shape[1], Frustum_Voxel.shape[3]))
    # for i in range(Frustum_Voxel_num.shape[0]):
    #     for j in range(Frustum_Voxel_num.shape[1]):
    #         k = int(Frustum_Voxel_num[i,j]/2)
    #         lidar_center[i, j, :] = Frustum_Voxel[i , j, k, :]



    # u,v = np.where(lidar_center1[..., 0] > 0) or np.where(lidar_center1[..., 1] > 0) or np.where(lidar_center1[..., 2] > 0)
    # lidar_centeradd = lidar_center1[u, v, :]
    # print(lidar_cesnteradd.shapes)
    # np.set_printoptions(threshold=np.inf)

    # lidar_center = np.concatenate((lidar_center, lidar_center1),0)
    return lidar_center

def readTarget(filename, w, root = './TargetsLabel'):
    Frustum_Voxel_path = root + '_' + str(w) + '/Frustum_Voxel/' + filename + '_' + str(w) + '.npy'
    Frustum_Voxel_num_path = root + '_' + str(w) + '/Frustum_Voxel_num/' + filename + '_' + str(w) + '.npy'
    Frustum_Voxel = np.load(Frustum_Voxel_path)
    Frustum_Voxel_num = np.load(Frustum_Voxel_num_path)
    # with open(Frustum_Voxel_path, 'r') as f:
    #     lines = f.readlines()
    # print('+++++++++++{}'.format(lines[0]))

    # for j in range(len(lines)):
    #     obj = lines[j].strip()
    #     # print('+++++++++++{}'.format(obj))
    #     # obj_class = obj[0].strip()
    # Frustum_Voxel = np.zeros((64, 512, 8, 3))
    # Frustum_Voxel_num = np.zeros((64, 512))
    return Frustum_Voxel,Frustum_Voxel_num

def writeTarget(Frustum_Voxel, Frustum_Voxel_num, filename, w, root = './TargetsLabel'):
    Frustum_Voxel_path = root + '_' + str(w) + '/Frustum_Voxel/' + filename + '_' + str(w)
    Frustum_Voxel_num_path = root + '_' + str(w) + '/Frustum_Voxel_num/' + filename + '_' + str(w)  
    print(filename)
    # print(Frustum_Voxel.shape)
    # print(Frustum_Voxel_num.shape)
    np.set_printoptions(threshold = np.inf, suppress=True)
    # t = time.time()
    np.save(Frustum_Voxel_path,Frustum_Voxel)
    np.save(Frustum_Voxel_num_path,Frustum_Voxel_num)


def FindPointnear(Frustum_Voxel, Frustum_Voxel_num):
        pointcloudsFeature = []
        for i in range(int(Frustum_Voxel.shape[0]/4)):
            Layer_pointclouds = []
            for j in range(Frustum_Voxel.shape[1]):
                ptsnum = int(Frustum_Voxel_num[i, j])
                
                pointclouds1 = Frustum_Voxel[i, j, :ptsnum, :]
                pointclouds2 = Frustum_Voxel[i + 1, j, :ptsnum, :]
                pointclouds3 = Frustum_Voxel[i + 2, j, :ptsnum, :]
                pointclouds = np.concatenate((pointclouds1, pointclouds2, pointclouds3))
        
                if pointclouds.shape[0] > 0:
                    # pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0]).unsqueeze(0).cuda()
                    pointclouds = np.expand_dims(np.array(pointclouds, dtype=np.float32).transpose([1,0]), axis=0)
                else:
                    # pointclouds = torch.zeros([3,1]).unsqueeze(0).cuda()
                    pointclouds = np.expand_dims(np.zeros([3,1]), axis=0)
                # pointclouds = self.pointnet(pointclouds.cuda())
                # Layer_pointclouds = torch.cat((Layer_pointclouds,pointclouds))
                Layer_pointclouds.append(pointclouds)
            # pointcloudsFeature = torch.cat((pointcloudsFeature,Layer_pointclouds.unsqueeze(0)))
            pointcloudsFeature.append(Layer_pointclouds)
        return pointcloudsFeature

def Frustum_Voxel_ADD(Frustum_Voxel, Frustum_Voxel_num, div):
    # div = 1
    Frustum_Voxel_new = np.zeros((int(Frustum_Voxel.shape[0]/div), Frustum_Voxel.shape[1], Frustum_Voxel.shape[2]*div*2, Frustum_Voxel.shape[3]))
    Frustum_Voxel_num_new = np.zeros((int(Frustum_Voxel_num.shape[0]/div), Frustum_Voxel_num.shape[1]))
    for i in range(int(Frustum_Voxel.shape[0]/div)):
        for j in range(int(Frustum_Voxel.shape[1])):
            ptsnumall = 0
            pointclouds = []
            for k in range(div):
                ptsnum = int(Frustum_Voxel_num[div*i + k, j])
                ptsnumall = ptsnumall + ptsnum
                pointclouds1 = Frustum_Voxel[div*i + k, j, :ptsnum, :]
                if k == 0:
                    pointclouds = pointclouds1
                else:
                    pointclouds = np.concatenate((pointclouds, pointclouds1))

            Frustum_Voxel_num_new[i, j] = ptsnumall
            Frustum_Voxel_new[i, j, :ptsnumall, :] = pointclouds
    # print('000000000000000000001111111111111111111{}'.format(Frustum_Voxel_num_new.shape))
    return Frustum_Voxel_new, Frustum_Voxel_num_new


def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    N = deltas.shape[0]

    deltas = deltas.contiguous().view(N, -1, 7)
    anchors = torch.FloatTensor(anchors.cpu())
    boxes3d = torch.zeros_like(deltas)
    # print(boxes3d.shape)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)

    anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1,2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    # print('jjjjjjjjjjjjjjjjj',deltas.shape,anchors_reshaped.shape)
    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
    # boxes3d[..., [0, 1]] = anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[...,[3]]) + anchors_reshaped[..., [2]] + 1.56

    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    # boxes3d[..., 6] = anchors_reshaped[..., 6]
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d


def lidar_center_to_Predict_center(lidar_box_center, psm, rm, point_cloud_xyz):
    #output: w*h*8*3
    batch = 0
    predict_box_out = []
    predict_scores_out = []
    # print(rm)
    boxes3d_batch = delta_to_boxes3d(rm, lidar_box_center[batch, ...])
    print('ooooooooooooooooooooooooo',boxes3d_batch.shape)
    # for batch in range(lidar_box_center.shape[0]):
    psm = psm[batch, ...]
    x, y, t= np.where(psm > cfg.posss_threshold)
    predict_scores = psm[x, y, t]
    # print(psm)
    # print('111111111111111111111111111111111111111111111111')
    print(psm.shape)
    boxes3d_batch_xyt = boxes3d_batch.reshape((1, psm.shape[0], psm.shape[1], psm.shape[2], 7))
    predict_box = boxes3d_batch_xyt[0, x, y, t, :]
    # W = psm.shape[1]
    # predict_box = boxes3d_batch[0, (x + y * W ) * (t+1), :]
        # predict_box_out.append(predict_box)
        # predict_scores_out.append(predict_scores)s
    predict_box_corner = np.zeros((predict_box.shape[0], 8, 3))

    for i in range(predict_box.shape[0]):
        predict_box_corner[i,:] = center_to_corner_box3d(predict_box[i,:], xyzhwlr = True)
    index = py_cpu_nms2(predict_box_corner, predict_scores.detach().cpu().numpy(), cfg.nms_threshold)
    predict_box_cornerall = torch.tensor(box3d_corner_to_center_batch(predict_box_corner))
    predict_box = predict_box_corner[index]
    predict_box = torch.tensor(box3d_corner_to_center_batch(predict_box))
    # print('aaaaaaaaaaa{}'.format(predict_box.shape))
    return predict_box, predict_scores, predict_box_cornerall


def py_cpu_nms_3d(dets, scores, thresh):  
    """Pure Python NMS baseline."""  
    standup_2d = utils.corner_to_standup_box2d_batch(dets)
    iou2d = bbox_overlaps(
        np.ascontiguousarray(standup_2d).astype(np.float32), # anchors_standup_2d ： ((lidarNum * anchornum), 4)
        np.ascontiguousarray(standup_2d).astype(np.float32), # gt_standup_2d ( GT ) ： (N, 4) 
    )
    # iou = utils.caculate_3d_iou(iou2d, gt_xyzhwlr, lidar_box_center)  #3D IOU
    ovr = iou2d

    order = scores.argsort()[::-1]

    keep = []  
    while order.size > 0:  
        #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr[i, :] <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
    return keep
  
  
def py_cpu_nms(dets, scores, thresh):  
    """Pure Python NMS baseline.""" 
    # inpurt 8x3  
    x1 = dets[:, 0, 0]  
    y1 = dets[:, 0, 1]  
    # z1 = dets[:, 0, 2]
    x2 = dets[:, 2, 0]  
    y2 = dets[:, 2, 1] 
    print('7777777777777',scores.shape)
    # z2 = dets[:, 2, 2] 
    # height = dets[:, 4, 2] - dets[:, 0, 2]
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
        #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
  
    return keep


def py_cpu_nms2(dets, scores, thresh):      
    x1 = dets[:, 0, 0]
    y1 = dets[:, 0, 1]
    x2 = dets[:, 2, 0]
    y2 = dets[:, 2, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  
    # order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]

def ChangeSize(psm, rm, lidar_box_center):
    pos4 = np.arange(4)
    pos128 = np.arange(128)

    Npsm = psm[:, :, pos4*2, :]
    Nrm = rm[:, :, pos4*2, :]
    Nlidar_box_center = lidar_box_center[:, pos128*2, :, :]
    return Npsm, Nrm, Nlidar_box_center

def ChangeSizeDiv2(lidar_box_center, W, H):
    posH = np.arange(int(H/2))
    posW = np.arange(int(W/2))
    # print('aaaaaaaaaaaa{}'.format(W))
    Nlidar_box_center = lidar_box_center[:, posW*2, :, ...]
    Nlidar_box_center = Nlidar_box_center[:, :, posH*2, ...]
    return Nlidar_box_center

def ChangeSizeLidarDiv2(lidar_box_center, W, H):
    posH = np.arange(int(H))
    posW = np.arange(int(W))
    # print('aaaaaaaaaaaa{}'.format(W))
    Nlidar_box_center = lidar_box_center[posH*2, :, ...]
    Nlidar_box_center = Nlidar_box_center[:, posW*2, ...]
    return Nlidar_box_center

def ShowResult(predict_box_center, PonitCloudrData):
    
    # print('aaaaaaaaaaassssssssssssssss{}'.format(predict_box_corner.shape))
    batch = 0
    x = PonitCloudrData[batch, :, 0]
    y = PonitCloudrData[batch, :, 1]
    z = PonitCloudrData[batch, :, 2]
    mlab.points3d(x, y, z, scale_factor=.05, scale_mode='vector')
    # print(box_center.shape)
    for i in range(predict_box_center.shape[0]):
    # for i in range(100,110):
        box_center = predict_box_center[i, :].detach().cpu().numpy()
        # box_center = box_center[[3,4,5,0,1,2,6]]
        box_corner = center_to_corner_box3d(box_center, xyzhwlr = True)
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # px = box_corner[:, 0]
        # py = box_corner[:, 1]
        # pz = box_corner[:, 2]
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()
    # x = lidar_box_center[batch, ..., 0]
    # y = lidar_box_center[batch, ..., 1]
    # z = lidar_box_center[batch, ..., 2]
    # r = np.ones_like(x)
    # mlab.points3d(x, y, z, r, scale_factor=.1, scale_mode='vector')
    mlab.show()


def ShowResultAnchor(predict_box_center, PonitCloudrData, div = 20):
    
    # print('aaaaaaaaaaassssssssssssssss{}'.format(predict_box_corner.shape))
    predict_box_center = predict_box_center.view(-1, 7)
    batch = 0
    x = PonitCloudrData[batch, :, 0]
    y = PonitCloudrData[batch, :, 1]
    z = PonitCloudrData[batch, :, 2]
    mlab.points3d(x, y, z, scale_factor=.05, scale_mode='vector')

    # for i in range(predict_box_center.shape[0]):
    for i in range(int(predict_box_center.shape[0]/div)):
        box_center = predict_box_center[i*div+int(div/20), :].detach().cpu().numpy()
        # box_center = box_center[[3,4,5,0,1,2,6]]
        box_corner = center_to_corner_box3d(box_center, xyzhwlr = True)
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # px = box_corner[:, 0]
        # py = box_corner[:, 1]
        # pz = box_corner[:, 2]
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()
    print(x.shape)
    x = predict_box_center[..., 0]
    y = predict_box_center[..., 1]
    z = predict_box_center[..., 2]
    r = np.ones_like(x)
    mlab.points3d(x, y, z, r, scale_factor=.1, scale_mode='vector')
    mlab.show()


def ShowResultAndGT(predict_box_center, gt_box3d_corner, PonitCloudrData): 

    batch = 0
    x = PonitCloudrData[batch, :, 0]
    y = PonitCloudrData[batch, :, 1]
    z = PonitCloudrData[batch, :, 2]
    mlab.points3d(x, y, z, scale_factor=.05, scale_mode='vector')

    # for i in range(predict_box_center.shape[0]):
    for i in range(gt_box3d_corner.shape[1]):
        box_corner = np.array(gt_box3d_corner[batch, i, ...])
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        # print(box_corner)
        mlab.points3d(px, py, pz, scale_factor=.1, scale_mode='vector', color=(0.23,0.6,1),colormap='Spectral')
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # mlab.points3d(cx, cy, cz, scale_factor=.5, scale_mode='vector')
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()


    for i in range(predict_box_center.shape[0]):
    # for i in range(100,110):
        box_center = predict_box_center[i, :].detach().cpu().numpy()
        # box_center = box_center[[3,4,5,0,1,2,6]]
        box_corner = center_to_corner_box3d(box_center, xyzhwlr = True)
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # px = box_corner[:, 0]
        # py = box_corner[:, 1]
        # pz = box_corner[:, 2]
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(1,1,0),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(1,1,0),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(1,1,0),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(1,1,0),colormap='Spectral')
        mlab.colorbar()
        print(i)
    mlab.show()

def ShowResult2(gt_box3d_corner, box2d, PonitCloudrData): 

    batch = 0
    x = PonitCloudrData[batch, :, 0]
    y = PonitCloudrData[batch, :, 1]
    z = PonitCloudrData[batch, :, 2]
    mlab.points3d(x, y, z, scale_factor=.05, scale_mode='vector')


    x = box2d[..., 0]
    y = box2d[..., 1]
    z = np.zeros_like(x)

    mlab.points3d(x, y, z, scale_factor=.2, color=(1,1,0), scale_mode='vector')


    # print(box2d.shape)
    # for i in range(int(box2d.shape[0])):
    #     box_corner = np.array(box2d[i, ...])
    #     px = np.zeros(4)
    #     py = np.zeros(4)
    #     pz = np.zeros(4)
    #     for pts in range(4):
    #         px[pts] = box_corner[pts, 0]
    #         py[pts] = box_corner[pts, 1]
    #     # print(box_corner)
    #     # mlab.points3d(px, py, pz, scale_factor=.1, scale_mode='vector', color=(0.23,0.6,1),colormap='Spectral')
    #     list1 = np.array([0,1,2,3,0])
    #     # mlab.points3d(cx, cy, cz, scale_factor=.5, scale_mode='vector')
    #     mlab.plot3d(px[list1], py[list1], pz[list1],color=(1,1,0),colormap='Spectral')
    #     print(i)
    #     mlab.colorbar()


    for i in range(gt_box3d_corner.shape[1]):
        box_corner = np.array(gt_box3d_corner[batch, i, ...])
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        # print(box_corner)
        mlab.points3d(px, py, pz, scale_factor=.1, scale_mode='vector', color=(0.23,0.6,1),colormap='Spectral')
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # mlab.points3d(cx, cy, cz, scale_factor=.5, scale_mode='vector')
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()


    mlab.show()

def ShowResult3(gt_box3d_corner, pre, PonitCloudrData): 

    batch = 0
    x = PonitCloudrData[batch, :, 0]
    y = PonitCloudrData[batch, :, 1]
    z = PonitCloudrData[batch, :, 2]
    mlab.points3d(x, y, z, scale_factor=.05, scale_mode='vector')

    # for i in range(predict_box_center.shape[0]):
    print(pre.shape)
    

    for i in range(gt_box3d_corner.shape[1]):
        box_corner = np.array(gt_box3d_corner[batch, i, ...])
        px = np.zeros(8)
        py = np.zeros(8)
        pz = np.zeros(8)
        for pts in range(8):
            px[pts] = box_corner[pts, 0]
            py[pts] = box_corner[pts, 1]
            pz[pts] = box_corner[pts, 2]
        # print(box_corner)
        mlab.points3d(px, py, pz, scale_factor=.1, scale_mode='vector', color=(0.23,0.6,1),colormap='Spectral')
        list1 = np.array([0,1,2,3,0])
        list2 = np.array([4,5,6,7,4])
        list3 = np.array([0,4,5,1])
        list4 = np.array([2,6,7,3])
        # mlab.points3d(cx, cy, cz, scale_factor=.5, scale_mode='vector')
        mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()


    for i in range(pre.shape[0]):
        box_center = np.array(pre[i, ...])
        box_corner = center_to_corner_box3d(box_center, xyzhwlr = True)
        cx = box_corner[..., 0]
        cy = box_corner[..., 1]
        cz = box_corner[..., 2]
        print(i)
        mlab.points3d(cx, cy, cz, scale_factor=.5, color=(1,1,0), scale_mode='vector')
        # px = np.zeros(8)
        # py = np.zeros(8)
        # pz = np.zeros(8)
        # for pts in range(8):
        #     px[pts] = box_corner[pts, 0]
        #     py[pts] = box_corner[pts, 1]
        #     pz[pts] = box_corner[pts, 2]
        # # print(box_corner)
        # mlab.points3d(px, py, pz, scale_factor=.1, scale_mode='vector', color=(0.23,0.6,1),colormap='Spectral')
        # list1 = np.array([0,1,2,3,0])
        # list2 = np.array([4,5,6,7,4])
        # list3 = np.array([0,4,5,1])
        # list4 = np.array([2,6,7,3])
        # # mlab.points3d(cx, cy, cz, scale_factor=.5, scale_mode='vector')
        # mlab.plot3d(px[list1], py[list1], pz[list1],color=(0.23,0.6,1),colormap='Spectral')
        # mlab.plot3d(px[list2], py[list2], pz[list2],color=(0.23,0.6,1),colormap='Spectral')
        # mlab.plot3d(px[list3], py[list3], pz[list3],color=(0.23,0.6,1),colormap='Spectral')
        # mlab.plot3d(px[list4], py[list4], pz[list4],color=(0.23,0.6,1),colormap='Spectral')
        mlab.colorbar()

    mlab.show()


#     """Computes approximate 3D IOU between a 3D bounding box 'box' and a list
#     of 3D bounding boxes 'boxes'. All boxes are assumed to be aligned with
#     respect to gravity. Boxes are allowed to rotate only around their z-axis.
#     :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]
#     :param boxes: a numpy array of the form:
#         [[ry, l, h, w, tx, ty, tz], [ry, l, h, w, tx, ty, tz]]
#     :return iou: a numpy array containing 3D IOUs between box and every element
#         in numpy array boxes.
#     """
#     # First, rule out boxes that do not intersect by checking if the spheres
#     # which inscribes them intersect.

#     if len(boxes.shape) == 1:
#         boxes = np.array([boxes])

#     box_diag = np.sqrt(np.square(box[1]) +
#                        np.square(box[2]) +
#                        np.square(box[3])) / 2

#     boxes_diag = np.sqrt(np.square(boxes[:, 1]) +
#                          np.square(boxes[:, 2]) +
#                          np.square(boxes[:, 3])) / 2

#     dist = np.sqrt(np.square(boxes[:, 4] - box[4]) +
#                    np.square(boxes[:, 5] - box[5]) +
#                    np.square(boxes[:, 6] - box[6]))

#     non_empty = box_diag + boxes_diag >= dist

#     iou = np.zeros(len(boxes), np.float64)

#     if non_empty.any():
#         height_int, _ = height_metrics(box, boxes[non_empty])
#         rect_int = get_rectangular_metrics(box, boxes[non_empty])

#         intersection = np.multiply(height_int, rect_int)

#         vol_box = np.prod(box[1:4])

#         vol_boxes = np.prod(boxes[non_empty, 1:4], axis=1)

#         union = vol_box + vol_boxes - intersection

#         iou[non_empty] = intersection / union

#     if iou.shape[0] == 1:
#         iou = iou[0]

    # return iou

def test():
    # 数据采集
    path = "/media/ct/CODE&FILE/ubuntu-beifen/ct/Code/KITTI/data/000023.bin"
    point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    point_cloud_xyz = point_cloud[:, 0:3]
    # print(KDAnchorCenter(point_cloud_xyz, 6))
    KDAnchorCenterShow(point_cloud_xyz, 1024)
    



if __name__ == '__main__':
    # a = [[1,2,3,4,5,6,7]]

    # print(hwlxyzr_to_xyzhwlr(np.array(a)))

    iou2d = np.array([[0.6, 0.1],[0.2, 0.9]])
    gt_box3d_center  = np.array([[0., 0., 0., 1., 2., 1., 0.], [0., 0., 1., 1., 2., 1., 0.]])
    lidar = np.array([[0, 0, -1, 1.5, 2, 1, 0],[0, 1 , 0.8, 1, 2, 1, 0 ]])

    iou3d = caculate_3d_iou(iou2d, gt_box3d_center, lidar)

    print('iou3d:{}'.format(iou3d))

    # test()