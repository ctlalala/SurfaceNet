import math
import numpy as np

class config:

    nms_threshold = 0.0

    posss_threshold = 0.9

    device_ids = [0]

    div = 4

    FV_W = 128
    FV_H = 64

    # classes
    class_list = ['Car', 'Van']

    # batch size
    N=4

    # maxiumum number of points per voxel
    T=35

    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2

    # points cloud range
    xrange = (0, 70.4)
    yrange = (-40, 40)
    zrange = (-3, 1)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)

    # iou threshold
    pos_threshold = 0.5     #0.6
    neg_threshold = 0.4     #0.45

    #   anchors: (200, 176, 2, 7) x y z h w l r
    x = np.linspace(xrange[0]+vw, xrange[1]-vw, W/2)
    y = np.linspace(yrange[0]+vh, yrange[1]-vh, H/2)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * -1.0
    w = np.ones_like(cx) * 1.6
    l = np.ones_like(cx) * 3.9
    h = np.ones_like(cx) * 1.56
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi/2
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    anchors_per_position = 2

    # non-maximum suppression
    # nms_threshold = 0.1
    # score_threshold = 0.96

if __name__=='__main__':

    cfg = config()

    print(cfg.W)
    print(cfg.H)
    print(cfg.D)
    # print(type(cfg.anchors.shape))
    # print(cfg.anchors.shape)
    # print(cfg.cx.shape)


    # print(cfg.anchors[0, 0, 0, 0:7])
    # print(cfg.anchors[1, 0, 0, 0:7])
    # print(cfg.anchors[0, 0, 1, 0:7])
    # print(cfg.anchors[1, 0, 1, 0:7])
    # print(cfg.anchors[0, 1, 0, 0:7])
    # print(cfg.anchors[0, 0, 0, 0:7])

    print('+++')
