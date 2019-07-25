from numpy import *
import numpy as np
from mayavi import mlab
import time
import cv2
import sys
import matplotlib.pyplot as plt

def Show(PonitCloudrData):
    x = PonitCloudrData[:, 0]
    y = PonitCloudrData[:, 1]
    z = PonitCloudrData[:, 2]
    r = PonitCloudrData[:, 3]
    mlab.points3d(x, y, z, r, scale_factor=.05, scale_mode='vector')
    mlab.show()

def LinemedianBlur(img):
    simg = img
    for i in range(img.shape[0]):
        for j in range(1, img.shape[1]-1):
            for k in range(3):
                # if(simg[i,j] == 0 and simg[i, j-1]>0 and simg[i, j+1]>0):
                #     img[i,j] = int((img[i,j-1]+img[i,j+1])/2)
                img[i, j] = int(simg[i, j - 1] + simg[i, j + 1] + simg[i, j]-min(simg[i, j - 1] , simg[i, j + 1] , simg[i, j])-max(simg[i, j - 1] , simg[i, j + 1] , simg[i, j]))
    return img


def main():
    # 数据采集
    path = "/media/ct/CODE&FILE/ubuntu-beifen/ct/Code/KITTI/data/000023.bin"
    point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    global layer_point_num, PonitCloudLayerData
    point_cloud_xyzr = point_cloud[:, 0:4]
    point_num = point_cloud_xyzr.shape[0]
    layer_point_num = int(point_num / 64)
    PonitCloudLayerData = np.zeros((layer_point_num, 4, 64))
    LB = [[] for i in range(64)]
    PtoN = 0
    k = 1
    LB[0] = 0
    maxLB = 0
    minLB = 65535
    #find first point
    for pts in range(point_num-1):
        #because y1= 0.967,find y1>0
        if(point_cloud_xyzr[pts, 1] > 0 and point_cloud_xyzr[pts + 1, 1] <= 0):
            PtoN = 1
        if(point_cloud_xyzr[pts, 1] < 0 and point_cloud_xyzr[pts + 1, 1] >= 0 and PtoN == 1):
            PtoN = 0
            LB[k] = pts
            if((LB[k]-LB[k-1]) > maxLB):
                maxLB = LB[k]-LB[k-1]
            if ((LB[k] - LB[k - 1]) < minLB):
                minLB = LB[k] - LB[k - 1]
            k = k + 1

    print(maxLB)
    print(minLB)
    # print(LB)
    # print(layer_point_num)
    #每一层变换到中心位置的图
    NPonitCloudLayerData = []
    theata_acc = 6  #360*6=2160
    maxLB = int(maxLB / 2)
    img = mat(zeros((64, maxLB)))
    for layer in range(64):
        layer_begin = LB[layer]
        if (layer == 63):
            layer_end = point_num

        else:
            layer_end = LB[layer+1]-1
        LayerData = []
        LayerData = point_cloud_xyzr[layer_begin:layer_end, :] #/ 30
        NPonitCloudLayerData.append(LayerData)
        for i in range(len(LayerData)):
            x = NPonitCloudLayerData[layer][i][0]
            y = NPonitCloudLayerData[layer][i][1]
            theata = int((np.arctan(y / x) / np.pi * maxLB)) - 1
            len_r = np.sqrt(x * x + y * y)
            if(len_r > 40):
                len_r = 40
            len_r = int(len_r * 255 / 40)
            img[layer, theata] = len_r

    maxLB = int(maxLB / 2)
    img2 = mat(zeros((64, maxLB)))
    for layer in range(64):
        layer_begin = LB[layer]
        if (layer == 63):
            layer_end = point_num

        else:
            layer_end = LB[layer + 1] - 1
        LayerData = []
        LayerData = point_cloud_xyzr[layer_begin:layer_end, :]  # / 30
        NPonitCloudLayerData.append(LayerData)
        for i in range(len(LayerData)):
            x = NPonitCloudLayerData[layer][i][0]
            y = NPonitCloudLayerData[layer][i][1]
            theata = int((np.arctan(y / x) / np.pi * maxLB)) - 1
            len_r = np.sqrt(x * x + y * y)
            if (len_r > 40):
                len_r = 40
            len_r = int(len_r * 255 / 40)
            img2[layer, theata] = len_r

    # cv2.imshow("1", img)
    print(img[1, 128])
    print(img.shape[1])

    cv2.imwrite("1.jpg", img)
    img3 = cv2.imread("1.jpg", 0)
    cv2.imshow("source1", img3)
    img4 = cv2.medianBlur(img3, 3)
    cv2.imshow("medianBlur1", img4)
    img5 = LinemedianBlur(img3)
    cv2.imshow("LinemedianBlur1", img5)

    # cv2.imwrite("2.jpg", img2)
    # img3 = cv2.imread("2.jpg", 0)
    # cv2.imshow("source2", img3)
    # img3 = cv2.medianBlur(img3, 3)
    # cv2.imshow("medianBlur2", img3)
    Show(point_cloud_xyzr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()