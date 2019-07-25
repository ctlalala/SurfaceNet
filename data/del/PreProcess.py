from numpy import *
import numpy as np
from mayavi import mlab
import time
import cv2
import sys
import matplotlib.pyplot as plt


global SectorFlag, SectorValue, count123, sector, r_seg, theata_acc, rmax, rmin, r_len, r_translate, Value
Value = 1
r_seg = 40
theata_acc = 1 / 4
rmax = 80
rmin = 0
# use e^x - 1
r_len = (np.log(rmax + 1) - np.log(rmin + 1)) / r_seg
r_translate = np.zeros(r_seg)
SectorFlag = [[[[] for i in range(64)] for i in range(int(r_seg))] for i in
              range(int(360 / theata_acc))]
SectorValue = [[[[] for i in range(64)] for i in range(int(r_seg))] for i in
               range(int(360 / theata_acc))]
count123 = 1
sector = [[[[] for i in range(64)] for i in range(int(r_seg))] for i in
              range(int(360 / theata_acc))]  # 存储的是value信息，0～10000的数值
# sys.setrecursionlimit(1000000)


# def FindUpOctNear(SaveList, theta, r, layer):
#     # SaveList.append([theta, r])
#     if len(sector[theta][r][layer]) > 0:
#         value_Part = []
#         for i in range(len(sector[theta][r][layer])):
#             if sector[theta][r][layer][i][3] not in value_Part:
#                 value_Part.append(sector[theta][r][layer][i][3])
#         for j in range(len(value_Part)):
#
#
#     return
def Show(PonitCloudrData):
    x = PonitCloudrData[:, 0]
    y = PonitCloudrData[:, 1]
    z = PonitCloudrData[:, 2]
    r = PonitCloudrData[:, 3]
    mlab.points3d(x, y, z, r, scale_factor=.05, scale_mode='vector')
    mlab.show()


def NormalizeShow(PonitCloudrData):
    x1 = np.array(PonitCloudrData[:, 0])
    y1 = np.array(PonitCloudrData[:, 1])
    z1 = np.array(PonitCloudrData[:, 2])
    cv2.normalize(PonitCloudrData[:, 0], x1, alpha=200, beta=0, dtype=-1)
    cv2.normalize(PonitCloudrData[:, 1], y1, alpha=200, beta=0, dtype=-1)
    cv2.normalize(PonitCloudrData[:, 2], z1, alpha=100, beta=0, dtype=-1)
    r = PonitCloudrData[:, 3]
    show = mlab.points3d(x1, y1, z1, r, scale_factor=.05, scale_mode='vector')
    mlab.show()


def LayerBreakPoint(layer_point_num, PonitCloudLayerData, point_cloud_xyzr):
    Layer_BreakPoint = []
    for layer in range(64):
        layer_begin = layer_point_num * layer
        layer_end = layer_point_num * (layer + 1)
        PonitCloudLayerData[:, :, layer] = point_cloud_xyzr[layer_begin:layer_end, :]
        # begin to process
        center_x = 0
        center_y = 0
        threshole_rec = 0.1
        threshole_length = 1
        layer_x = PonitCloudLayerData[:, 0, layer]
        layer_y = PonitCloudLayerData[:, 1, layer]
        BreakPoint = []
        for i in range(layer_point_num - 1):
            neighborhood_K = (layer_y[i] - layer_y[i + 1]) / (layer_x[i] - layer_x[i + 1])
            center_K = (layer_y[i] - center_y) / (layer_x[i] - center_x)
            neighborhood_length = np.sqrt(
                np.square(layer_y[i] - layer_y[i + 1]) + np.square(layer_x[i] - layer_x[i + 1]))
            if (abs(neighborhood_K - center_K) < threshole_rec or
                    neighborhood_length > threshole_length):
                BreakPoint.append(i)
        # if(len(BreakPoint) > 0):
        #     Layer_BreakPoint.append(BreakPoint)
        Layer_BreakPoint.append(BreakPoint)
    return Layer_BreakPoint


def BreakPath(Layer_BreakPoint, PonitCloudLayerData):
    Part_Data = []
    Part_color = []
    for layer in range(64):
        Layer_Part_Data = []
        for i in range(len(Layer_BreakPoint[layer]) - 1):
            part = PonitCloudLayerData[Layer_BreakPoint[layer][i]:Layer_BreakPoint[layer][i + 1], :, layer]
            Layer_Part_Data.append(part)
        # # add up first and last part
        # last_startPts = Layer_BreakPoint[layer][i+1]
        # last_endPts = len(PonitCloudLayerData)
        # first_startPts = 0
        # first_endPts = Layer_BreakPoint[layer][1]
        # part_last = PonitCloudLayerData[last_startPts:last_endPts, :, layer]
        # part_first = PonitCloudLayerData[first_startPts:first_endPts, :, layer]
        # part = np.vstack((np.array(part_last), np.array(part_first)))
        # #整合起来的作为第一个path
        # Part_Data.append(part)
        Part_Data.append(Layer_Part_Data)
    # oneD = sum(Part_Data, [])
    return Part_Data


def rewrite_r(Part_Data):
    # 把每个Part的list转换成np进行组合,功能:每层分块上色
    Part_Data = np.array(Part_Data)
    Part_Data_all = np.empty([1, 4])
    # Part_record = np.empty([1, 1])
    k = 0
    # p = 0
    for layer in range(64):
        for i in range(len(Part_Data[layer])):
            # Part_Data[layer][i][:, 3] = k / 10                              #颜色分为十等级
            # k = k + 1
            # if k > 9:
            #     k = 0
            Part_Data[layer][i][:, 3] = k  # 先记录part信息
            k = k + 1
            Part_Data_all = np.vstack((Part_Data_all, Part_Data[layer][i]))
    return Part_Data_all


def SectorDefine(layer_point_num, PonitCloudLayerData):
    # 扇形分割体素,定义精度
    global sector
    for i in range(r_seg):
        r_translate[i] = np.exp(r_len * i) - 1
    # print(r_translate)
    # sector = np.empty([int(360/theata_acc), int(15/r_acc), 64])
    # sector.tolist()
    time_start = time.time()
    r = 0

    for layer in range(64):
        for i in range(layer_point_num):
            x = PonitCloudLayerData[i, 0, layer]
            y = PonitCloudLayerData[i, 1, layer]
            theata = int(np.trunc((np.arctan(y / x) / np.pi) * 360 / theata_acc) + 180) - 1
            # r = int(np.trunc(np.sqrt(x*x + y*y)*r_seg))
            len_r = np.sqrt(x * x + y * y)
            for j in range(r_seg - 1, 0, -1):
                # print('len_r',len_r)
                # print(r_translate[j])
                if len_r > r_translate[j]:
                    r = j
                    break
            # if r < int(rmax*r_seg):
            if PonitCloudLayerData[i, 3, layer] not in sector[theata][r][layer]:
                sector[theata][r][layer].append(PonitCloudLayerData[i, 3, layer])
                # print(theata, '&', r, '&', PonitCloudLayerData[i, 3, layer])
            # sector[theata][r][layer][:] = []
    return sector  # sector只存了在这个扇形区域内的点的value，也就是之前rewrite_r()里面的k


def MyIterator(thetaValue, rValue, layerValue, Value):
    global count123, SectorValue, SectorFlag
    for layer1 in range(layerValue-1, layerValue + 2):
        if 0 <= layer1 < 64:
            theta_start = thetaValue - theta_size + int(360 / theata_acc)
            theta_end = thetaValue + theta_size + 1 + int(360 / theata_acc)
            for theta1 in range(theta_start, theta_end):
                theta1 = theta1 % (int(360 / theata_acc))
                r_start = rValue - r_size
                r_end = rValue + r_size + 1
                # print(r_start,r_end)
                for r1 in range(r_start, r_end):
                    if r_start >= 0 and r_end < r_seg:
                        if SectorFlag[theta1][r1][layer1] == 1:
                            SectorValue[theta1][r1][layer1] = Value
                            SectorFlag[theta1][r1][layer1] = 0
                            count123 += 1
                            # print(theta1, r1, layer1, Value)
                            MyIterator(theta1, r1, layer1, Value)
                            return 0
                        else:
                            return 0


def NewFindContours(sector):
    global count123, SectorValue, SectorFlag
    # 定义一个sectorflag
    for layer in range(64):
        for r in range(r_seg):
            for theta in range(int(360 / theata_acc)):
                if len(sector[theta][r][layer]) > 0:
                    SectorFlag[theta][r][layer] = 1
                else:
                    SectorFlag[theta][r][layer] = 0
    # 深度遍历求分割,标记后的点清空
    global r_size, theta_size, Value
    r_size = 1
    theta_size = 2
    for layer in range(64):
        # print('layer'+str(layer))
        for r in range(r_seg):
            for theta in range(int(360 / theata_acc)):
                # print('theta'+str(theta))
                if SectorFlag[theta][r][layer] == 1:
                    MyIterator(theta, r, layer, Value)
                    Value += 1
                    # print(count123)
                    a=1
    # seesee = [[[[] for i in range(int(360 / theata_acc))] for i in range(64)] for i in
    #               range(int(r_seg))]
    # for i in range(64):
    #     seesee[i][:][:] = SectorValue[:][:][i]
    # return seesee
    # for i in range(64):
    #     for j in range(r_seg):
    #         print(SectorValue[:][j][i])
    # print(Value)
    # print(SectorValue)


    # 遍历所有点,归队
    All_Part = []
    All_Value = []
    # color = 0
    for layer in range(64):
        for i in range(layer_point_num):
            x = PonitCloudLayerData[i, 0, layer]
            y = PonitCloudLayerData[i, 1, layer]
            theata = int(np.trunc((np.arctan(y / x) / np.pi) * 360 / theata_acc) + 180) - 1
            len_r = np.sqrt(x * x + y * y)
            for j in range(r_seg - 1, 0, -1):
                if len_r > r_translate[j]:
                    r = j
                    break
            value = SectorValue[theata][r][layer]  # 后期考虑多个点
            if value not in All_Value:
                All_Value.append(SectorValue[theata][r][layer])
                All_Part.append([PonitCloudLayerData[i, :, layer]])
            else:
                pos = All_Value.index(value)
                All_Part[pos].append(PonitCloudLayerData[i, :, layer])

    # 上色
    color = np.empty(len(All_Value))
    k = 0
    for i in range(len(color)):
        # if k % 2 == 0:
        #     color[i] = k / 20
        # else:
        #     color[i] = 1 - k / 20
        # k = k + 1
        # if k > 20:
        #     k = 0
        color[i] = k / 10                              #颜色分为十等级
        k = k + 1
        if k > 9:
            k = 0
    Finall_Finall_Part = np.empty([1, 4])
    for i in range(len(All_Part)):
        for j in range(len(All_Part[i])):
            All_Part[i][j][3] = color[i]
            Finall_Finall_Part = np.vstack((Finall_Finall_Part, All_Part[i][j]))

    return Finall_Finall_Part
    # print(SectorValue)

def FindCounters(sector):
    for i in range(r_seg):
        r_translate[i] = np.exp(r_len * i) - 1
    # 遍历求上一层九领域
    Final_Part = []  # 位置信息，只是记录每个点的r和theta的值，即所在的扇形区域
    Value_List = []  # Value信息
    for layer in range(64):
        Final_Part_Layer = []  # 存储点在扇形区域的theta和r
        Value_List_Layer = []  # 存储扇形内的点的value
        for r in range(r_seg):
            for theata in range(int(360 / theata_acc)):
                if len(sector[theata][r][layer]) > 0:  # 后期考虑多个点
                    # for i in range(len(sector[theata][r][layer])):
                    value = sector[theata][r][layer][0]
                    if value not in Value_List_Layer:
                        Value_List_Layer.append(value)
                        Final_Part_Layer.append([[theata, r]])
                    else:
                        Final_Part_Layer[Value_List_Layer.index(value)].append([theata, r])
                    # sector[theta][r][layer][:] = []  #清空
        Final_Part.append(Final_Part_Layer)
        Value_List.append(Value_List_Layer)

    # 开始找九领域
    flag = 1
    theta_size = 1
    r_size = 1
    sector_flag = sector
    for layer in range(63):
        for i in range(len(Final_Part[layer])):
            for j in range(len(Final_Part[layer][i])):
                [u, v] = Final_Part[layer][i][j]
                # print(u,'&',v)
                for x in range(u - theta_size, u + theta_size + 1):
                    if x >= 0 and x < int(360 / theata_acc):
                        for y in range(v - r_size, v + r_size + 1):
                            if y >= 0 and y < r_seg:
                                if len(sector_flag[x][y][layer + 1]) > 0:  # 考虑多个点
                                    for length in range(len(sector[x][y][layer + 1])):  # 根据这个索引找出所有layer层位置然后赋值
                                        value = sector[x][y][layer + 1][length]
                                        if value in Value_List[layer + 1]:
                                            pos = Value_List[layer + 1].index(value)
                                            arr = Final_Part[layer + 1][pos]
                                            for arr_i in range(len(arr)):
                                                [u1, v1] = arr[arr_i]
                                                # if sector_flag[u1][v1][layer + 1][0] != 0:
                                                sector[u1][v1][layer + 1][0] = sector[u][v][layer][0]  # 后期考虑多个点
                                                # for de in range(len(sector_flag[u1][v1][layer + 1]))  # 把那些处理过的点的flag都清了
                                                # sector_flag[u1][v1][layer + 1][:] = []                                                              #注释了这里！！！！
                                                break
                                            break
                                            # if flag:
                                            #     print(arr[arr_i])
                                            #     flag = 0

                                            # sector[u1][v1][layer + 1][0] = sector[u][v][layer][0]         #考虑多个点
    for layer in range(63, 0, -1):
        for i in range(len(Final_Part[layer])):
            for j in range(len(Final_Part[layer][i])):
                [u, v] = Final_Part[layer][i][j]
                # print(u,'&',v)
                for x in range(u - theta_size, u + theta_size + 1):
                    if x >= 0 and x < int(360 / theata_acc):
                        for y in range(v - r_size, v + r_size + 1):
                            if y >= 0 and y < r_seg:
                                if len(sector[x][y][layer - 1]) > 0:  # 考虑多个点
                                    for length in range(len(sector[x][y][layer - 1])):
                                        value = sector[x][y][layer - 1][length]
                                        if value in Value_List[layer - 1]:
                                            pos = Value_List[layer - 1].index(value)
                                            arr = Final_Part[layer - 1][pos]
                                            for arr_i in range(len(arr)):
                                                [u1, v1] = arr[arr_i]
                                                sector[u1][v1][layer - 1][0] = sector[u][v][layer][0]  # 后期考虑多个点


    # 遍历所有点,归队
    All_Part = []
    All_Value = []
    # color = 0
    for layer in range(64):
        for i in range(layer_point_num):
            x = PonitCloudLayerData[i, 0, layer]
            y = PonitCloudLayerData[i, 1, layer]
            theata = int(np.trunc((np.arctan(y / x) / np.pi) * 360 / theata_acc) + 180) - 1
            len_r = np.sqrt(x * x + y * y)
            for j in range(r_seg - 1, 0, -1):
                if len_r > r_translate[j]:
                    r = j
                    break
            value = sector[theata][r][layer][0]  # 后期考虑多个点
            if value not in All_Value:
                All_Value.append(sector[theata][r][layer][0])
                All_Part.append([PonitCloudLayerData[i, :, layer]])
            else:
                pos = All_Value.index(value)
                All_Part[pos].append(PonitCloudLayerData[i, :, layer])

    # 上色
    color = np.empty(len(All_Value))
    k = 0
    for i in range(len(color)):
        if k % 2 == 0:
            color[i] = k / 20
        else:
            color[i] = 1 - k / 20
        k = k + 1
        if k > 20:
            k = 0
    Finall_Finall_Part = np.empty([1, 4])
    for i in range(len(All_Part)):
        for j in range(len(All_Part[i])):
            All_Part[i][j][3] = color[i]
            Finall_Finall_Part = np.vstack((Finall_Finall_Part, All_Part[i][j]))

    return Finall_Finall_Part


# def main():
#     # 数据采集
#     path = "/media/ct/CODE&FILE/ubuntu-beifen/ct/Code/KITTI/data/000023.bin"
#     point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
#     global layer_point_num, PonitCloudLayerData
#     point_cloud_xyzr = point_cloud[:, 0:4]
#     point_num = point_cloud_xyzr.shape[0]
#     layer_point_num = int(point_num / 64)
#     print(layer_point_num)
#     PonitCloudLayerData = np.zeros((layer_point_num, 4, 64))
#     print(point_num)
#     # 全部数据分开每层并记录断点返回
#     Layer_BreakPoint = LayerBreakPoint(layer_point_num, PonitCloudLayerData, point_cloud_xyzr)
#     # 根据断点找到每一层每一个path
#     Part_Data = BreakPath(Layer_BreakPoint, PonitCloudLayerData)
#     # 根据每个path不同重写R值以便后面标记每个不同的path
#     Part_Data_all = rewrite_r(Part_Data)
#     # 记录时间
#     time_start = time.time()
#     # 定义扇形空间,得出每个点在哪个扇形空间内
#     sector = SectorDefine(layer_point_num, PonitCloudLayerData)
#     # 寻找九领域counters
#     # Finall_Finall_Part = FindCounters(sector, layer_point_num, PonitCloudLayerData)
#     Finall_Finall_Part = FindCounters(sector)
#     time_end = time.time()
#     print('totally cost', time_end - time_start)
#     Show(Finall_Finall_Part)

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
    theata_acc = 3  #360*6=2160
    img = mat(ones((64, 360 * theata_acc)))
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
            theata = int(((np.arctan(y / x) / np.pi * 360 ) * theata_acc)) - 1
            len_r = np.sqrt(x * x + y * y)
            if(len_r > 40):
                len_r = 40
            len_r = int(len_r * 255 / 40)
            # img[8 * layer, theata] = len_r
            # img[8 * layer+1, theata] = len_r
            # img[8 * layer+2, theata] = len_r
            # img[8 * layer+3, theata] = len_r
            # img[8 * layer+4, theata] = len_r
            # img[8 * layer+5, theata] = len_r
            # img[8 * layer+6, theata] = len_r
            # img[8 * layer+7, theata] = len_r
            # img[4 * layer, theata] = len_r
            # img[4 * layer+1, theata] = len_r
            # img[4 * layer+2, theata] = len_r
            # img[4 * layer+3, theata] = len_r
            img[layer, theata] = len_r
    # cv2.normalize(img, img, 0, 255, cv2.NORM_HAMMING)

    # print(img)
    # img = img.astype(np.int)
    cv2.imshow("1", img)
    cv2.imwrite("1.jpg", img)
    img3 = cv2.imread("1.jpg", 0)
    cv2.imshow("2", img3)
    # img3 = cv2.bilateralFilter(img3, 9, 75, 75)
    img3 = cv2.medianBlur(img3, 3)
    cv2.imshow("pppp", img3)
    img3 = cv2.medianBlur(img3, 3)
    cv2.imshow("ppp3333p", img3)
    print(img)
    print(img3)
    # imgnum = img.flatten()
    # print(imgnum)
    # plt.hist(imgnum)
    # plt.show()
    # print(img[1, 3])
    # print("111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    # print(img3[1, 3])
    # print(img2)
    # img1 = cv2.medianBlur(img, 5)
    # cv2.imshow("2", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Show(point_cloud_xyzr)

if __name__ == '__main__':
    main()
