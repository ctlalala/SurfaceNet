import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import time
from data.config import config as cfg

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self,in_channels,out_channels,k,s,p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k,stride=s,padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x


class myPointNet(nn.Module):
    def __init__(self):
        super(myPointNet, self).__init__()
        # self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 256, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        # self.fc3 = nn.Linear(128, 128)

    def forward(self, x):
        # n_pts = x.size()[2]
        # print('+++++++++++++++++++++{}'.format(x.shape))
        x = F.relu(self.conv1(x))
        # pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        # if self.global_feat:
        #     return x
        # else:
        #     x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1)
        return x

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        x = self.block_1(x)
        x_skip_1 = x
        # print(x.shape)
        x = self.block_2(x)
        x_skip_2 = x
        # print(x.shape)
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        # print(x_2.shape)
        # print(x_1.shape)
        # print(x_0.shape)
        x = torch.cat((x_0,x_1,x_2),1)
        return self.score_head(x),self.reg_head(x)

# #卷积点特征，之后输入到RPN
# class FVFConv2(nn.Module):
# 	def __init__(self, FrontViewFeat_channels = 128):
# 		super(FVFConv2,self).__init__()
# 		self.FVF_C = FrontViewFeat_channels
# 		#宽度直接提前补全，所以这里不用padding
# 		self.conv2d_1 = nn.Conv2d(self.FVF_C, 128, 3, 1, p = (1, 0))  
# 		self.conv2d_2 = nn.Conv2d(128, 128, 3, 1, p = (1, 0))  
# 		self.conv2d_3 = nn.Conv2d(128, 64, 3, 1, p = (1, 0))  
# 		self.conv2d_4 = nn.Conv2d(64, 64, 3, 1, p = (1, 0))  
# 		self.conv2d_5 = nn.Conv2d(64, 64, 3, 1, p = (1, 0))  

# 	def loop_padding(self, Feat_tensor_in):
# 		Feat_tensor_out = Feat_tensor_in
# 		return Feat_tensor_out

# 	def forward(self, x):
# 		x = self.conv2d_1(self.loop_padding(x))
# 		x = self.conv2d_2(self.loop_padding(x))
# 		x = self.conv2d_3(self.loop_padding(x))
# 		x = self.conv2d_4(self.loop_padding(x))
# 		x = self.conv2d_5(self.loop_padding(x))
# 		return x

class FVFConv(nn.Module):
    def __init__(self, FrontViewFeat_channels = 128):
        super(FVFConv,self).__init__()
        self.FVF_C = FrontViewFeat_channels
        #宽度直接提前补全，所以这里不用padding
        self.conv2d_1 = Conv2d(256, 256, 3, 1, p = (1, 1))  
        self.conv2d_2 = Conv2d(256, 512, 3, 1, p = (1, 1))  
        self.conv2d_3 = Conv2d(512, 512, 3, 1, p = (1, 1))  
        self.conv2d_4 = Conv2d(512, 1024, 3, 1, p = (1, 1))  
        self.conv2d_5 = Conv2d(1024, 1024, 3, 1, p = (1, 1)) 
        self.conv2d_6 = Conv2d(1024, 128, 3, 1, p = (1, 1))
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.Linear = nn.Linear(1024, 128)

    def loop_padding(self, Feat_tensor_in):
        Feat_tensor_out = Feat_tensor_in
        return Feat_tensor_out

    def forward(self, x):
        x = self.conv2d_1(self.loop_padding(x))
        # x = self.maxpool(x)
        x = self.conv2d_2(self.loop_padding(x))
        # x = self.maxpool(x)
        x = self.conv2d_3(self.loop_padding(x))
        # x = self.maxpool(x)
        x = self.conv2d_4(self.loop_padding(x))
        # x = self.maxpool(x)
        x = self.conv2d_5(self.loop_padding(x))
        # x = self.maxpool(x)
        x = self.conv2d_6(self.loop_padding(x))
        # x = self.maxpool(x)
        # x = self.Linear(x)
        return x


class UnzipConv(nn.Module):
    def __init__(self):
        super(UnzipConv, self).__init__()
        self.conv1d = nn.Conv1d(pts, 1, 1)

    def forward(self, x):
        pass



# class FrontViewFeat(nn.Module):

#     def __init__(self):
#         super(FrontViewFeat, self).__init__()

#     def forward():
#     	return 0 


class SurfaceNet(nn.Module):

    def __init__(self):
        super(SurfaceNet, self).__init__()
        self.pointnet = myPointNet()
        self.fvf = FVFConv()
        self.rpn = RPN()
        self.device_ids = cfg.device_ids

    def Pointnear(self, Frustum_Voxel, Frustum_Voxel_num):
        pointcloudsFeatureBatch = torch.tensor([]).cuda(self.device_ids[0])
        for batch in range(Frustum_Voxel.shape[0]):
            pointcloudsFeature = torch.tensor([]).cuda(self.device_ids[0])
            for i in range(int(Frustum_Voxel.shape[1]/4)):
                Layer_pointclouds = torch.tensor([]).cuda(self.device_ids[0])
                for j in range(Frustum_Voxel.shape[2]):
                    ptsnum1 = int(Frustum_Voxel_num[batch, i, j])
                    ptsnum2 = int(Frustum_Voxel_num[batch, i + 1, j])
                    ptsnum3 = int(Frustum_Voxel_num[batch, i + 2, j])
                    ptsnum4 = int(Frustum_Voxel_num[batch, i + 3, j])
                    # print('ptsnum  {}'.format(ptsnum))
                    pointclouds1 = Frustum_Voxel[batch, i, j, :ptsnum1, :]
                    pointclouds2 = Frustum_Voxel[batch, i + 1, j, :ptsnum2, :]
                    pointclouds3 = Frustum_Voxel[batch, i + 2, j, :ptsnum3, :]
                    pointclouds4 = Frustum_Voxel[batch, i + 3, j, :ptsnum4, :]
                    pointclouds = np.concatenate((pointclouds1, pointclouds2, pointclouds3, pointclouds4))
                    # print('++++++++++pointcloudspointclouds+++++++++++{}'.format(pointclouds.shape))
                    # pointclouds4 = Frustum_Voxel[i + 3, j, :ptsnum, :]
                    # pointclouds5 = Frustum_Voxel[i + 4, j, :ptsnum, :]
                    # pointclouds = np.concatenate((pointclouds1, pointclouds2, pointclouds3, pointclouds4, pointclouds5))
                    # print(pointclouds.shape)
                    if pointclouds.shape[0] > 0:
                        pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0]).unsqueeze(0).cuda(self.device_ids[0])
                        # print(pointclouds.shape)
                    else:
                        pointclouds = torch.zeros([3,1]).unsqueeze(0).cuda(self.device_ids[0])
                    # print('+++++++pointcloudspointclouds+++++++{}'.format(pointclouds.shape))
                    # start1 = time.time()
                    pointclouds = self.pointnet(pointclouds.cuda(self.device_ids[0]))
                    # print('+++++++++++++++++++++{}'.format(time.time()-start1))
                    # pointcloudsFeature = torch.cat()
                    Layer_pointclouds = torch.cat((Layer_pointclouds,pointclouds))
                pointcloudsFeature = torch.cat((pointcloudsFeature,Layer_pointclouds.unsqueeze(0)))
            # print('++++++++++pointcloudspointclouds+++++++++++{}'.format(pointcloudsFeature.shape))
            pointcloudsFeatureBatch = torch.cat((pointcloudsFeatureBatch,pointcloudsFeature.unsqueeze(0)))
                # print(pointcloudsFeature.shape)
        return pointcloudsFeatureBatch

    def Pointnear1(self, Frustum_Voxel, Frustum_Voxel_num):
        pointcloudsFeatureBatch = torch.tensor([]).cuda(self.device_ids[0])
        for batch in range(Frustum_Voxel.shape[0]):
            pointcloudsFeature = torch.tensor([]).cuda(self.device_ids[0])
            for i in range(int(Frustum_Voxel.shape[1])):
                Layer_pointclouds = torch.tensor([]).cuda(self.device_ids[0])
                for j in range(Frustum_Voxel.shape[2]):
                    ptsnum = int(Frustum_Voxel_num[batch, i, j])
                    pointclouds = Frustum_Voxel[batch, i, j, :ptsnum, :]
                    if pointclouds.shape[0] > 0:
                        pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0]).unsqueeze(0)
                    else:
                        pointclouds = torch.zeros([3,1]).unsqueeze(0)
                    # pointclouds = torch.cat((pointclouds,pointclouds))
                    pointclouds = self.pointnet(pointclouds.cuda(self.device_ids[0]))
                    # pointclouds = pointclouds[0, :].unsqueeze(0)
                    # print('------------------------------{}'.format(pointclouds.shape))
                    Layer_pointclouds = torch.cat((Layer_pointclouds,pointclouds))
                pointcloudsFeature = torch.cat((pointcloudsFeature,Layer_pointclouds.unsqueeze(0)))
            pointcloudsFeatureBatch = torch.cat((pointcloudsFeatureBatch,pointcloudsFeature.unsqueeze(0)))

        return pointcloudsFeatureBatch

    def SpeedUp(self, Frustum_Voxel, Frustum_Voxel_num):
        # print('1111111111111111111111111111{}'.format(Frustum_Voxel.shape))
        # a = [[] for i in range(Frustum_Voxel.shape[3])]
        # aaa = torch.tensor([])
        # bbb = [aaa for i in range(Frustum_Voxel.shape[3])]
        pointcloudsFeature = [[] for i in range(Frustum_Voxel.shape[3])]
        # pointcloudsFeature1 = torch.Tensor(pointcloudsFeature)
        pointcloudsFeatureIndex = [[] for i in range(Frustum_Voxel.shape[3])]
        index_inv = np.zeros((Frustum_Voxel.shape[0], Frustum_Voxel.shape[1], Frustum_Voxel.shape[2], 2))
        for batch in range(Frustum_Voxel.shape[0]):
            for i in range(int(Frustum_Voxel.shape[1])):
                for j in range(Frustum_Voxel.shape[2]):
                    ptsnum = int(Frustum_Voxel_num[batch, i, j])
                    pointclouds = Frustum_Voxel[batch, i, j, :ptsnum, :]
                    # print('+++++++++++++{}'.format(pointclouds.shape))
                    if pointclouds.shape[0] > 0:
                        pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0])
                    else:
                        pointclouds = torch.zeros([3,1])
                    # print('+++++++++++++{}'.format(pointclouds.shape))
                    # print('-------------{}'.format(pointcloudsFeature[ptsnum].shape))
                    # pointcloudsFeature[ptsnum] = torch.cat((pointcloudsFeature[ptsnum], pointclouds))
                    pointcloudsFeature[ptsnum].append(pointclouds)
                    # print('___________________{}'.format(len(pointcloudsFeature[ptsnum])))
                    index_inv[batch, i, j, :] = np.array((ptsnum, len(pointcloudsFeature[ptsnum])))
                    index = np.array([batch, i, j])
                    pointcloudsFeatureIndex[ptsnum].append(index)
        # pointcloudsFeatureTensorAll = []
        PointNetFeatureALL = []
        PointNetFeatureNULL = self.pointnet(torch.zeros([3,1]).unsqueeze(0).cuda(self.device_ids[0]))
        for pts in range(len(pointcloudsFeature)):
            PtsnumFeature = torch.tensor([])
            for w in range(len(pointcloudsFeature[pts])):
                PtsnumFeature = torch.cat((PtsnumFeature, pointcloudsFeature[pts][w].unsqueeze(0))) 
            if PtsnumFeature.shape[0] > 0:
                PointNetFeature = self.pointnet(PtsnumFeature.cuda(self.device_ids[0]))
            else:
                PointNetFeature = PointNetFeatureNULL
            PointNetFeatureALL.append(PointNetFeature)
            # print('----------+++++++++++--------{}'.format(PointNetFeatureALL[pts].shape))

        pointcloudsFeatureBatch = torch.tensor([]).cuda(self.device_ids[0])
        for batch in range(Frustum_Voxel.shape[0]):
            pointcloudsFeaturei = torch.tensor([]).cuda(self.device_ids[0])
            for i in range(int(Frustum_Voxel.shape[1])):
                pointcloudsFeaturej = torch.tensor([]).cuda(self.device_ids[0])
                for j in range(Frustum_Voxel.shape[2]):
                    ptsnum = int(index_inv[batch, i, j, 0])
                    ind = int(index_inv[batch, i, j, 1]) - 1
                    # print(ptsnum)
                    # print(ind)
                    # print('----------------------------{}'.format(PointNetFeatureALL[ptsnum][ind, :].shape))
                    pointcloudsFeaturej = torch.cat((pointcloudsFeaturej.cuda(self.device_ids[0]), PointNetFeatureALL[ptsnum][ind, :].unsqueeze(0)))
                pointcloudsFeaturei = torch.cat((pointcloudsFeaturei, pointcloudsFeaturej.unsqueeze(0)))
            pointcloudsFeatureBatch = torch.cat((pointcloudsFeatureBatch, pointcloudsFeaturei.unsqueeze(0)))
        # print('+++++++++++++++++++++++{}'.format(pointcloudsFeatureBatch.shape))

        # PointNetFeatureNULL = self.pointnet(torch.zeros([3,1]).unsqueeze(0).cuda())
        # array = np.random.rand(Frustum_Voxel.shape[0], Frustum_Voxel.shape[1], Frustum_Voxel.shape[2], PointNetFeatureNULL.shape[1])
        # pointcloudsFeatureBatch = torch.tensor(array, dtype=torch.float32).cuda()
        # # print('-=-=-=-=-=-=-=={}'.format(pointcloudsFeatureBatch.shape))
        # for pts in range(len(pointcloudsFeature)):
        #     PtsnumFeature = torch.tensor([])
        #     for w in range(len(pointcloudsFeature[pts])):
        #         PtsnumFeature = torch.cat((PtsnumFeature, pointcloudsFeature[pts][w].unsqueeze(0))) 
        #     if PtsnumFeature.shape[0] > 0:
        #         PointNetFeature = self.pointnet(PtsnumFeature.cuda())
        #     else:
        #         PointNetFeature = PointNetFeatureNULL
        #     for u in range(len(pointcloudsFeature[pts])):
        #         batch = pointcloudsFeatureIndex[pts][u][0]
        #         i = pointcloudsFeatureIndex[pts][u][1]
        #         j = pointcloudsFeatureIndex[pts][u][2]
        #         # print('++++++++++++++{}'.format(PointNetFeature.shape))
        #         pointcloudsFeatureBatch[batch, i, j, :] = PointNetFeature[u, :]

            # else:
            #     PointNetFeature = self.pointnet(torch.zeros([3,1]).unsqueeze(0).cuda())
                # print('-=-=-=-=-=-=-=={}'.format(PointNetFeature.shape))

        # for i in range(len(pointcloudsFeature)):
        #     # print('+++++++++++++{}'.format(len(pointcloudsFeature[i])))
        #     pointcloudsFeatureTensor = torch.tensor(pointcloudsFeature[i])
        #     self.pointnet(pointcloudsFeatureTensor)

        #     pointcloudsFeatureTensorAll.append(pointcloudsFeatureTensor)           
        # for i in range(len(pointcloudsFeatureTensorAll)):
        #     print('-----------------------------{}'.format(pointcloudsFeatureTensorAll[0]))
        return pointcloudsFeatureBatch

    def SpeedUp_tensor(self, Frustum_Voxel, Frustum_Voxel_num):
        pointcloudsFeature = [torch.tensor([]) for i in range(Frustum_Voxel.shape[3])]
        # pointcloudsFeature1 = torch.Tensor(pointcloudsFeature)
        pointcloudsFeatureIndex = [[] for i in range(Frustum_Voxel.shape[3])]
        for batch in range(Frustum_Voxel.shape[0]):
            for i in range(int(Frustum_Voxel.shape[1])):
                for j in range(Frustum_Voxel.shape[2]):
                    ptsnum = int(Frustum_Voxel_num[batch, i, j])
                    pointclouds = Frustum_Voxel[batch, i, j, :ptsnum, :]
                    # print('+++++++++++++{}'.format(pointclouds.shape))
                    if pointclouds.shape[0] > 0:
                        pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0])
                    else:
                        pointclouds = torch.zeros([3,1])
                    # print('+++++++++++++{}'.format(pointclouds.shape))
                    # print('-------------{}'.format(pointcloudsFeature[ptsnum].shape))
                    pointcloudsFeature[ptsnum] = torch.cat((pointcloudsFeature[ptsnum], pointclouds.unsqueeze(0)))
                    # pointcloudsFeature[ptsnum].append(pointclouds)
                    index = np.array([batch, i, j])
                    pointcloudsFeatureIndex[ptsnum].append(index)
        # pointcloudsFeatureTensorAll = []
        PointNetFeatureNULL = self.pointnet(torch.zeros([3,1]).unsqueeze(0).cuda(self.device_ids[0]))
        array = np.zeros((Frustum_Voxel.shape[0], Frustum_Voxel.shape[1], Frustum_Voxel.shape[2], PointNetFeatureNULL.shape[1]))
        pointcloudsFeatureBatch = torch.tensor(array, dtype=torch.float32).cuda()
        # print('-=-=-=-=-=-=-=={}'.format(pointcloudsFeatureBatch.shape))
        for pts in range(len(pointcloudsFeature)):
            if pointcloudsFeature[pts].shape[0] > 0:
                PointNetFeature = self.pointnet(pointcloudsFeature[pts].cuda(self.device_ids[0]))
            else:
                PointNetFeature = PointNetFeatureNULL
            for u in range(len(pointcloudsFeature[pts])):
                batch = pointcloudsFeatureIndex[pts][u][0]
                i = pointcloudsFeatureIndex[pts][u][1]
                j = pointcloudsFeatureIndex[pts][u][2]
                # print('++++++++++++++{}'.format(PointNetFeature.shape))
                pointcloudsFeatureBatch[batch, i, j, :] = PointNetFeature[u, :]
        return pointcloudsFeatureBatch


    def forward(self, Frustum_Voxel, Frustum_Voxel_num):
        start = time.time()
        pointcloudsFeature = self.SpeedUp(Frustum_Voxel, Frustum_Voxel_num)
        time1 = time.time()
        # print('+++++++++++++++++{}'.format(Frustum_Voxel.shape))
        # pointcloudsFeature = self.Pointnear1(Frustum_Voxel, Frustum_Voxel_num)
        # print('-------------time: {}'.format(time1-start))
        # print('+++++++++++++time: {}'.format(time.time() - time1))
        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqq{}'.format(pointcloudsFeature.shape))
        # print('+++++++++++++++++++++{}'.format(time.time()-start))
        
        pointcloudsFeature = pointcloudsFeature.permute([0, 3, 2, 1]).contiguous()
        # pointcloudsFeature = pointcloudsFeature.permute([2, 1, 0]).unsqueeze(0).contiguous()
        # pointcloudsFeature = pointcloudsFeature.view(cfg.N, pointcloudsFeature.size(1), pointcloudsFeature.size(2), -1)

        # print('pointcloudsFeature:{}'.format(pointcloudsFeature.shape))
        pntime = time.time()
        x = self.fvf(pointcloudsFeature)
        # print('x_shape:{}'.format(x.shape))
        psm,rm = self.rpn(x)
        # print('here!!')
        # print(psm.shape)
        # print(rm.shape)
        print('pointnet time  {}'.format(pntime-start))
        print('rpn&fvf time  {}'.format(time.time()-pntime))
        psm = psm.permute([0, 2, 3, 1])
        rm = rm.permute([0, 2, 3, 1]) 
        return psm,rm


