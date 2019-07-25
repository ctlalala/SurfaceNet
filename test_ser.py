from data import Dataset,config,utils
from models import surfacenet
import time
import numpy as np
import torch
from data.config import config as cfg
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import torch
from loss import VoxelLoss
import os
import os.path
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

save_path = os.path.join('./', 'checkpointNew/')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def detection_collate(batch):
    Frustum_Voxel = []
    Frustum_Voxel_ptsnum = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    lidar_center = []
    point_cloud_xyz = []
    gt_box3d_corner = []
    Tr = []
    lidar_box_center = []
    anchors2D = []
    Anchor_box= []

    for i, sample in enumerate(batch):
        # print('1111111111111111111111111{}      {}'.format(sample[3].shape,sample[4].shape))
        Frustum_Voxel.append(sample[3])
        Frustum_Voxel_ptsnum.append(sample[4])

        # voxel_coords.append(
        #     np.pad(sample[1], ((0, 0), (1, 0)),
        #         mode='constant', constant_values=i))

        pos_equal_one.append(sample[0])
        neg_equal_one.append(sample[1])
        targets.append(sample[2])
        lidar_center.append(sample[5])
        point_cloud_xyz.append(sample[6])
        gt_box3d_corner.append(sample[7])
        Tr.append(sample[8])
        lidar_box_center.append(sample[9])
        anchors2D.append(sample[10])
        Anchor_box.append(sample[11])
        # images.append(sample[5])
        # calibs.append(sample[6])
        # ids.append(sample[7])
    return np.array(pos_equal_one),\
           np.array(neg_equal_one),\
           np.array(targets),\
           np.array(Frustum_Voxel), \
           np.array(Frustum_Voxel_ptsnum), \
           np.array(lidar_center),\
           np.array(point_cloud_xyz),\
           np.array(gt_box3d_corner),\
           np.array(Tr),\
           np.array(lidar_box_center),\
           np.array(anchors2D),\
           np.array(Anchor_box)
           
           # images, calibs, ids

torch.backends.cudnn.enabled=True

# dataset
dataset=Dataset.KittiData(cfg=cfg,root='/home/ct/KITTI',set='train', TargetLoad = False, TargetWrite = False)
data_loader = data.DataLoader(dataset, batch_size=cfg.N, num_workers=2, collate_fn=detection_collate, shuffle=False, \
                              pin_memory=False)

device_ids = cfg.device_ids
net = surfacenet.SurfaceNet().cuda(device_ids[0])
net = nn.DataParallel(net, device_ids = device_ids)
# define optimizer
epoch_size = len(dataset) // cfg.N
step_size = epoch_size*1
# optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
# define loss function
criterion = VoxelLoss(alpha=1.5, beta=1)


def train():
    meanloss_all = 0
    net.train()

    print('Initializing weights...')

    net.apply(weights_init)
    batch_iterator = None
    print('Epoch size', epoch_size)
    writer = SummaryWriter()
    for iteration in range(100*epoch_size):
        t2 = time.time()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            print('iteration:{}'.format(iteration))
                # create batch iterator
            batch_iterator = iter(data_loader)
        
            # print('model save in ./checkpoint)
        pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_ptsnum, lidar_center, point_cloud_xyz, gt_box3d_corner, Tr, lidar_box_center, anchors2D, Anchor_box = next(batch_iterator)
        # print(targets)
        # print('---+++++++++++++++++++++++-----------')
        # print(pos_equal_one.shape)
        # print(neg_equal_one.shape)
        # print(targets.shape)
        # print(Frustum_Voxel.shape)
        # print(Frustum_Voxel_ptsnum.shape)

        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # start_time = time.time()

        # Frustum_Voxel = Variable(torch.cuda.FloatTensor(Frustum_Voxel))
        # Frustum_Voxel_ptsnum = Variable(torch.cuda.FloatTensor(Frustum_Voxel_ptsnum))

        # pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one)).permute([0, 2, 1, 3])
        # neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one)).permute([0, 2, 1, 3])
        # targets = Variable(torch.cuda.FloatTensor(targets)).permute([0, 2, 1, 3])
        pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one)).cuda(device_ids[0])
        neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one)).cuda(device_ids[0])
        targets = Variable(torch.cuda.FloatTensor(targets)).cuda(device_ids[0])
		
		# zero the parameter gradients
        optimizer.zero_grad()

        # forward
        t0 = time.time()
        # print('++++++++++++++++++++++++++++++++++++++{}'.format(pos_equal_one.device))
        psm,rm = net(Frustum_Voxel,Frustum_Voxel_ptsnum)
        print('net time:{}'.format(time.time()-t0))
        # print('psm:{}'.format(psm.shape))
        # print('rm:{}'.format(rm.shape))

        # calculate loss
        # print('look shape:{}   {}   {}'.format(pos_equal_one.shape,neg_equal_one.shape,targets.shape))
        # print('4444444444444444')
        # print(rm.shape)
        # print(pos_equal_one.shape)
        # pos_x = np.arange(int(pos_equal_one.shape[1]/2))
        # pos_y = np.arange(int(pos_equal_one.shape[2]/2))
        # # print(pos_x)
        # 
        # pos_equal_one = pos_equal_one[:, pos_x*2, :, :]
        # pos_equal_one = pos_equal_one[:, :, pos_y*2, :]
        # neg_equal_one = neg_equal_one[:, pos_x*2, :, :]
        # neg_equal_one = neg_equal_one[:, :, pos_y*2, :]
        # targets = targets[:, pos_x*2, :, :]
        # targets = targets[:, :, pos_y*2, :]

        # 临时除以二匹配一下
        # targets = utils.ChangeSizeDiv2(targets, cfg.FV_W, cfg.FV_H / cfg.div)
        # neg_equal_one = utils.ChangeSizeDiv2(neg_equal_one, cfg.FV_W, cfg.FV_H / cfg.div)
        # pos_equal_one = utils.ChangeSizeDiv2(pos_equal_one, cfg.FV_W, cfg.FV_H / cfg.div)
        psm = psm.type(torch.cuda.FloatTensor)
        rm = rm.type(torch.cuda.FloatTensor)
        # print('333333333333333')
        # print(rm.shape)
        # print(pos_equal_one.shape)
        conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
        loss = conf_loss + reg_loss

        # backward
        losstime = time.time()
        loss.backward()
        # print('backward:{}'.format(time.time()-losstime))
        optimizer.step()
        scheduler.step()


        t1 = time.time()
        meanloss_all = loss.item() + meanloss_all
        meanloss = meanloss_all/(iteration+1)

        # print('Timer_net: %.4f sec.' % (t1 - t0))
        print('Timer_all: %.4f sec.' % (t1 - t2))
        print('mean_loss: ',meanloss)
        print('iter ' + repr(iteration) + ' || Loss: %.6f || Conf Loss: %.6f || Loc Loss: %.6f' % \
              (loss.item(), conf_loss.item(), reg_loss.item()))

        writer.add_scalar('data/conf_loss', conf_loss, iteration)
        writer.add_scalar('data/reg_loss', reg_loss, iteration)
        writer.add_scalar('data/loss', loss, iteration)

        if iteration % (epoch_size*2) == 0:
            time1 = time.strftime("%Y_%m_%d_%H_%M_%S")
            path = save_path + 'device' + str(device_ids[0]) + '_' + str(cfg.FV_W) + '_' + time1 + '.pkl'
            print(path)
            torch.save(net.state_dict(),path)
        print('lr:{}'.format(scheduler.get_lr()))
    writer.export_scalars_to_json("./train.json")
    writer.close()

def test():
    # dataset
    # dataset=Dataset.KittiData(cfg=cfg,root='/home/ct/KITTI',set='test', TargetLoad = True, TargetWrite = False)
    # data_loader = data.DataLoader(dataset, batch_size=cfg.N, num_workers=4, collate_fn=detection_collate, shuffle=False, \
    #                               pin_memory=False)
    path = './device0_512_2019_07_08_06_37_58.pkl'
    net = surfacenet.SurfaceNet()
    net = nn.DataParallel(net, device_ids = device_ids)
    net.load_state_dict(torch.load(path,  map_location='cuda:0'))
    batch_iterator = None
    # pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num = dataset.__getitem__(1)
    Testone = False
    if Testone == True:
        for iteration in range(25):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                    # create batch iterator
                batch_iterator = iter(data_loader)
            pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num ,lidar_center, point_cloud_xyz, gt_box3d_corner, Tr, lidar_box_center, anchors2D= next(batch_iterator)
            if iteration == 24:
                psm,rm = net(Frustum_Voxel,Frustum_Voxel_num)
                psm = torch.sigmoid(psm)
                # print(psm[0,...])
                lidar_box_center = utils.lidar_center_to_3dbox_center_surface(lidar_center, mode = 'test')
                # utils.ShowResult2(lidar_center, point_cloud_xyz)
                lidar_box_center = np.transpose(lidar_box_center, (0, 2, 1, 3, 4))
                lidar_box_center = utils.ChangeSizeDiv2(lidar_box_center, cfg.FV_W)
                rm = rm.view(rm.size(0),rm.size(1), rm.size(2),-1,7)
                # print('2375932759872953{}'.format(lidar_box_center.shape))

                # 临时用错误数据跑,以后变回128*8
                # psm,rm,lidar_box_center = utils.ChangeSize(psm, rm, lidar_box_center)

                # psm = Variable(torch.cuda.FloatTensor(psm)).cuda(device_ids[0])
                # rm = Variable(torch.cuda.FloatTensor(rm)).cuda(device_ids[0])
                lidar_box_center = Variable(torch.cuda.FloatTensor(lidar_box_center)).cuda(device_ids[0])
                # Predict_box_corner = utils.lidar_center_to_Predict_corner(lidar_box_center, psm, rm)
                predict_box_center, predict_scores, predict_box_cornerall = utils.lidar_center_to_Predict_corner(lidar_box_center, psm, rm, point_cloud_xyz)
                # print(lidar_box_center.shape)
                # print(predict_box_center.shape)
                # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa{}'.format(gt_box3d_corner))
                # utils.ShowResultGT(gt_box3d_corner, point_cloud_xyz)
                # utils.ShowResult(predict_box_center, point_cloud_xyz)
                # utils.ShowResult2(lidar_box_center, point_cloud_xyz)
                utils.ShowResult2(anchors2D, point_cloud_xyz)
    else:
        picnum = 0
        timestart = time.time()
        pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num ,lidar_center, point_cloud_xyz, gt_box3d_corner, Tr, lidar_box_center111, anchors2D, Anchor_box= dataset.__getitem__(picnum)
        psm,rm = net(Frustum_Voxel[np.newaxis, ...],Frustum_Voxel_num[np.newaxis, ...])
        psm = torch.sigmoid(psm)

        # lidar_box_center = utils.ChangeSizeDiv2(lidar_center[np.newaxis, ...], cfg.FV_H / cfg.div, cfg.FV_W)
        # # #lidar_box_center:xyzhwlr
        # lidar_box_center = utils.lidar_center_to_3dbox_center_surface(lidar_box_center, mode = 'test')

        # print('5555555555555555555555555',Anchor_box.shape)

        lidar_box_center = Anchor_box.reshape((1, int(cfg.FV_H/cfg.div/2), int(cfg.FV_W/2), 2, 7))
        lidar_box_center = np.transpose(lidar_box_center, (0, 2, 1, 3, 4))

        lidar_box_center = Variable(torch.cuda.FloatTensor(lidar_box_center)).cuda(device_ids[0])
        predict_box_center, predict_scores, predict_box_cornerall = utils.lidar_center_to_Predict_center(lidar_box_center, psm, rm, point_cloud_xyz)

        print(psm.shape)
        x, y, t= np.where(psm[0, ...] > cfg.posss_threshold)
        # x, y, t= np.where(pos_equal_one == 1)  #(256, 16, 2)
        # print(pos_equal_one.shape)
        # print(x)
        # print(y)
        preAnchor = lidar_box_center[0,x,y,t,:]    #[1, 256, 16, 2, 7]
        print('timePredict:{}'.format(time.time()-timestart))
        print('2222222222')
        print(predict_box_cornerall.shape)
        print(predict_box_center.shape)
        print(lidar_box_center.shape)       

        # utils.ShowResultAnchor(preAnchor, point_cloud_xyz[np.newaxis, ...], 1)
        # utils.ShowResultAnchor(lidar_box_center, point_cloud_xyz[np.newaxis, ...],20)
        # utils.ShowResultGT(gt_box3d_corner[np.newaxis, ...], anchors2D, point_cloud_xyz[np.newaxis, ...])
        # torch.set_printoptions(threshold=10000)
        # print(predict_box_cornerall)
        # print(predict_scores)
        # utils.ShowResult2(gt_box3d_corner[np.newaxis, ...], anchors2D, point_cloud_xyz[np.newaxis, ...])
        # utils.ShowResult3(gt_box3d_corner[np.newaxis, ...], preAnchor, point_cloud_xyz[np.newaxis, ...])
        # utils.ShowResult(predict_box_cornerall, point_cloud_xyz[np.newaxis, ...])
        utils.ShowResultAndGT(predict_box_center, gt_box3d_corner[np.newaxis, ...], point_cloud_xyz[np.newaxis, ...])

if __name__== '__main__':
    train()
    # dataset=Dataset.KittiData(cfg=cfg,root='/home/ct/KITTI',set='train', TargetLoad = False, TargetWrite = True)
    # # dataset.__getitem__(24)
    # for i in range(2780,dataset.__len__()):
    # # pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_num = dataset.__getitem__(i)
    #     dataset.__getitem__(i)
    
    # dataset = Dataset.KittiData(cfg = config.config, root = '/home/ct/KITTI')
    # print(test.__getitem__(1)[1][1][0])
    # test.PrintTest()
    # net = surfacenet.SurfaceNet()
    # net = nn.DataParallel(net, device_ids=[0]).cuda(0)
    # net = net.cuda(0)
    
    # start = time.time()
    # for i in range(dataset.__len__()):
    #     lidar, file_list, gt_hwlxyzr, pos_equal_one, neg_equal_one, targets, Frustum_Voxel, Frustum_Voxel_ptsnum = dataset.__getitem__(i)
    #     # padding = np.zeros((1, Frustum_Voxel.shape[1],Frustum_Voxel.shape[2],Frustum_Voxel.shape[3]))
    #     # print('paddingshape  {}'.format(padding.shape))
    #     # print('Frustum_Voxel  {}'.format(Frustum_Voxel.shape))
    #     # Frustum_Voxel = np.concatenate((padding, Frustum_Voxel, padding))

    #     net(Frustum_Voxel,Frustum_Voxel_ptsnum)
    #     # net(torch.from_numpy(Frustum_Voxel).cuda(0), torch.from_numpy(Frustum_Voxel_ptsnum).cuda(0))


    #     # for i in range(int(Frustum_Voxel.shape[0]/3)):
    #     # 	for j in range(Frustum_Voxel.shape[1]):
    #     # 		ptsnum = int(Frustum_Voxel_ptsnum[i, j])
    #     # 		# print('ptsnum  {}'.format(ptsnum))
    #     # 		pointclouds1 = Frustum_Voxel[i, j, :ptsnum, :]
    #     # 		pointclouds2 = Frustum_Voxel[i + 1, j, :ptsnum, :]
    #     # 		pointclouds3 = Frustum_Voxel[i + 2, j, :ptsnum, :]
    #     # 		pointclouds = np.concatenate((pointclouds1, pointclouds2, pointclouds3))
    #     # 		# pointclouds4 = Frustum_Voxel[i + 3, j, :ptsnum, :]
    #     # 		# pointclouds5 = Frustum_Voxel[i + 4, j, :ptsnum, :]
    #     # 		# pointclouds = np.concatenate((pointclouds1, pointclouds2, pointclouds3, pointclouds4, pointclouds5))
    #     # 		print(pointclouds.shape)
    #     # 		if len(pointclouds) > 0:
    #     # 			pointclouds = torch.tensor(pointclouds, dtype=torch.float32).permute([1,0]).unsqueeze(0)
    #     # 			# print(pointclouds.dtype)
    #     # 		else:
    #     # 			pointclouds = torch.zeros([3,1]).unsqueeze(0)




    #     			#print(pointclouds.dtype)

    #     		# pointclouds = pointclouds.cuda(0)
    #     		# start_t = time.time()
    #     		# feature = net(pointclouds)
    #     		# print('time  -- GPU {}'.format(time.time()-start_t))

    #     		# print(feature.shape)
        

    #     # print('time:{}'.format(end-start))
    #     # print('gt_box3d_center:{}'.format(gt_box3d_center))
    #     # print('gt_xyzhwlr{}'.format(gt_xyzhwlr))
    #     # print(utils.hwlxyzr_to_xyzhwlr(gt_hwlxyzr))
    #     # print(gt_xyzhwlr)
    #     # point_cloud_xyz = lidar[:, 0:3]

    #     # start = time.time()
    #     # utils.KDAnchorCenter(point_cloud_xyz, 8)
    #     # end = time.time()

    #     # print('time:{}'.format(end-start))
    #     # print(KDAnchorCenter(point_cloud_xyz, 6))
    #     if i>5:
    #         break

    # print('timetest:{}'.format(time.time()-start))
