import torch
import torch.nn as nn
from datetime import datetime
from AMF import AMF , AMF_G , AMF_L
from SSR_Layer import SSRLayer
GPU_ID = 0



input_delta = 64
input_local = 64
input_pred = 64

drop = 0.50

dim_pca = 8


class Module_3(nn.Module): 
    def __init__(self , channels , outchannels , stream = 0):
        super().__init__()
        num_inter = outchannels // 4
        if stream == 0 :
            self.Convlayer = nn.Sequential(
                nn.Conv2d(channels, num_inter, 3, 1, 1),
                nn.BatchNorm2d(num_inter),
                nn.ReLU()
            )
        else :
            self.Convlayer = nn.Sequential(
                nn.Conv2d(channels, num_inter, 3, 2, 1),
                nn.BatchNorm2d(num_inter),
                nn.ReLU()
            )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 3, 1, 1 ,groups=num_inter),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 3, 1, 1 ,groups=num_inter),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )
        self.PWConv2 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 1, 1, 0),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )

        self.PWConv3 = nn.Sequential(
            nn.Conv2d(num_inter * 2, num_inter // 2 , 1, 1, 0 ),
            nn.BatchNorm2d(num_inter // 2 ),
            nn.ReLU(),
            nn.Conv2d(num_inter // 2, num_inter  , 1, 1, 0 ),
            nn.BatchNorm2d(num_inter ),
            nn.ReLU()
        )
    def forward(self , x):
        x = self.Convlayer(x)
        x_GC1 = self.Conv1(x)
        x_GC2 = self.Conv2(x)
        x_GC2 = self.PWConv2(x_GC2)
        x_GC = torch.cat((x_GC1 , x_GC2) , dim=1)
        x_GC3 = self.PWConv3(x_GC)
        x = torch.cat((x ,x_GC1 ,x_GC2 , x_GC3) , dim=1)
        return x 

class Module_5(nn.Module): 
    def __init__(self , channels , outchannels , stream = 0):
        super().__init__()
        num_inter = outchannels // 4
        if stream == 0 :
            self.Convlayer = nn.Sequential(
                nn.Conv2d(channels, num_inter, 5, 1, 2),
                nn.BatchNorm2d(num_inter),
                nn.ReLU()
            )
        else :
            self.Convlayer = nn.Sequential(
                nn.Conv2d(channels, num_inter, 5, 2, 2),
                nn.BatchNorm2d(num_inter),
                nn.ReLU()
            )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 5, 1, 2 ,groups=num_inter),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 5, 1, 2 ,groups=num_inter),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )
        self.PWConv2 = nn.Sequential(
            nn.Conv2d(num_inter, num_inter, 1, 1, 0),
            nn.BatchNorm2d(num_inter),
            nn.ReLU()
        )

        self.PWConv3 = nn.Sequential(
            nn.Conv2d(num_inter * 2, num_inter // 2 , 1, 1, 0 ),
            nn.BatchNorm2d(num_inter // 2 ),
            nn.ReLU(),
            nn.Conv2d(num_inter // 2, num_inter  , 1, 1, 0 ),
            nn.BatchNorm2d(num_inter ),
            nn.ReLU()
        )
        
    def forward(self , x):
        x = self.Convlayer(x)
        x_GC1 = self.Conv1(x)
        x_GC2 = self.Conv2(x)
        x_GC2 = self.PWConv2(x_GC2)
        x_GC = torch.cat((x_GC1 , x_GC2) , dim=1)
        x_GC3 = self.PWConv3(x_GC)
        x = torch.cat((x ,x_GC1 ,x_GC2 , x_GC3) , dim=1)
        return x 
     
class Conv(nn.Module):
    def __init__(self , inchannels ,outchannels, Flag):
        super().__init__()
        self.Convlayer1 = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1 ),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
        self.Convlayer2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, 3, 1, 1 , groups=outchannels),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
        )
        self.PWConvlayer2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels // 2, 1, 1, 0),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU()
        )
        self.Convlayer3 = nn.Sequential(
            nn.Conv2d(outchannels // 2, outchannels // 2, 3, 1, 1 , groups=outchannels // 2),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU()
        )
        self.PWConvlayer3 = nn.Sequential(
            nn.Conv2d(outchannels // 2, outchannels, 1, 1, 0),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
        if Flag == 0 :
            self.Identity = nn.Identity()
        else:
            self.Identity = nn.AvgPool2d(2)


    def forward(self , x):
        x = self.Identity(x)
        x = self.Convlayer1(x)

        x = self.Convlayer2(x)
        x = self.PWConvlayer2(x)
        x = self.Convlayer3(x)
        x = self.PWConvlayer3(x)


        return x
    
class CrossNet(nn.Module):
    def __init__(self , mode = 0):
        super().__init__()
        self.ssr_layer = SSRLayer()

        self.Flayer1 = Conv(3 , 16 , 0)
        self.Flayer2 = Conv(16 , 32 , 1)
        self.Flayer3 = Conv(32 , 64 , 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.Flayer4 = Conv(64 , 64 , 1)
        self.Flayer5 = Conv(64 , 128 , 0)
        self.Flayer6 = Conv(128 , 64 , 1)
        self.Flayer7 = Conv(64 , 64 , 0)

        
        self.Glayer11 = Module_3(64 , 64 , 0)
        self.Glayer12 = Module_3(64 , 32 , 0)
        self.Glayer13 = Module_3(32 , 64 , 0)
        self.Glayer14 = Module_3(64 , 32 , 0)
        self.Glayer15 = Module_3(32 , 64 , 0)

        self.Glayer21 = Module_5(64 , 64 , 0)
        self.Glayer22 = Module_5(64 , 32 , 0)
        self.Glayer23 = Module_5(32 , 64 , 0)
        self.Glayer24 = Module_5(64 , 32 , 0)
        self.Glayer25 = Module_5(32 , 64 , 0)

        if mode == 0 :
            self.att1 = AMF(64)
            self.att2 = AMF(64)
            self.att3 = AMF(64)
        elif mode == 1 :
            self.att1 = AMF_L(64)
            self.att2 = AMF_L(64)
            self.att3 = AMF_L(64)
        else :
            self.att1 = AMF_G(64)
            self.att2 = AMF_G(64)
            self.att3 = AMF_G(64)
        

        self.relu = nn.ReLU()
        self.Avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

        self.M_Layer1 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_Layer2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_Layer3 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )

        self.M_deltaLayer1 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_deltaLayer2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_deltaLayer3 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )

        self.M_localLayer1 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_localLayer2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_localLayer3 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )

        self.M_predLayer1 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_predLayer2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.M_predLayer3 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )

        
        self.Encode_deltaA1 = nn.Sequential(
            nn.Linear(64 , dim_pca),
            nn.Softmax(dim=1)
        )
        self.Encode_deltaA2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_deltaA3 = nn.Sequential(
            nn.Linear(64 ,dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_deltaM1 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_deltaM2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_deltaM3 = nn.Sequential(
            nn.Linear(64 ,dim_pca ),
            nn.Softmax(dim=1)
        )

        self.Encode_localA1 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_localA2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_localA3 = nn.Sequential(
            nn.Linear(64 ,dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_localM1 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_localM2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_localM3 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )

        self.Encode_predA1 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_predA2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_predA3 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_predM1 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_predM2 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        self.Encode_predM3 = nn.Sequential(
            nn.Linear(64 , dim_pca ),
            nn.Softmax(dim=1)
        )
        
        self.delta_s1 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_delta,3),
            nn.Tanh()
        )
        self.delta_s2 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_delta,3),
            nn.Tanh()
        )
        self.delta_s3 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_delta,3),
            nn.Tanh()
        )
        self.local_s1 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_local,3),
            nn.Tanh()
        )
        self.local_s2 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_local,3),
            nn.Tanh()
        )
        self.local_s3 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_local,3),
            nn.Tanh()
        )
        self.pred_s1 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_pred,9),
            nn.ReLU(inplace = True)
        )
        self.pred_s2 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_pred,9),
            nn.ReLU(inplace = True)
        )
        self.pred_s3 = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_pred,9),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x_F1 = self.Flayer1(x)
        x_F2 = self.Flayer2(x_F1)
        x_F3 = self.Flayer3(x_F2)
        x_F4 = self.Flayer4(x_F3)
        x_F5 = self.Flayer5(x_F4)
        x_F6 = self.Flayer6(x_F5)
        x_F7 = self.Flayer7(x_F6)
        
        x_G11 = self.Glayer11(x_F7)
        x_G12 = self.Glayer12(x_G11)
        x_G13 = self.Glayer13(x_G12)
        x_G14 = self.Glayer14(x_G13)
        x_G15 = self.Glayer15(x_G14)

        x_G21 = self.Glayer21(x_F7)
        x_G22 = self.Glayer22(x_G21)
        x_G23 = self.Glayer23(x_G22)
        x_G24 = self.Glayer24(x_G23)
        x_G25 = self.Glayer25(x_G24)

        feat1 = self.relu(self.att1(x_G21 , x_G11))
        feat2 = self.relu(self.att2(x_G23 , x_G13))
        feat3 = self.relu(self.att3(x_G25 , x_G15))

        feat_A1 = self.Avgpool(feat1).view(-1 , 64)
        feat_A2 = self.Avgpool(feat2).view(-1 , 64)
        feat_A3 = self.Avgpool(feat3).view(-1 , 64)


        feat_A1 = self.Avgpool(feat1).view(-1 , 64)
        feat_A2 = self.Avgpool(feat2).view(-1 , 64)
        feat_A3 = self.Avgpool(feat3).view(-1 , 64)

        feat_deltaM1 = self.M_deltaLayer1(feat1).view(-1 , 64)
        feat_deltaM2 = self.M_deltaLayer2(feat2).view(-1 , 64)
        feat_deltaM3 = self.M_deltaLayer3(feat3).view(-1 , 64)

        feat_localM1 = self.M_localLayer1(feat1).view(-1 , 64)
        feat_localM2 = self.M_localLayer2(feat2).view(-1 , 64)
        feat_localM3 = self.M_localLayer3(feat3).view(-1 , 64)

        feat_predM1 = self.M_predLayer1(feat1).view(-1 , 64)
        feat_predM2 = self.M_predLayer2(feat2).view(-1 , 64)
        feat_predM3 = self.M_predLayer3(feat3).view(-1 , 64)

        feat_deltaA1 = self.Encode_deltaA1(feat_A1).view(-1 , dim_pca , 1)
        feat_deltaA2 = self.Encode_deltaA2(feat_A2).view(-1 , dim_pca , 1)
        feat_deltaA3 = self.Encode_deltaA3(feat_A3).view(-1 , dim_pca , 1)
        feat_deltaM1 = self.Encode_deltaM1(feat_deltaM1).view(-1 , 1 , dim_pca)
        feat_deltaM2 = self.Encode_deltaM2(feat_deltaM2).view(-1 , 1 , dim_pca)
        feat_deltaM3 = self.Encode_deltaM3(feat_deltaM3).view(-1 , 1 , dim_pca)
        feat_delta1 = ( feat_deltaA1 @ feat_deltaM1 ).view(-1 , dim_pca *dim_pca)
        feat_delta2 = ( feat_deltaA2 @ feat_deltaM2 ).view(-1 , dim_pca *dim_pca)
        feat_delta3 = ( feat_deltaA3 @ feat_deltaM3 ).view(-1 , dim_pca *dim_pca)

        feat_localA1 = self.Encode_localA1(feat_A1).view(-1 , dim_pca , 1)
        feat_localA2 = self.Encode_localA2(feat_A2).view(-1 , dim_pca , 1)
        feat_localA3 = self.Encode_localA3(feat_A3).view(-1 , dim_pca , 1)
        feat_localM1 = self.Encode_localM1(feat_localM1).view(-1 , 1 , dim_pca)
        feat_localM2 = self.Encode_localM2(feat_localM2).view(-1 , 1 , dim_pca)
        feat_localM3 = self.Encode_localM3(feat_localM3).view(-1 , 1 , dim_pca)
        feat_local1 = ( feat_localA1 @ feat_localM1 ).view(-1 , dim_pca * dim_pca)
        feat_local2 = ( feat_localA2 @ feat_localM2 ).view(-1 , dim_pca * dim_pca)
        feat_local3 = ( feat_localA3 @ feat_localM3 ).view(-1 , dim_pca * dim_pca)

        feat_predA1 = self.Encode_predA1(feat_A1).view(-1 , dim_pca , 1)
        feat_predA2 = self.Encode_predA2(feat_A2).view(-1 , dim_pca , 1)
        feat_predA3 = self.Encode_predA3(feat_A3).view(-1 , dim_pca, 1)
        feat_predM1 = self.Encode_predM1(feat_predM1).view(-1 , 1 , dim_pca)
        feat_predM2 = self.Encode_predM2(feat_predM2).view(-1 , 1 , dim_pca)
        feat_predM3 = self.Encode_predM3(feat_predM3).view(-1 , 1 , dim_pca)
        feat_pred1 = ( feat_predA1 @ feat_predM1 ).view(-1 , dim_pca * dim_pca)
        feat_pred2 = ( feat_predA2 @ feat_predM2 ).view(-1 , dim_pca * dim_pca)
        feat_pred3 = ( feat_predA3 @ feat_predM3 ).view(-1 , dim_pca * dim_pca)

        delta_s1 = self.delta_s1(feat_delta1)
        delta_s2 = self.delta_s2(feat_delta2)
        delta_s3 = self.delta_s3(feat_delta3)
        
        local_s1 = self.local_s1(feat_local1)
        local_s2 = self.local_s2(feat_local2)
        local_s3 = self.local_s3(feat_local3)

        pred_s1 = self.pred_s1(feat_pred1).view(-1,3,3)
        pred_s2 = self.pred_s2(feat_pred2).view(-1,3,3)
        pred_s3 = self.pred_s3(feat_pred3).view(-1,3,3)

        return self.ssr_layer([pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3])




