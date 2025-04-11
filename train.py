from model import CrossNet
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset_Yulin import BIWI ,Pose_300W_LP 
from datetime import datetime

id_GPU = 0


path_Param = "./Param/"
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
def Train(model , dev=id_GPU):

    model.to(dev)
    model.apply(init_weights)

    LR = 0.001
    optim = torch.optim.Adam(model.parameters(), lr=LR , )
   

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    transformations1 = transforms.Compose([transforms.Resize(68),
                                           transforms.RandomResizedCrop(size=64,scale=(0.8,1)),
                                           transforms.ToTensor(),
                                          normalize])
    transformations2 = transforms.Compose([transforms.Resize(68),
                                          transforms.RandomResizedCrop(64,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])
    Train_data = Pose_300W_LP("/data/300W_LP/" , 
                              "./data.txt" ,
                              transformations1,
                              )

    Test_data = BIWI(filename_path="./Data_BIWI.npz" , transform=transformations2)
    
    
    
    print(len(Train_data), "###################" , len(Test_data))
    Train_loader = DataLoader(Train_data, batch_size=16,shuffle=True,  num_workers=0)
    Test_loader = DataLoader(Test_data, batch_size=64,shuffle=False, num_workers=0)



    loss_func = nn.L1Loss().to(dev)
    loss_test = nn.L1Loss().to(dev)
    


    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("GPU_ID : %d\n"%(dev) ,"learning rate : %f\n"%(LR) , "start time : " ,formatted_time)



    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        num = 0

        for imgs, label , _  , _ in Train_loader:
            num += 1
            if epoch == 40 :
                for param_group in optim.param_groups:
                    param_group['lr'] *= 0.1
        
            imgs = imgs.to(dev)
            label = label.to(dev)
            
            outputs = model(imgs)
            loss = loss_func(outputs, label)
            loss = loss.mean()
            

            loss.backward()
           
            optim.step()
            optim.zero_grad()
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for imgs, label ,_  , _ in Test_loader:
            
                imgs = imgs.to(dev)
                label = label.to(dev)
                outputs = model(imgs)
                loss = loss_test(outputs, label)

                count = outputs.shape[0] * outputs.shape[1]
                total_loss = total_loss + loss * count

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("( {} ) {} loss(y , p , r):\n {}, ".format(formatted_time, epoch + 1,(total_loss) /15182 ))
        torch.save(model.state_dict(), "./Param/%s_%d.pth" % ("model" , epoch + 1))




if __name__ == "__main__":
    model = CrossNet(1)
    Train(model)

    


