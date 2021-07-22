import os
import sys 
import tensorflow as tf
import numpy as np 
import torch
import clip
from PIL import Image
import pickle
import copy
import matplotlib.pyplot as plt

def GetAlign(out,dt,model,preprocess):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))
    
    tmp=[]
    for i in range(len(imgs1)):
        
        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)
    
    image=torch.cat(tmp)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    image_features1=image_features.cpu().numpy()
    
    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])
    
    fd=image_features1[:,1:,:]-image_features1[:,:-1,:]
    
    fd1=fd.reshape([-1,512])
    fd2=fd1/np.linalg.norm(fd1,axis=1)[:,None]
    
    tmp=np.dot(fd2,dt)
    m=tmp.mean()
    acc=np.sum(tmp>0)/len(tmp)
    print(m,acc)
    return m,acc


def SplitS(ds_p,M,if_std):
    all_ds=[]
    start=0
    for i in M.mindexs:
        tmp=M.dlatents[i].shape[1]
        end=start+tmp
        tmp=ds_p[start:end]
        
        all_ds.append(tmp)
        start=end
    
    all_ds2=[]
    tmp_index=0
    for i in range(len(M.s_names)):
        if (not 'RGB' in M.s_names[i]) and (not len(all_ds[tmp_index])==0):
            
#            tmp=np.abs(all_ds[tmp_index]/M.code_std[i])
#            print(i,tmp.mean())
#            tmp=np.dot(M.latent_codes[i],all_ds[tmp_index])
#            print(tmp)
            if if_std:
                tmp=all_ds[tmp_index]*M.code_std[i]
            else:
                tmp=all_ds[tmp_index]
            
            all_ds2.append(tmp)
            tmp_index+=1
        else:
            tmp=np.zeros(len(M.dlatents[i][0]))
            all_ds2.append(tmp)
    return all_ds2

templates = ['{}', 'a photo of {}'] 

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def GetDt(classnames,model):
    text_features=zeroshot_classifier(classnames, templates, model).t()
    
    dt=text_features[0]-text_features[1]
    dt=dt.cpu().numpy()

    print(np.linalg.norm(dt))
    dt=dt/np.linalg.norm(dt)
    return dt


def GetBoundary(fs3,dt,M,threshold):
    tmp=np.dot(fs3,dt)
    
    ds_imp=copy.copy(tmp)
    select=np.abs(tmp)<threshold
    num_c=np.sum(~select)


    ds_imp[select]=0
    tmp=np.abs(ds_imp).max()
    ds_imp/=tmp
    
    boundary_tmp2=SplitS(ds_imp,M,if_std=True)
    print('num of channels being manipulated:',num_c)
    return boundary_tmp2,num_c

def GetFs(file_path):
    fs=np.load(file_path+'single_channel.npy')
    tmp=np.linalg.norm(fs,axis=-1)
    fs1=fs/tmp[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # 5*sigma - (-5)* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    return fs3


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    sys.path.append('/cs/labs/danix/wuzongze/Gan_Manipulation/play')
    from example_try import Manipulator4
    
    M=Manipulator4(dataset_name='ffhq',code_type='S')
    np.set_printoptions(suppress=True)

    file_path='/cs/labs/danix/wuzongze/Tansformer_Manipulation/CLIP/results/'+M.dataset_name+'/'
    fs3=GetFs(file_path)
    

    '''
    all_tmp=[]
    tmp=torch.load('/cs/labs/danix/wuzongze/downloads/harris_latent.pt')
    tmp=tmp.cpu().detach().numpy() #[:,:14,:]
    all_tmp.append(tmp)
    
    tmp=torch.load('/cs/labs/danix/wuzongze/downloads/ariana_latent.pt')
    tmp=tmp.cpu().detach().numpy() #[:,:14,:]
    all_tmp.append(tmp)
    
    tmp=torch.load('/cs/labs/danix/wuzongze/downloads/federer.pt')
    tmp=tmp.cpu().detach().numpy() #[:,:14,:]
    all_tmp.append(tmp)
    
    all_tmp=np.array(all_tmp)[:,0]
    
    dlatent_tmp=M.W2S(all_tmp)
    '''
    '''
    tmp=torch.load('/cs/labs/danix/wuzongze/downloads/all_cars.pt')
    tmp=tmp.cpu().detach().numpy()[:300]
    dlatent_tmp=M.W2S(tmp)
    '''
    '''
    tmp=torch.load('/cs/labs/danix/wuzongze/downloads/faces.pt')
    tmp=tmp.cpu().detach().numpy()[:100]
    dlatent_tmp=M.W2S(tmp)
    '''

#    M.viz_size=1024
    M.img_index=0
    M.num_images=30
    dlatent_tmp=[tmp[M.img_index:(M.img_index+M.num_images)] for tmp in M.dlatents]
    
    classnames=['face','face with glasses']

    dt=GetDt(classnames,model)
#    tmp=np.dot(fs3,dt)
#    
#    ds_imp=copy.copy(tmp)
#    select=np.abs(tmp)<0.1
#    num_c=np.sum(~select)
#
#
#    ds_imp[select]=0
#    tmp=np.abs(ds_imp).max()
#    ds_imp/=tmp
#    
#    boundary_tmp2=SplitS(ds_imp,M,if_std=True)
#    print('num of channels being manipulated:',num_c)
    
    boundary_tmp2=GetBoundary(fs3,dt,M,threshold=0.13)
    
    M.start_distance=-20
    M.end_distance=20
    M.step=7
#    M.num_images=100
    codes=M.MSCode(dlatent_tmp,boundary_tmp2)
    out=M.GenerateImg(codes)
    M.Vis2(str('tmp'),'filter2',out)
    
#    full=GetAlign(out,dt,model,preprocess)
    
    boundary_tmp3=copy.copy(boundary_tmp2) #primary
    boundary_tmp4=copy.copy(boundary_tmp2) #condition
    boundary_tmp2=copy.copy(boundary_tmp3)
    for i in range(len(boundary_tmp3)):
        select=boundary_tmp4[i]==0
        boundary_tmp2[i][~select]=0
