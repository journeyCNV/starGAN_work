import dlib, imageio, glob ,cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from main_func import main

def find_face(img):
    #人脸分类器
    detector = dlib.get_frontal_face_detector()
    #获取人脸检测器
    predictor=dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"
    )
    
    num = 1
    count = 0
    startX=0
    startY=0
    endX=0
    endY=0
    
    print('are you ok?')
    #1 对图片进行上采样一次，有利于检测到更多的人脸
    dets = detector(img,1)
    
    yuan_pot = [] #记录初始框的坐标
    for d in dets:
        shape = predictor(img,d) #寻找人脸的68个标定点
        chang=[]
        kuan=[]
        for pt in shape.parts():
            chang.append(pt.x)
            kuan.append(pt.y)
        startX = min(chang)
        startY = min(kuan)
        endX = max(chang)
        endY = max(kuan)
        
        if startX>=0&startY>=0&endX>=0&endY>=0:
            temp_frame = [startX,endX,startY,endY]
            yuan_pot.append(temp_frame)  #存储初始框
            
            
            centerX = (startX+endX)/2
            centerY = (startY+endY)/2
            #num是裁剪放宽的倍数
            startX = (startX-centerX)*num+centerX
            startY = (startY-centerY)*num+centerY
            endX = (endX-centerX)*num+centerX
            endY = (endY-centerY)*num+centerY
            #rect=dlib.rectangle(int(startX),int(startY),int(endX),int(endY))
            
        
            pic = img
            piece = pic[int(startY):int(endY),int(startX):int(endX)]
            piece = Image.fromarray(piece)
        
            #将裁剪下来的piece放大
            #transforms.Resize(size)
            #如果size为一个int值，则将短边resize成这个尺寸，长边安装对应比例进行缩放
            size = 256
            #transforms.CenterCrop((size, size)) 按中心裁剪  好像应该并没有这个必要
            transform1 = transforms.Compose([
            transforms.Resize(size)])
    
            mode = transform1(piece)
            #print(type(mode))
        
            mode.save(f'data/RaFD/test/happy/{count}img_new.jpg')
            count+=1
            #print(count)
        
    print("ok\n")
    return yuan_pot
        
        
def change_face(test_iters):
    return main(test_iters)


#坐标、生成图像片、原图
def revert_face(pot,img_array,pic):
    for i in range(len(pot)):
        size = 0  #源切片短边
        lsize = 0   #源切片长边
        if (pot[i][3]-pot[i][2]) > (pot[i][1]-pot[i][0]):
            lsize = pot[i][3]-pot[i][2]
            size = pot[i][1]-pot[i][0]
            dis = (lsize - size)/2
            pot[i][2]+=dis
            pot[i][3]-=dis
        else:
            lsize = pot[i][1]-pot[i][0]
            size = pot[i][3]-pot[i][2]
            dis = (lsize - size)/2
            pot[i][0]+=dis
            pot[i][1]-=dis
        
        transform_m = transforms.Compose([transforms.Resize(size)])
        piece = img_array[i]
        #img_array是CUDA tensor
        #将tensor转换成numpy之间转换会出事情 维度不一样 Transpose一定要加
        piece = piece.cpu().numpy()
        #piece = np.transpose(piece,(1,2,0)) 这样还是不行
        piece = (np.transpose(piece,(1,2,0))+1)/1.0*255.0
        piece = Image.fromarray(np.uint8(piece))
        piece = transform_m(piece)
        #print(piece)
        #print(pot[i])
        piece = np.array(piece)
        pic[int(pot[i][2]):int(pot[i][3]),int(pot[i][0]):int(pot[i][1])] = piece
        #pppp = Image.fromarray(np.uint8(pic))
        #pppp.save(f'ans.jpg')
    print('finished')
    #-----删除缓存图片------#
    paths = glob.glob('data/RaFD/test/happy/*.jpg')
    for f in paths:
        os.remove(f)
    return pic


#加泊松融合
def revert_face_bs(pot,img_array,pic):
    #泊松融合参数要uint8
    pic = np.uint8(pic)
    for i in range(len(pot)):
        size = 0  #源切片短边
        lsize = 0   #源切片长边
        if (pot[i][3]-pot[i][2]) > (pot[i][1]-pot[i][0]):
            lsize = pot[i][3]-pot[i][2]
            size = pot[i][1]-pot[i][0]
            dis = (lsize - size)/2
            pot[i][2]+=dis
            pot[i][3]-=dis
        else:
            lsize = pot[i][1]-pot[i][0]
            size = pot[i][3]-pot[i][2]
            dis = (lsize - size)/2
            pot[i][0]+=dis
            pot[i][1]-=dis
        
        transform_m = transforms.Compose([transforms.Resize(size)])
        piece = img_array[i]
        #img_array是CUDA tensor
        #将tensor转换成numpy之间转换会出事情 维度不一样 Transpose一定要加
        piece = piece.cpu().numpy()
        #piece = np.transpose(piece,(1,2,0)) 这样还是不行
        piece = (np.transpose(piece,(1,2,0))+1)/1.0*255.0
        piece = Image.fromarray(np.uint8(piece))
        piece = transform_m(piece)
        #print(piece)
        #print(pot[i])
        piece = np.array(piece)
        
        #--------泊松融合--------#
        piece = np.uint8(piece)
        #目标区域的中心
        center=(int((pot[i][1]+pot[i][0])//2),int((pot[i][3]+pot[i][2])//2))
        #掩膜
        mask = 255*np.ones(piece.shape,piece.dtype)
        #cv2.NORMAL_CLONE  cv2.MIXED_CLONE
        pic = cv2.seamlessClone(piece,pic,mask,center,cv2.NORMAL_CLONE)
        
        #pic[int(pot[i][2]):int(pot[i][3]),int(pot[i][0]):int(pot[i][1])] = piece
        #pppp = Image.fromarray(np.uint8(pic))
        #pppp.save(f'ans.jpg')
    print('finished')
    #-----删除缓存图片------#
    paths = glob.glob('data/RaFD/test/happy/*.jpg')
    for f in paths:
        os.remove(f)
    return pic
           
        
#np.uint8(img)
#img= Image.fromarray(np.uint8(img)).convert('RGB')
        
        
        
        
        
    