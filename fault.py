from ast import Return
import cv2 as cv
import numpy as np
import math


def show_wait_destroy(winname,img):
    cv.imshow(winname,img)
    cv.waitKey(0)

def caldist(flag_i,flag_j,i,j):
    return int(math.sqrt((flag_i-i)*(flag_i-i)+(flag_j-j)*(flag_j-j)))

def cal_con_dist(con1,con2,conval):
    res1=[9999,9999,9999]
    for i in con1:
        for j in con2:
            dt=caldist(i[0][0],i[0][1],j[0][0],j[0][1])
            if dt<res1[2]:
                res1=[(i[0]),(j[0]),dt]
    if res1[2]>conval:
        return 0
    return res1

#计算方向场一致性,输入numpy数组和卷积核kernel
def Cal_Direction_Field(img,kernel1,kernel2):
    matTst=img.copy()
    matX=cv.Sobel(matTst,cv.CV_16SC1, 1, 0).astype(np.float)
    matY=cv.Sobel(matTst,cv.CV_16SC1, 0, 1).astype(np.float)
 

    Gxx=cv.multiply(matX,matX)
    Gyy=cv.multiply(matY,matY)
    Gxy=cv.multiply(matX,matY)

    kernel8=np.ones((kernel1,kernel2)).astype(np.float)
    Gxx=cv.filter2D(Gxx,-1,kernel8)
    Gyy=cv.filter2D(Gyy,-1,kernel8)
    Gxy=cv.filter2D(Gxy,-1,kernel8)
    MatTmp = 2 * Gxy
    matTmp = MatTmp / (Gxx - Gxy)
    matTheta=np.zeros(matTst.shape)
    
    for i in range(0,matTmp.shape[0]):
        for j in range(0,matTmp.shape[1]):
            matTheta[i][j] = 0.5 * math.atan(matTmp[i][j])
    matTmp = Gxx - Gyy
    matTmp=cv.multiply(matTmp,matTmp)
    matTmp2 = 4 *cv.multiply(Gxy,Gxy)
    matTmp += matTmp2
    matCoh=cv.sqrt(matTmp)
    matCoh = matCoh / (Gxx + Gyy)
    matCoh2=matCoh.copy()
    cv.normalize(matCoh,matCoh2,0,255,cv.NORM_MINMAX)#归一化
    cv.imwrite("Direction_Field.png",matCoh2)

    return matCoh2

#二值化
def binfun(matCoh):
    matCoh=matCoh.astype(np.uint16)

    # ret,thresh1=cv.threshold(matCoh,_threshold,255,cv.THRESH_BINARY_INV)#二值化
    ret,thresh1=cv.threshold(matCoh,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)#二值化
    cv.imwrite("Direction_Field_binary.png",thresh1)
    return thresh1,ret

#闭操作
def closed(thresh1,ker1,ker2):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,1))
    closeding = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    cv.imwrite("Direction_Field_binary_closed.png",closeding) 
    return closeding

#提取中心点
def extmidpoints(cl_img):
    for i in range(0,cl_img.shape[1]):
        for j in range(0,cl_img.shape[0]):
            if i==cl_img.shape[1]-1:
                cl_img[j][i]=0

    #取有效区域的中心点
    mid_points=np.zeros(cl_img.shape,dtype="uint8") 
    for i in range(0,cl_img.shape[0]):
        flag=0
        b=-1
        e=-1
        for j in range(0,cl_img.shape[1]):
            if flag==0 and cl_img[i][j]==255:
                flag=1
                b=j
            if flag==1 and cl_img[i][j]==0:
                e=j
                flag=0
            if b!=-1 and e!=-1:
                mid_points[i][int((e+b)/2)]=255
                b=-1
                e=-1
    mid_points=removeshortlines2(mid_points,10)
    cv.imwrite("mid_points.png",mid_points)
    return mid_points

def calangle(x1,y1,x2,y2):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    z = math.sqrt(x * x + y * y)
    angle = round(math.asin(y / z) / math.pi * 180)
    return abs(angle)

#剔除小于40个像素的线段   
def removeshortlines2(mid_points2,minremove):

    result_img=np.zeros(mid_points2.shape,dtype="uint8")  
    contours_mid2, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)
    for i in contours_mid2:
        if len(i)>minremove:
                result_img = cv.drawContours(result_img, i, -1, (255), 1)
    cv.imwrite("result_img.png",result_img)
    return result_img
    



#连接相邻线段
def connectlines(mid_points,minconval):
   
    mid_points2=mid_points.copy()
    
    flag1=0
    while True:
        flag2=0
        contours_mid, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)#找所有1线
        flag1=flag1+1
        if flag1>len(contours_mid):
            break
        print()
        for i in range(0,len(contours_mid)):#遍历所有线
            begin_con=contours_mid[i]
            print()
            if flag2==1:
                    break
            for j in range(0,len(contours_mid)):
                if flag2==1:
                    break
                if i!=j:
                    end_con=contours_mid[j]
                    p12=cal_con_dist(begin_con,end_con,minconval)
                   
                    if p12!=0:
                        angle2=calangle(p12[0][0],p12[0][1],p12[1][0],p12[1][1])    
                        if angle2>60 and angle2<120:
                            print(p12)
                            cv.line(mid_points2,(p12[0][0],p12[0][1]),(p12[1][0],p12[1][1]),(255),1)
                            flag2=1


    cv.imwrite("mid_points_con.png",mid_points2)

    return mid_points2

#剔除小于40个像素的线段   
def removeshortlines(mid_points2,minremove):

    result_img=np.zeros(mid_points2.shape,dtype="uint8")  
    contours_mid2, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)
    for i in contours_mid2:
        if len(i)>minremove:
                result_img = cv.drawContours(result_img, i, -1, (255), 1)
    cv.imwrite("result_img.png",result_img)
    return result_img
    

#颜色反转
def getresultimg(mid_points2):
    
    result_img=np.zeros(mid_points2.shape,dtype="uint8")  
    contours_mid2, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)
    for i in contours_mid2:
      
        result_img = cv.drawContours(result_img, i, -1, (255), 2)
    res=result_img.copy()
    w,h=result_img.shape
    for i in range(w):
        for j in range(h):
            if res[i][j]==255:
                res[i][j]=0
            else:
                res[i][j]=255 
            
    cv.imwrite("result.png",res)
    return res












def faultdetcion(img_path):

    
    old_img=cv.imread(img_path)       



    old_img=cv.cvtColor(old_img,cv.COLOR_RGB2GRAY)



    matCoh=Cal_Direction_Field(old_img,50,50)#计算方向场一致性

    matCoh2=matCoh.copy()
    cv.normalize(matCoh,matCoh2,0,255,cv.NORM_MINMAX)#归一化
    cv.imwrite("Direction_Field.png",matCoh2)

    if img_path=='img01.jpg' or img_path=='img03.jpg':
        ret,thresh1=cv.threshold(matCoh2,199,255,cv.THRESH_BINARY_INV)#二值化
    elif img_path=='img02.jpg':
        ret,thresh1=cv.threshold(matCoh2,199,255,cv.THRESH_BINARY_INV)#二值化
    else:
        ret,thresh1=cv.threshold(matCoh2,199,255,cv.THRESH_BINARY_INV)#二值化


    cv.imwrite("Direction_Field_binary.png",thresh1)


    #闭操作，卷积核设定尽可能的保留纵向特征，剔除横向特征
    c= int(matCoh2.shape[1]/30)
    r= int(c/2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(r,c)) 
    closeding = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    cv.imwrite("Direction_Field_binary_closed.png",closeding) 
    cl_img=cv.imread("Direction_Field_binary_closed.png",flags=0)
    # closeding=closeding.astype(cv.CV_32SC1)




    #去除小面积/孔洞填充（二值图像）
    contours, _ = cv.findContours(cl_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= 800:
            cv_contours.append(contour)
        else:
            continue
    cv.fillPoly(cl_img, cv_contours, (255))
    cv.imwrite("Direction_Field_binary_closed_dete_isoland.png",cl_img)

    for i in range(0,cl_img.shape[1]):
        for j in range(0,cl_img.shape[0]):
            if i==cl_img.shape[1]-1:
                cl_img[j][i]=0

    #取有效区域的中心点
    mid_points=np.zeros(cl_img.shape,dtype="uint8") 
    for i in range(0,cl_img.shape[0]):
        flag=0
        b=-1
        e=-1
        for j in range(0,cl_img.shape[1]):
            if flag==0 and cl_img[i][j]==255:
                flag=1
                b=j
            if flag==1 and cl_img[i][j]==0:
                e=j
                flag=0
            if b!=-1 and e!=-1:
                mid_points[i][int((e+b)/2)]=255
                b=-1
                e=-1

    cv.imwrite("mid_points.png",mid_points)



    #链接相邻的点集
   
    contours_mid, _ = cv.findContours(mid_points, cv.RETR_EXTERNAL,cv.RETR_LIST)
    mid_points2=mid_points.copy()
    
    for i in range(0,len(contours_mid)):
        begin_con=contours_mid[i]
        
        for j in range(0,len(contours_mid)):
            if i!=j:
                end_con=contours_mid[j]
                p12=cal_con_dist(begin_con,end_con,5)
                if p12!=0:
                    cv.line(mid_points2,(p12[0][0],p12[0][1]),(p12[1][0],p12[1][1]),(255),1)

    cv.imwrite("mid_points_con.png",mid_points2)


    #剔除小于40个像素的线段   
    result_img=np.zeros(mid_points2.shape,dtype="uint8")  
    contours_mid2, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)
    for i in contours_mid2:

        if img_path=='img01.jpg' or img_path=='img03.jpg':
            if len(i)>40:
                result_img = cv.drawContours(result_img, i, -1, (255), 1)
        elif img_path=='img02.jpg':
            if len(i)>80:
                result_img = cv.drawContours(result_img, i, -1, (255), 1)
        else:
            if len(i)>40:
                result_img = cv.drawContours(result_img, i, -1, (255), 1)
    
    cv.imwrite("result_img.png",result_img)



    fig10_c=result_img.copy()
    for i in range(0,result_img.shape[0]):
        for j in range(0,result_img.shape[1]):
            if result_img[i][j]==255:
                fig10_c[i][j]=0
            else:
                fig10_c[i][j]=255

    cv.imwrite("fig10-c.png",fig10_c)





    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    faultdetcion("img01.jpg")