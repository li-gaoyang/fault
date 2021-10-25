import cv2 as cv
import numpy as np
import math

def show_wait_destroy(winname,img):
    cv.imshow(winname,img)
    cv.waitKey(0)



def Cal_Direction_Field(img):
    matTst=img.copy()
    matX=cv.Sobel(matTst,cv.CV_16SC1, 1, 0).astype(np.float)
    matY=cv.Sobel(matTst,cv.CV_16SC1, 0, 1).astype(np.float)
 

    Gxx=cv.multiply(matX,matX)
    Gyy=cv.multiply(matY,matY)
    Gxy=cv.multiply(matX,matY)

    kernel8=np.ones((20,20)).astype(np.float)
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
    
    return matCoh

#img_path='img01.jpg'
img_path='img02.jpg'
#img_path='img03.jpg'
old_img=cv.imread(img_path)       



old_img=cv.cvtColor(old_img,cv.COLOR_RGB2GRAY)

 


matCoh=Cal_Direction_Field(old_img)#计算方向场一致性

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


#先开操作，再闭操作，卷积核设定尽可能的保留纵向特征，剔除横向特征
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
def caldist(flag_i,flag_j,i,j):
    return int(math.sqrt((flag_i-i)*(flag_i-i)+(flag_j-j)*(flag_j-j)))

def cal_con_dist(con1,con2):
    res1=[9999,9999,9999]
    for i in con1:
        for j in con2:
            dt=caldist(i[0][0],i[0][1],j[0][0],j[0][1])
            if dt<res1[2]:
                res1=[(i[0]),(j[0]),dt]
    if res1[2]>5:
        return 0
    return res1
contours_mid, _ = cv.findContours(mid_points, cv.RETR_EXTERNAL,cv.RETR_LIST)
mid_points2=mid_points.copy()
 
for i in range(0,len(contours_mid)):
    begin_con=contours_mid[i]
    
    for j in range(0,len(contours_mid)):
        if i!=j:
            end_con=contours_mid[j]
            p12=cal_con_dist(begin_con,end_con)
            if p12!=0:
                cv.line(mid_points2,(p12[0][0],p12[0][1]),(p12[1][0],p12[1][1]),(255),1)

result_img=np.zeros(mid_points2.shape,dtype="uint8")  
contours_mid2, _ = cv.findContours(mid_points2, cv.RETR_EXTERNAL,cv.RETR_LIST)
cv.imwrite("mid_points_con.png",mid_points2)


#剔除小于40个像素的线段   
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

