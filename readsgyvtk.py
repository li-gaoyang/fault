import numpy as np
import cv2 as cv
import vtk


def load_data_from_sgy(sgy_path):
    print(sgy_path)
    segYReader = vtk.vtkSegYReader()
    segYReader.SetFileName(sgy_path)
    segyImage = segYReader.GetImageDataOutput()
    segyGrid = segYReader.GetStructuredGridOutput()
    points = segYReader.GetStructuredPointsOutput()
   
    segYReader.Update()
    dim=segyGrid.GetDimensions()
    print(dim)

    writesgrid = vtk.vtkStructuredGridWriter()
    writesgrid.SetFileName("sgrid.vtk")
    writesgrid.SetInputData(segyGrid)
    writesgrid.Write()
    scalar_range = segyGrid.GetScalarRange()
    dims=segyGrid.GetDimensions()
    arr=segyGrid.GetPointData().GetArray(0)
    #img=np.ones([1168,923])
    img=np.ones([dim[2],dim[0]])
    for i in range(dim[2]):
        for j in range(dim[0]):
            img[i][j]=arr.GetTuple(i*dim[0]+j)[0]
    
    return img
   
    

if __name__ == '__main__':

    sesmicdt=load_data_from_sgy("demo01.sgy")
    cv.imwrite('sesmicdt.jpg', sesmicdt)    