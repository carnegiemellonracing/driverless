import numpy as np


left_points = np.array([])  #np.array([[-1,1],[-1,2],[-1,3]])
right_points = np.array([[1.,1.],[1.,2.],[2.,3.]])
 
print("left",left_points)
print("right",right_points)

#interpolate points if we only see one of a single color
if(len(left_points) == 0 and len(right_points)> 0 ):
    #interpolate the other side
    for i in range(len(right_points)):
        if(i == 0):
            #just take first cone and flip x axis
            temp = np.copy(right_points[0])
            temp[0] = -1*temp[0] #flip axis
            left_points = np.append(left_points,temp)
            left_points = np.reshape(left_points,(1,2))
            # print("after first",left_points)
        else: #take linear transform
            difference = [right_points[i][0]-right_points[i-1][0],right_points[i][1]-right_points[i-1][1]]
            difference = np.array(difference,dtype=float)
            # print("difference",difference)
            # print("left_points[i-1]",left_points[i-1])
            newCone = left_points[i-1]+difference
            # print("newCone",newCone)
            left_points = np.vstack((left_points,newCone))

if(len(left_points) > 0 and len(right_points)== 0 ):
    for i in range(len(left_points)):
        if(i == 0):
            temp = np.copy(left_points[0])
            temp[0] = -1*temp[0] #flip axis
            right_points = np.append(right_points,temp)
            right_points = np.reshape(right_points,(1,2))
        else: #take linear transform
            difference = [left_points[i][0]-left_points[i-1][0],left_points[i][1]-left_points[i-1][1]]
            difference = np.array(difference,dtype=float)
            newCone = right_points[i-1]+difference
            right_points = np.vstack((right_points,newCone))

print("left",left_points)
print("right",right_points)