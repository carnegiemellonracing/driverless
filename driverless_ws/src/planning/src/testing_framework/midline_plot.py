import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Point
from scipy.interpolate import interp1d



def plot_lin_input(blue_cone_pos, yellow_cone_pos, midline,plotnum):
    bx = []
    by = []
    
    yx = []
    yy = []
    
    mx = []
    my = []
    
    for i in range(len(blue_cone_pos)):
        
        bx.append(blue_cone_pos[i.x])
        by.append(blue_cone_pos[i.y])
    
    for j in range(len(yellow_cone_pos)):
        yx.append(yellow_cone_pos[j.x])
        yy.append(yellow_cone_pos[j.y])
        
    for k in range(len(midline)):
        mx.append(midline[k.x])
        my.append(midline[k.y])
        
    mx_interp = np.linspace(np.min(mx), np.max(mx), 100)
    my_linear = interp1d(mx, my)


    plt.plot(bx, by, 'b*', yx, yy, 'y*', mx, my, 'r*', mx_interp, my_linear(mx_interp), "black")
    # plt.show()
    file_name = f'midline_visualization_linear_{plotnum}'
    plt.savefig(file_name)

def plot_cub_input(blue_cone_pos, yellow_cone_pos, midline,plotnum):
    bx = []
    by = []
    
    yx = []
    yy = []
    
    mx = []
    my = []
    
    for i in range(len(blue_cone_pos)):
        bx.append(blue_cone_pos[i.x])
        by.append(blue_cone_pos[i.y])
    
    for j in range(len(yellow_cone_pos)):
        yx.append(yellow_cone_pos[j.x])
        yy.append(yellow_cone_pos[j.y])
        
    for k in range(len(midline)):
        mx.append(midline[k][0])
        my.append(midline[k][1])
        
    mx_interp = np.linspace(np.min(mx), np.max(mx), 100)
    my_cubic = interp1d(mx, my, kind="cubic")
    
    

    plt.plot(bx, by, 'b*', yx, yy, 'y*', mx, my, 'r*', mx_interp, my_cubic(mx_interp), "black")
    # plt.show()
    file_name = f'midline_visualization_quadratic_{plotnum}'
    plt.savefig(file_name)