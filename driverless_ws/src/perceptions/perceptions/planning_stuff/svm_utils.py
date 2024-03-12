
from sklearn import svm
import numpy as np

def process(data):

    X = data[:, :2]
    y = data[:, -1]
    model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
    model.fit(X, y)

    step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    boundary_points = []
    for i in range(1, len(xx)):
        for j in range(1, len(xx[0])):
            if Z[i][j] != Z[i-1][j-1]:
                boundary_points.append([xx[i][j], yy[i][j]])

    def norm_func(x): return np.sqrt(x[0]**4 + x[1]**2)
    boundary_points.sort(key=norm_func)
    # print(boundary_points)

    downsampled = []
    accumulated_dist = 0
    for i in range(1, len(boundary_points)):
        p1 = boundary_points[i]
        p0 = boundary_points[i-1]
        curr_dist = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        accumulated_dist += curr_dist
        if np.abs(accumulated_dist - 0.5) < 0.1: # TODO: make this 50cm
            downsampled.append(p1)
            accumulated_dist = 0
        
        if accumulated_dist > 0.55:
            accumulated_dist = 0
    
    downsampled = np.array(list(downsampled))
    # print(downsampled)

    return downsampled