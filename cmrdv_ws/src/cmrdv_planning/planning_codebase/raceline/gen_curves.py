import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

##################################################
# Helper Functions
##################################################

# Midpoint between 2 points
def calcMidPoint(x1,y1,x2,y2):
    x = (x1 + x2)/2
    y = (y1 + y2)/2
    return x, y

# Distance between 2 points
def distance(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Distance D between midpoint of (x1,y1),(x3,y3)
# and point (x2,y2)
def calcDist(x,y,i,rangeLen):
    x1,y1 = x[i],y[i]
    x2,y2 = x[i+rangeLen],y[i+rangeLen]
    mx,my = calcMidPoint(x1,y1,x2,y2)
    # between x1,y1 and x2,y2
    x_,y_ = x[math.floor(i+rangeLen/2)], y[math.floor(i+rangeLen/2)]
    return distance(mx,my,x_,y_)

# Returns end of range of last consecutive point after i
# that exceeds threshold thresh
def consecutiveGreaterThanThresh(x,y,i,thresh,rangeLen):
    thisD = calcDist(x,y,i+1,rangeLen)
    while thisD >= thresh:
        i+=1
        thisD = calcDist(x,y,i,rangeLen)

    if (i+rangeLen < x.shape[0]):
        return i+rangeLen 
    return x.shape[0]-1 

# If point between [start, stop] is in 
# R, return that point. Else, return None
def idxInRange(start, stop, R):
    for i in range(start, stop+1):
        if (R.get(i) != None):
            return (i) 
    return None

# Both plots curve and stores it in list
def draw(start, stop, x, y, deg, store):
    segx, segy = x[start:stop], y[start:stop]
    coeff = np.polyfit(segx, segy, deg)
    curve = np.poly1d(coeff)
    X = np.linspace(x[start],x[stop])
    Y = curve(X)
    plt.plot(X,Y)
    # is this what we want to store?
    store.append(curve)

# Temporary special treatment for severe curves
def severeCurve(start, stop, step, x, y, deg, store):

    for i in range(start, stop, step):
        stop_ = i + step
        if (i + step >= stop):
            stop_ = stop

        draw(i, stop_, x, y, deg, store)

    return None

################################################## 
# GLOBALS
################################################## 

# step between (x1,y1) and (x3,y3) (see algo for severe curvatures) 
STEP_THRESH = 10

# how many points to gen polynomial for severe curves
STEP_SPECIAL = 2

# how many points to gen polynomial for normal curves
STEP_NORMAL = 10

# polynomial degree for severe curvature generation
DEG_SPECIAL = 2

# polynomial degree for normal curvature generation
DEG_NORMAL = 3 

# the OUTLIER_PERCENT percentile highest distance D is used as threshold value
OUTLIER_PERCENT = 0.95

def main():
   
    ################################################## 
    # OVERVIEW
    ################################################## 
    

    ################################################## 
    # IDEA: 
    # - Generate polynomial curves to fit given points
    # - Split points into segments of ranges, and generate curves
    # - Two classes of ranges: normal curves and severe curves
    
    # Currently, finds all ranges that cover severe curvatures.
    # But, need to develop good special treatment for those ranges - smooth curve generation
    
    # to blank out severe curves and see where they are, comment out severeCurve() below. 
    ################################################## 
    
    ################################################## 
    # ALGO FOR FINDING SEVERE CURVATURES:
    
    # A) draw straight line between (x1,y1), (x3,y3), and get its midpoint M

    # B) get (x2,y2), which has the same number of points before it and (x1,y1) and same number
    # of points after it and (x3, y3). 

    # C) calculate the distance between M and (x2,y2).

    # D) 
    # if this distance > some threshold, between x1,y1 and x3,y3 the points form large curvature
    # save points in range [(x1,y1), (x3,y3)] to do special treatment on them
    ################################################## 

    
    # load csv
    x = genfromtxt('xMCP.csv', delimiter=',')
    y = genfromtxt('yMCP.csv', delimiter=',')
    
    print('*'*50)
    print(f'X shape: {x.shape}, Y shape: {y.shape}')
    print('\n')


    plt.rcParams["figure.figsize"] = (15, 10)
       
    # calc all distances D betweeen (x2,y2) and midpoint M
    dists = []
    for i in range(x.shape[0]-STEP_THRESH):
        D = calcDist(x,y,i,STEP_THRESH) 
        dists.append(D)
   
    # do some stats

    # you can tweak outlier thresh,
    # currently, it only flags ranges with D of 95th percentile 

    outlierThresh = OUTLIER_PERCENT 
    dists.sort()
    threshIdx = math.floor(len(dists) * outlierThresh)
    thresh = dists[threshIdx]
    
    print('*'*50)
    print('choosing thresh...')
    print('average', sum(dists)/len(dists))
    print('median',dists[len(dists)//2])
    print(f'thresh at {int(OUTLIER_PERCENT * 100)}th percentile:', thresh)
    print('\n')
    

    # save all severe curvatures     
    ranges = dict() 
    inRange = False
    rangeEnd = float('inf') 

    for i in range(x.shape[0]-STEP_THRESH):
        # if we are out of already added range, allow
        # adding new ranges again
        if (inRange and i == rangeEnd):
            inRange = False

        D = calcDist(x,y,i,STEP_THRESH) 
        
        # add to ranges [first ith point greater than thresh, last ith point greater than thresh + rangeLen] 
        if (D >= thresh and not inRange): 
            inRange = True
            rangeEnd = consecutiveGreaterThanThresh(x,y,i,thresh,STEP_THRESH)

            Start = i 
            End = rangeEnd
            ranges[Start] = End
    
    # check ranges
    print('*' * 50)
    print('severe curvatures detected so far', ranges)
    print('\n')
    
    # store all Curves for controls 
    store = []
    
    print('*'*50)
    print('generating curvatures...')
    # generate all curvatures, and do special treatment to generate severe curvatures
    inRangeIdx = float('-inf') 
    nextStart = 0
    for i in range(x.shape[0]):

        # severe curvatures treatment 
        if (ranges.get(i) != None):
            inRangeIdx = ranges.get(i)
            
            # NEED BETTER TREATMENT OPTIONS!!!

            # option A: larger seg, larger deg
            #draw(i, inRangeIdx+1, x, y, 4, store)

            # option B: smaller seg, smaller deg 
            severeCurve(i, inRangeIdx, STEP_SPECIAL, x, y, DEG_SPECIAL, store)
            nextStart = inRangeIdx

            print(f'drawing seg [{i}, {inRangeIdx}]') 
        # all other curvatures
        elif (i > inRangeIdx and i >= nextStart):
            nextStart += STEP_NORMAL
            start = i
            stop = i+STEP_NORMAL
            n = idxInRange(i, i+STEP_NORMAL, ranges)
            if (i + STEP_NORMAL >= x.shape[0]):
                stop = x.shape[0]-1 
            elif (n != None):
                stop = n 
            
            print(f'drawing seg [{start}, {stop}]') 
            draw(start, stop, x, y, DEG_NORMAL, store)


    plt.show()

if '__main__' == __name__:
    main()
