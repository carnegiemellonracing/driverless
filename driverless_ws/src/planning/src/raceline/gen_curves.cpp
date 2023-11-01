#include <gsl/gsl_math.h>
#include <math.h>
#include <map>
#include "raceline.hpp"

#define STEP_THRESH = 10
#define STEP_SPECIAL = 2
#define STEP_NORMAL = 10
#define DEG_SPECIAL = 2
#define DEG_NORMAL = 3 
#define OUTLIER_PERCENT = 0.95



std::pair<double,double> calcMidPoint(double x1,double y1,double x2,double y2){
    return std::make_pair((x1+x2)/2,(y1+y2)/2);
}

double distance(double x1,double y1,double x2,double y2){
    return sqrt(gsl_pow_2(x2-x1)+gsl_pow_2(y2-y1));
}

double calcDist(std::vector<double> x,std::vector<double> y,int i, int rangeLen){
    double x1,y1 = x[i],y[i];
    double x2,y2 = x[i+rangeLen],y[i+rangeLen];
    std::pair<double,double> mxy = calcMidPoint(x1,y1,x2,y2);
    double x_,y_ = x[i+rangeLen/2], y[i+rangeLen/2];
    return distance(mxy.first,mxy.second,x_,y_);
}

double consecutiveGreaterThanThresh(std::vector<double> x,std::vector<double> y,int i,double thresh,int rangeLen){
    double thisD = calcDist(x,y,i+1,rangeLen);
    while (thisD >= thresh){
        i+=1;
        thisD = calcDist(x,y,i,rangeLen);
    }
    if (i+rangeLen < x.size())
        return i+rangeLen ;
    return x.size();
}


// WILL HAVE TO FIX TYPE FOR R
int idxInRange(int start, int stop, std::map<int,int> R){
    for (int i=start;i< stop;i++)
        if (R.count(i) ==0)
            return (i);
    return -1;
}

// NEED TO ADD DRAW