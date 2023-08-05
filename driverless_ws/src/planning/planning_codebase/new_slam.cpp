/*
* Extended Kalman Filter SLAM example
* author: Atsushi Sakai (@Atsushi_twi)
* Translator: Rohan Raavi & Ankit Khandelwal 
*/
#include <iostream>
#include <cmath>
#include <gsl/gsl_block.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>



using namespace std;

struct ArrStruct {
    int rows;
    int cols;
    double** arr;
};

double deg2Rad(double ang) {
    double rad = (ang*M_PI)/180.0;
    return rad;
}



// EKF state covariance
double Cx [3][3] = { {(pow(0.5,2)),0,0}, {0,(pow(0.5,2)),0}, {0,0,(pow(deg2Rad(30.0),2))}};
double alphas[6] = {0.11, 0.01, 0.18, 0.08, 0.0, 0.0};



// Simulation parameter
double Q_sim [2][2] = { {(pow(0.2,2)),0}, {0, (pow(deg2Rad(1.0),2)) } };
double R_sim [2][2] = { {1,0}, {0, (pow(deg2Rad(10.0),2)) } };


//# DT = 0.1  # time tick [s]
double const SIM_TIME = 50.0; // # simulation time [s]
double const MAX_RANGE = 20.0;  //# maximum observation range
double const M_DIFF_TH = 1.6;
double const M_DIST_TH = 2;
// double const M_DIST_TH_FIRST = 0.25; // # Threshold of Mahalanobis distance for data association.
double const M_DIST_TH_ALL = 1;
double const STATE_SIZE = 3;  //# State size [x,y,yaw]
double const LM_SIZE = 2;  //# LM state size [x,y]

bool show_animation = true;









gsl_matrix* motion_model(gsl_matrix* x, gsl_matrix* u, double dt)
{
    gsl_matrix* F = gsl_matrix_calloc(3,3);
    gsl_matrix_set(F,0,0,1);
    gsl_matrix_set(F,1,1,1);
    gsl_matrix_set(F,2,2,1);
    


    gsl_matrix* B = gsl_matrix_calloc(3,2);
    double val1 = gsl_matrix_get(x,2,0);
    gsl_matrix_set(B,0,0,dt*cos(val1));
    gsl_matrix_set(B,1,0,dt*sin(val1));
    gsl_matrix_set(B,2,1,dt);
    
    gsl_matrix * arr1 = gsl_matrix_alloc(F->size1,x->size2);
    int i = gsl_linalg_matmult(F,x,arr1);
    gsl_matrix * arr2 = gsl_matrix_alloc(B->size1,u->size2);
    i = gsl_linalg_matmult(B,u,arr2);
    gsl_matrix_add(arr1,arr2);
    gsl_matrix_memcpy(x, arr1);
    
    gsl_matrix_free(F);
    gsl_matrix_free(B);
    gsl_matrix_free(arr1);
    gsl_matrix_free(arr2);
    return x;
}

int calc_n_lm(gsl_matrix* x) {
    int len;
    if(x->size1==1)
    {
        len = x->size2;
    }
    else {
        len = x->size1;
    }

    int n = ((len - STATE_SIZE)/LM_SIZE);
    return n;
}

double pi_2_pi(double angle){
    double ans = fmod((angle + M_PI), (2 * M_PI)) - M_PI;
    return ans;  
}
    

gsl_matrix* calc_landmark_position(gsl_matrix* x, gsl_matrix* z)
{
    gsl_matrix* zp = gsl_matrix_calloc(2,1);
    

    double val1 = gsl_matrix_get(x,0,0) + (gsl_matrix_get(z,0,0)*cos(gsl_matrix_get(x,2,0) + gsl_matrix_get(z,0,1)));
    gsl_matrix_set(zp,0,0,val1);
    double val2 = gsl_matrix_get(x,1,0) + (gsl_matrix_get(z,0,0)*sin(gsl_matrix_get(x,2,0) + gsl_matrix_get(z,0,1)));
    gsl_matrix_set(zp,1,0,val2);
    
    return zp;
}

gsl_matrix* jacob_h(double q, gsl_matrix* delta, gsl_matrix* x, double i)
{
    double sq = sqrt(q);
    gsl_matrix* G = gsl_matrix_calloc(2,5);
    gsl_matrix_set(G,0,0,-sq*gsl_matrix_get(delta,0,0));
    gsl_matrix_set(G,0,1,-sq*gsl_matrix_get(delta,1,0));
    gsl_matrix_set(G,0,3,sq*gsl_matrix_get(delta,0,0));
    gsl_matrix_set(G,0,4,sq*gsl_matrix_get(delta,1,0));

    gsl_matrix_set(G,1,0,gsl_matrix_get(delta,1,0));
    gsl_matrix_set(G,1,1,-gsl_matrix_get(delta,0,0));
    gsl_matrix_set(G,1,2,-q);
    gsl_matrix_set(G,1,3,-gsl_matrix_get(delta,1,0));\
    gsl_matrix_set(G,1,4,gsl_matrix_get(delta,0,0));

    gsl_matrix_scale(G,(1/q));
    int nLM = calc_n_lm(x);

    gsl_matrix* F1 = gsl_matrix_calloc(3,3 + (2*nLM));
    gsl_matrix_set(F1,0,0,1);
    gsl_matrix_set(F1,1,1,1);
    gsl_matrix_set(F1,2,2,1);

    gsl_matrix* F2 = gsl_matrix_calloc(2,3 + (2*(i-1)) + 2 + (2*nLM) - 2*i);
    gsl_matrix_set(F1,0,(3+(2*(i-1))),1);
    gsl_matrix_set(F1,1,(4+(2*(i-1))),1);

    gsl_matrix* F = gsl_matrix_calloc(5,F1->size2);
    for(int r = 0; r < 5; r++)
    {
        for(int c = 0; c < F1->size2; c++)
        {
            if(r<=2)
            {
                gsl_matrix_set(F,r,c,gsl_matrix_get(F1,r,c));
            }
            else
            {
                gsl_matrix_set(F,r,c,gsl_matrix_get(F2,r-3,c));
            }
        }
    }

    gsl_matrix_free(F1);
    gsl_matrix_free(F2);

    gsl_matrix* H = gsl_matrix_calloc(G->size1,F->size2);
    gsl_linalg_matmult(G,F,H);
    gsl_matrix_free(G);
    gsl_matrix_free(F);

    return H;
    
}

gsl_matrix** jacob_motion(gsl_matrix* x, gsl_matrix* u, double dt){ //do we need logger?
    gsl_matrix** res = new gsl_matrix*[2];
    gsl_matrix* Fx = gsl_matrix_calloc(STATE_SIZE,STATE_SIZE + LM_SIZE * calc_n_lm(x));
    for (int i = 0; i < STATE_SIZE; i++)
    {
        gsl_matrix_set(Fx,i,i,1);
    }
    gsl_matrix* FxT = gsl_matrix_calloc(STATE_SIZE + LM_SIZE * calc_n_lm(x), STATE_SIZE);

    gsl_matrix_transpose_memcpy(FxT,Fx);

    gsl_matrix* jf = gsl_matrix_calloc(3,3); //does jf have to be a float 2d array? is it ok if it's a double
    
    double val1 = gsl_matrix_get(u,0,0);
    gsl_matrix_set(jf,0,2,-dt*val1*sin(gsl_matrix_get(x,2,0)));
    gsl_matrix_set(jf,0,2,-dt*val1*cos(gsl_matrix_get(x,2,0)));

    gsl_matrix* arr1 = gsl_matrix_calloc(FxT->size1,jf->size2);
    gsl_linalg_matmult(FxT,jf,arr1);

    gsl_matrix* arr2 = gsl_matrix_calloc(arr1->size1,Fx->size2);
    gsl_linalg_matmult(arr1,Fx,arr2);

    gsl_matrix* G = gsl_matrix_calloc(STATE_SIZE,STATE_SIZE);
    for (int i = 0; i < STATE_SIZE; i++)
    {
        gsl_matrix_set(G,i,i,1);
    }

    gsl_matrix_add(G,+arr2);

    res[0] = G;
    res[1] = Fx;

    gsl_matrix_free(FxT);
    gsl_matrix_free(arr1);
    gsl_matrix_free(arr2);
    gsl_matrix_free(jf);
    return res;

}


gsl_matrix** calc_innovation(gsl_matrix* lm, gsl_matrix* xEst, gsl_matrix* PEst, gsl_matrix* z, double LMid) {
    gsl_matrix* delta = gsl_matrix_calloc(lm->size1,lm->size2);
    gsl_matrix* xEst2 = gsl_matrix_calloc(lm->size1,lm->size2);
    for(int r = 0; r < 2; r++) {
        for(int c = 0; c < lm->size2; c++)
        {
            gsl_matrix_set(xEst2,r,c,gsl_matrix_get(xEst,r,c));
        }
    }
    gsl_matrix_memcpy(delta,lm);
    gsl_matrix_sub(delta,xEst2);
    
    gsl_matrix* deltaT = gsl_matrix_calloc(delta->size2,delta->size1);
    gsl_matrix_transpose_memcpy(deltaT,delta);

    gsl_matrix* dtd = gsl_matrix_calloc(deltaT->size1,delta->size2);
    gsl_linalg_matmult(deltaT,delta,dtd);
    double q = gsl_matrix_get(dtd,0,0);
    double z_angle = atan(gsl_matrix_get(delta,1,0)/gsl_matrix_get(delta,0,0)) -  gsl_matrix_get(xEst,2,0);

    gsl_matrix* y = gsl_matrix_calloc(2,1);
    gsl_matrix_set(y,0,0, gsl_matrix_get(z,0,0) - sqrt(q));
    gsl_matrix_set(y,1,0, gsl_matrix_get(z,0,1) - pi_2_pi(z_angle));
    gsl_matrix_set(y,1,0,pi_2_pi(gsl_matrix_get(y,1,0)));

    gsl_matrix* H = jacob_h(q,delta,xEst,LMid+1);
    gsl_matrix* HT = gsl_matrix_calloc(H->size2,H->size1);
    gsl_matrix_transpose_memcpy(HT,H);
    gsl_matrix* arr1 = gsl_matrix_calloc(H->size1,PEst->size2);
    gsl_linalg_matmult(H,PEst,arr1);
    gsl_matrix* S = gsl_matrix_calloc(arr1->size1,HT->size2);
    gsl_linalg_matmult(arr1,HT,S);

    gsl_matrix* Cx2x2 = gsl_matrix_calloc(2,2);
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            gsl_matrix_set(Cx2x2,i,j,Cx[i][j]);
        }
    }
    gsl_matrix_add(S,Cx2x2);

    gsl_matrix** res = new gsl_matrix*[3];
    res[0] = y;
    res[1] = S;
    res[2] = H;
    
    gsl_matrix_free(delta);
    gsl_matrix_free(deltaT);
    gsl_matrix_free(xEst2);
    gsl_matrix_free(dtd);
    gsl_matrix_free(HT);
    gsl_matrix_free(arr1);
    return res; 

}


void print2DArray(double** arr, int rows, int cols)
{
    
    cout << "rows = " << rows << " " << "cols = " << cols << endl;
    for(int a = 0; a < rows; a++)
    {
        for(int b = 0; b < cols; b++)
        {
            cout << arr[a][b] << " ";
        }
        cout << endl;
    } 
}

void printGSLMatrix(gsl_matrix *mat)
{
    int rows = mat->size1;
    int cols = mat->size2;
    for(int a = 0; a < rows; a++)
    {
        for(int b = 0; b < cols; b++)
        {
            cout << gsl_matrix_get(mat,a,b) << " ";
        }
        cout << endl;
    } 
}

gsl_matrix* calc_input(){
    double v = 1.0; //# [m/s]
    double yaw_rate = 0.1; //  # [rad/s]

    gsl_matrix* u = gsl_matrix_calloc(2,1);
    gsl_matrix_set(u,0,0,v);
    gsl_matrix_set(u,1,0,yaw_rate);

    
    return u;
}










int main() {
    
    gsl_matrix * r = gsl_matrix_calloc(3,3);
    gsl_matrix_set_all(r,2);
    printGSLMatrix(r);

    gsl_matrix *g = gsl_matrix_calloc(3,1);
    gsl_matrix_set_all(g,1);
    printGSLMatrix(g);

    gsl_matrix* ans = gsl_matrix_calloc(r->size1,g->size2);

    gsl_linalg_matmult(r,g,ans);
    gsl_matrix_free(r);
    gsl_matrix_free(g);
    printGSLMatrix(ans);
    
    gsl_matrix_free(ans);
    
    


    cout << "Hello World" << endl;
    return 0;
}