/*
* Extended Kalman Filter SLAM example
* author: Atsushi Sakai (@Atsushi_twi)
* Translator: Rohan Raavi & Ankit Khandelwal
* Last Edit: 08/06/2023 
*/
#include <iostream>
#include <cmath>
#include <gsl/gsl_block.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <cstdlib>
#include <list>



using namespace std;

struct ArrStruct {
    int rows;
    int cols;
    double** arr;
};

//special struct I made so that the ekf_slam function can return xEst, PEst, and cones
struct ekfPackage {
    gsl_matrix* x;
    gsl_matrix* p;
    list<gsl_matrix*> cone;
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
    //3x3 identity matrix
    gsl_matrix* F = gsl_matrix_calloc(3,3);
    gsl_matrix_set(F,0,0,1);
    gsl_matrix_set(F,1,1,1);
    gsl_matrix_set(F,2,2,1);
    


    gsl_matrix* B = gsl_matrix_calloc(3,2);
    double val1 = gsl_matrix_get(x,2,0);
    gsl_matrix_set(B,0,0,dt*cos(val1));
    gsl_matrix_set(B,1,0,dt*sin(val1));
    gsl_matrix_set(B,2,1,dt);
    
    //x = F*x + B*u
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


   //vertical stacking F1 and F2 into F
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
    
    //H = G*F
    gsl_matrix* H = gsl_matrix_calloc(G->size1,F->size2);
    gsl_linalg_matmult(G,F,H);
    gsl_matrix_free(G);
    gsl_matrix_free(F);

    return H;
    
}

gsl_matrix** jacob_motion(gsl_matrix* x, gsl_matrix* u, double dt){ //do we need logger?
    gsl_matrix** res = new gsl_matrix*[2];

    //horizontal stacking STATE_SIZExSTATE_SIZE identity matrix with zeros matrix 
    gsl_matrix* Fx = gsl_matrix_calloc(STATE_SIZE,STATE_SIZE + LM_SIZE * calc_n_lm(x));
    for (int i = 0; i < STATE_SIZE; i++)
    {
        gsl_matrix_set(Fx,i,i,1);
    }


    //G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx
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
    //getting xEst[0:2]
    gsl_matrix* xEst2 = gsl_matrix_calloc(lm->size1,lm->size2);
    for(int r = 0; r < 2; r++) {
        for(int c = 0; c < lm->size2; c++)
        {
            gsl_matrix_set(xEst2,r,c,gsl_matrix_get(xEst,r,c));
        }
    }
    gsl_matrix_memcpy(delta,lm);
    //delta = lm - xEst[0:2]
    gsl_matrix_sub(delta,xEst2);
    
    gsl_matrix* deltaT = gsl_matrix_calloc(delta->size2,delta->size1);
    gsl_matrix_transpose_memcpy(deltaT,delta);
    
    //(delta.T * delta)
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
    

    //fetching Cx[0:2, 0:2]
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

gsl_matrix* get_landmark_position_from_state(gsl_matrix* x, int ind) {
    //translating this operation
    //lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    int start = STATE_SIZE + LM_SIZE*ind;
    int end = STATE_SIZE + LM_SIZE * (ind + 1);
    int size = end - start;
    gsl_matrix* lm = gsl_matrix_calloc(size,x->size2);
    for(int r = 0; r < size; r++)
    {
        for(int c = 0; c < x->size2; c++)
        {
            gsl_matrix_set(lm,r,c,gsl_matrix_get(x,r+start,c));
        }
    }
    return lm;
}

//helper function that finds the min value in a matrix and returns
// a double pointer containing the min value and its row and col
//location
double* findMinLocation(gsl_matrix* mat)
{
    double row = 0;
    double col = 0;
    double min = gsl_matrix_get(mat,0,0);

    for(int r = 0; r < mat->size1; r++)
    {
        for(int c = 0; c < mat->size2; c++)
        {
            double elem = gsl_matrix_get(mat,r,c);
            if(min > elem)
            {
                min = elem;
                row = r;
                col = c;
            }
        }
    }

    double* res = new double[3];
    res[0] = min;
    res[1] = row;
    res[2] = col;
    return res;
}

int search_correspond_landmark_id(gsl_matrix* xAug, gsl_matrix* PAug, gsl_matrix* zi)
{
    int nLM = calc_n_lm(xAug);
    double r = gsl_matrix_get(zi,0,0);
    double theta = gsl_matrix_get(zi,0,1);
    int start = 1;
    gsl_matrix* min_dist = gsl_matrix_calloc(1,1);
    for(int i = 0; i < nLM; i++)
    {
        gsl_matrix* lm = get_landmark_position_from_state(xAug,i);
        gsl_matrix** arrs = calc_innovation(lm, xAug, PAug, zi, i);
        gsl_matrix* y = arrs[0];
        gsl_matrix* S = arrs[1];
        gsl_matrix* H = arrs[2];
        
        gsl_matrix* yT = gsl_matrix_calloc(y->size2,y->size1);
        gsl_matrix_transpose_memcpy(yT,y);

        gsl_permutation* p = gsl_permutation_calloc(S->size1);
        int * signum = new int[1];
        signum[0] = 0;
        gsl_linalg_LU_decomp(S,p,signum);
        gsl_matrix* S_inverse = gsl_matrix_calloc(S->size1,S->size2);
        gsl_linalg_LU_invert(S,p,S_inverse);

        gsl_matrix* arr1 = gsl_matrix_calloc(yT->size1,S_inverse->size2);
        gsl_linalg_matmult(yT,S_inverse,arr1);

        gsl_matrix* arr2 = gsl_matrix_calloc(1,1);
        gsl_linalg_matmult(arr1,y,arr2);

        double val = gsl_matrix_get(arr2,0,0);
        
        //appending the array by remaking it
        if(start==1)
        {
            gsl_matrix_set(min_dist,0,0,val);
            start = 0;
        }
        else
        {
            gsl_matrix* new_min_dist = gsl_matrix_calloc(1,min_dist->size2+1);
            for(int j = 0; j < min_dist->size2; j++)
            {
                gsl_matrix_set(new_min_dist,0,j,gsl_matrix_get(min_dist,0,j));
            }
            gsl_matrix_set(new_min_dist,0,min_dist->size2,val);
            
            gsl_matrix_free(min_dist);
            min_dist = gsl_matrix_calloc(1,new_min_dist->size2);
        
            for(int j = 0; j < new_min_dist->size2; j++)
            {
                gsl_matrix_set(min_dist,0,j,gsl_matrix_get(new_min_dist,0,j));
            }

            gsl_matrix_free(new_min_dist);
        }
        gsl_matrix_free(arrs[0]);
        gsl_matrix_free(arrs[1]);
        gsl_matrix_free(arrs[2]);
        delete arrs;
        gsl_matrix_free(yT);
        gsl_matrix_free(arr1);
        gsl_matrix_free(arr2);
        gsl_matrix_free(S_inverse);
        gsl_permutation_free(p);
        delete signum;


    }
    //appending M_DIST_TH
    gsl_matrix* new_min_dist = gsl_matrix_calloc(1,min_dist->size2+1);
    for(int j = 0; j < min_dist->size2; j++)
    {
        gsl_matrix_set(new_min_dist,0,j,gsl_matrix_get(min_dist,0,j));
    }
    gsl_matrix_set(new_min_dist,0,min_dist->size2,M_DIST_TH);      
    gsl_matrix_free(min_dist);
    min_dist = gsl_matrix_calloc(1,new_min_dist->size2);
    for(int j = 0; j < new_min_dist->size2; j++)
    {
        gsl_matrix_set(min_dist,0,j,gsl_matrix_get(new_min_dist,0,j));
    }
    gsl_matrix_free(new_min_dist);

    double* minimum = findMinLocation(min_dist);

    int min_id = minimum[2]; //getting the col location of the min value
    delete minimum;
    gsl_matrix_free(min_dist);


    return min_id;
}

//
ekfPackage ekf_slam(gsl_matrix* xEst, gsl_matrix* PEst, gsl_matrix* u, gsl_matrix* z, double dt)
{
    double S = STATE_SIZE;
    gsl_matrix* x = gsl_matrix_calloc(S,xEst->size2);
    for(int r = 0; r < x->size1; r++)
    {
        for(int c = 0; c < x->size2; c++)
        {
            gsl_matrix_set(x,r,c,gsl_matrix_get(xEst,r,c));
        }
    }
    gsl_matrix** gFx = jacob_motion(x,u,dt);
    gsl_matrix* G = gFx[0];
    gsl_matrix* Fx = gFx[1];

    gsl_matrix* M_t = gsl_matrix_calloc(2,2);
    double val1 = pow((alphas[0]*abs(gsl_matrix_get(u,0,0)) + alphas[1]*abs(gsl_matrix_get(u,1,0))),2);
    double val2 = pow((alphas[2]*abs(gsl_matrix_get(u,0,0)) + alphas[3]*abs(gsl_matrix_get(u,1,0))),2);
    gsl_matrix_set(M_t,0,0,val1);
    gsl_matrix_set(M_t,1,1,val2);
    
    val1 = gsl_matrix_get(x,2,0);
    gsl_matrix* V_t = gsl_matrix_calloc(3,2);
    gsl_matrix_set(V_t,0,0,cos(val1));
    gsl_matrix_set(V_t,0,1,-0.5*sin(val1));
    gsl_matrix_set(V_t,1,0,sin(val1));
    gsl_matrix_set(V_t,1,1,0.5*cos(val1));
    gsl_matrix_set(V_t,2,1,1);

    gsl_matrix* mm = motion_model(x,u,dt);
    for(int r = 0; r < x->size1; r++)
    {
        for(int c = 0; c < x->size2; c++)
        {
            gsl_matrix_set(xEst,r,c,gsl_matrix_get(mm,r,c));
        }
    }
    gsl_matrix_free(mm);

    gsl_matrix* PEst_S = gsl_matrix_calloc(S,S);
    for(int r = 0; r < S; r++)
    {
        for(int c = 0; c < S; c++)
        {
            gsl_matrix_set(PEst_S,r,c,gsl_matrix_get(PEst,r,c));
        }
    }

    gsl_matrix* GT = gsl_matrix_calloc(G->size2,G->size1);
    gsl_matrix_transpose_memcpy(GT,G);
    gsl_matrix* FxT = gsl_matrix_calloc(Fx->size2,Fx->size1);
    gsl_matrix_transpose_memcpy(FxT,Fx);

    gsl_matrix* Cxmat = gsl_matrix_calloc(3,3);
    for(int r = 0; r < 3; r++)
    {
        for(int c = 0; c < 3; c++)
        {
            gsl_matrix_set(Cxmat,r,c,Cx[r][c]);
        }
    }
    
    gsl_matrix* arr1 = gsl_matrix_calloc(GT->size1,PEst_S->size2);
    gsl_linalg_matmult(GT,PEst_S,arr1);
    gsl_matrix* arr2 = gsl_matrix_calloc(arr1->size1,G->size2);
    gsl_linalg_matmult(arr1,G,arr2);

    gsl_matrix* arr3 = gsl_matrix_calloc(FxT->size1,Cxmat->size2);
    gsl_linalg_matmult(FxT,Cxmat,arr3);
    gsl_matrix* arr4 = gsl_matrix_calloc(arr3->size1,Fx->size2);
    gsl_linalg_matmult(arr3,Fx,arr4);
    gsl_matrix_add(arr2,arr4);

    for(int r = 0; r < S; r++)
    {
        for(int c = 0; c < S; c++)
        {
            gsl_matrix_set(PEst,r,c,gsl_matrix_get(arr2,r,c));
        }
    }
    gsl_matrix_free(arr1);
    gsl_matrix_free(arr2);
    gsl_matrix_free(arr3);
    gsl_matrix_free(arr4);
    gsl_matrix* initP = gsl_matrix_calloc(2,2);
    gsl_matrix_set(initP,0,0,1);
    gsl_matrix_set(initP,1,1,1);
    gsl_matrix_free(x);
    gsl_matrix_free(G);
    gsl_matrix_free(GT);
    gsl_matrix_free(Fx);
    gsl_matrix_free(FxT);
    delete gFx;
    gsl_matrix_free(M_t);
    gsl_matrix_free(V_t);
    gsl_matrix_free(PEst_S);
    gsl_matrix_free(Cxmat);
    
    

    
    list<gsl_matrix*> cones;
    
    //cones[0] = gsl_matrix_calloc(2,1);
    
    gsl_matrix* newZ = gsl_matrix_calloc(1,2);
    for(int iz = 0; iz < z->size1; iz++)
    {
        gsl_matrix_set(newZ,0,0,gsl_matrix_get(z,iz,0));
        gsl_matrix_set(newZ,0,1,gsl_matrix_get(z,iz,1));

        int min_id = search_correspond_landmark_id(xEst,PEst,newZ);
        gsl_matrix* zRow = gsl_matrix_calloc(1,z->size2);
        for(int c = 0; c < z->size2; c++)
        {
            gsl_matrix_set(zRow,iz,c,gsl_matrix_get(z,iz,c));
        }
        gsl_matrix* landPos = calc_landmark_position(xEst,zRow);
        gsl_matrix_free(zRow);
        cones.push_back(landPos);
        int nLM = calc_n_lm(xEst);

        if(min_id==nLM) {
            cout << "New LM" << endl;
            gsl_matrix* xaug = gsl_matrix_calloc(xEst->size1+2,1);
            for(int i = 0; i < xEst->size1; i++)
            {
                gsl_matrix_set(xaug,i,0,gsl_matrix_get(xEst,i,0));
            }
            gsl_matrix_set(xaug,xEst->size1,0,gsl_matrix_get(landPos,0,0));
            gsl_matrix_set(xaug,xEst->size1+1,0,gsl_matrix_get(landPos,1,0));

            gsl_matrix* paug = gsl_matrix_calloc(PEst->size1 + LM_SIZE, PEst->size2 + LM_SIZE);
            for(int r = 0; r < PEst->size1; r++)
            {
                for(int c = 0; c < PEst->size2; c++)
                {
                    gsl_matrix_set(paug,r,c,gsl_matrix_get(PEst,r,c));
                }
            }
            
            int rowOffset = PEst->size1 + xEst->size1 - initP->size1;
            int colOffset = PEst->size2 + LM_SIZE - initP->size2;
            for(int r = 0; r < initP->size1; r++)
            {
                for(int c = 0; c < initP->size2; c++)
                {
                    gsl_matrix_set(paug,r+rowOffset,c+colOffset,gsl_matrix_get(initP,r,c));
                }
            }

            gsl_matrix_free(xEst);
            xEst = gsl_matrix_calloc(xaug->size1,xaug->size2);

            for(int r = 0; r < xaug->size1; r++)
            {
                for(int c = 0; c < xaug->size2; c++)
                {
                    gsl_matrix_set(xEst,r,c,gsl_matrix_get(xaug,r,c));
                }
            }
            gsl_matrix_free(xaug);

            gsl_matrix_free(PEst);
            PEst = gsl_matrix_calloc(paug->size1,paug->size2);

            for(int r = 0; r < paug->size1; r++)
            {
                for(int c = 0; c < paug->size2; c++)
                {
                    gsl_matrix_set(PEst,r,c,gsl_matrix_get(paug,r,c));
                }
            }
            gsl_matrix_free(paug);


        }
        gsl_matrix_free(landPos);
        gsl_matrix* lm = get_landmark_position_from_state(xEst,min_id);
        gsl_matrix** ySH = calc_innovation(lm,xEst,PEst,newZ,min_id);
        gsl_matrix* y = ySH[0];
        gsl_matrix* Smat = ySH[1];
        gsl_matrix* H = ySH[2];

        gsl_matrix* HT = gsl_matrix_calloc(H->size2,H->size1);
        gsl_matrix_transpose_memcpy(HT,H);

        gsl_permutation* p = gsl_permutation_calloc(Smat->size1);
        int * signum = new int[1];
        signum[0] = 0;
        gsl_linalg_LU_decomp(Smat,p,signum);
        gsl_matrix* S_inverse = gsl_matrix_calloc(Smat->size1,Smat->size2);
        gsl_linalg_LU_invert(Smat,p,S_inverse);
        gsl_permutation_free(p);

        delete signum;


        arr1 = gsl_matrix_calloc(PEst->size1,HT->size2);
        gsl_linalg_matmult(PEst,HT,arr1);
        gsl_matrix* K = gsl_matrix_calloc(arr1->size1,S_inverse->size2);
        gsl_linalg_matmult(arr1,S_inverse,K);
        gsl_matrix_free(arr1);

        gsl_matrix* xEst_3 = gsl_matrix_calloc(xEst->size1-3, xEst->size2);
        for(int r = 0; r < xEst_3->size1; r++)
        {
            for(int c = 0; c < xEst_3->size2; c++)
            {
                gsl_matrix_set(xEst_3,r,c,gsl_matrix_get(xEst,r+3,c));
            }
        }

        gsl_matrix* K_3 = gsl_matrix_calloc(K->size1-3, K->size2);
        for(int r = 0; r < K_3->size1; r++)
        {
            for(int c = 0; c < K_3->size2; c++)
            {
                gsl_matrix_set(xEst_3,r,c,gsl_matrix_get(xEst,r+3,c));
            }
        }

        
        arr1 = gsl_matrix_calloc(K_3->size1, y->size2);
        gsl_linalg_matmult(K_3,y,arr1);
        gsl_matrix_add(xEst_3,arr1);
        gsl_matrix_free(arr1);


        for(int r = 3; r < xEst->size1; r++)
        {
            for(int c = 0; c < xEst->size2; c++)
            {
                gsl_matrix_set(xEst,r,c,gsl_matrix_get(xEst_3,r-3,c));
            }
        }
        
        
        arr2 = gsl_matrix_calloc(K->size1,H->size2);
        gsl_linalg_matmult(K_3,y,arr2);

        gsl_matrix* eye = gsl_matrix_calloc(xEst->size1,xEst->size1);
        for(int i = 0; i < xEst->size1; i++)
        {
            gsl_matrix_set(eye,i,i,1);
        }

        gsl_matrix_sub(eye,arr2);
        gsl_matrix* prod = gsl_matrix_calloc(eye->size1,PEst->size2);
        gsl_linalg_matmult(eye,PEst,prod);
        gsl_matrix_free(arr2);

        gsl_matrix_free(PEst);
        PEst = gsl_matrix_calloc(prod->size1,prod->size2);

        for(int r = 0; r < prod->size1; r++)
        {
            for(int c = 0; c < prod->size2; c++)
            {
                gsl_matrix_set(PEst,r,c,gsl_matrix_get(prod,r,c));
            }
        }
        gsl_matrix_free(lm);
        gsl_matrix_free(ySH[0]);
        gsl_matrix_free(ySH[1]);
        gsl_matrix_free(ySH[2]);
        delete ySH;
        gsl_matrix_free(HT);
        gsl_matrix_free(S_inverse);
        gsl_matrix_free(K);
        gsl_matrix_free(xEst_3);
        gsl_matrix_free(K_3);
        gsl_matrix_free(eye);
        gsl_matrix_free(prod);


    
        
    }
    gsl_matrix_free(newZ);

    gsl_matrix_set(xEst,2,0,pi_2_pi(gsl_matrix_get(xEst,2,0)));

    ekfPackage result;
    result.x = xEst;
    result.p = PEst;
    result.cone = cones;
    
    return result;
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