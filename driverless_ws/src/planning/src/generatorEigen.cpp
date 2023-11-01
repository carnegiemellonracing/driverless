#include "generator.hpp"
#include <Eigen/Dense>

MidpointGenerator::MidpointGenerator(int interpolation_num){
    interpolation_number=interpolation_num;

}

bool ptnrm_cmp(std::pair<double,double> a,std::pair<double,double> b){
    return hypot(a.first,a.second) < hypot(b.first,b.second);
}

std::vector<std::pair<double,double>>  MidpointGenerator::sorted_by_norm(std::vector<std::pair<double,double>> inp){

    std::sort(inp.begin(),inp.end(),ptnrm_cmp);
    
    return inp;
}


Eigen::MatrixXd& midpoint(Eigen::MatrixXd& left,Eigen::MatrixXd& right){
    int cols = left.cols() +right.cols() -1;
    Eigen::MatrixXd midpt(2,cols);

    double lx = left(0 , 0); //x-coordinate of first inner point
    double ly = left(1 , 0); //y-coordinate
    double rx = right(0 , 0); //x-coordinate of first outer point
    double ry = right(1 , 0); //y-coordinate
    int l = 0; //index of inner
    int r = 0; //index of outer
    midpt(0,0)=(lx+rx)/2;
    midpt(1,0)=(ly+ry)/2;
    int m = 1; //index of midpt
    while(l<left.cols()-1 && r<right.cols()-1){
        double lxp1 = left(0 , l+1); //x-coordinater of inner[i+1]
        double lyp1 = left(1 , l+1); //y-coordinater of inner[i+1]
        double rxp1 = right(0 , r+1); //x-coordinater of outer[o+1]
        double ryp1 = right(1 , r+1); //y-coordinater of inner[o+1]
        double dist_l_rp1 = pow((rxp1-lx),2) + pow((ryp1-ly),2); //distance between inner[i] and outer[o+1]
        double dist_lp1_r = pow((rx-lxp1),2) + pow((ry-lyp1),2); //distance between inner[i+1] and outer[o]
        if(dist_l_rp1 <= dist_lp1_r){
            r++;
            rx = rxp1;
            ry = ryp1;
            midpt(0,m)=(lx+rx)/2;
            midpt(1,m)=(ly+ry)/2;
        }else{
            l++;
            lx = lxp1;
            ly = lyp1;
            midpt(0,m)=(lx+rx)/2;
            midpt(1,m)=(ly+ry)/2;
        }
        m++;
    }

    if(r == right.cols()-1)
    {
        while(l<left.cols()-1)
        {
            lx = left(0 , l); //x-coordinater of inner[i+1]
            ly = left(1 , l);
            midpt(0,m)=(lx+rx)/2;
            midpt(1,m)=(ly+ry)/2;
            l++;
            m++;
        }
    }
    else{
        //l == left->size2-1
        while(r<right.cols()-1)
        {
            rx = left(0 , r); //x-coordinater of inner[i+1]
            ry = left(1 , r);
            midpt(0,m)=(lx+rx)/2;
            midpt(1,m)=(ly+ry)/2;
            r++;
            m++;
        }
    }

    // for(int i=0;i<midpt->size1;i++)
    //     for(int j=0;j<midpt->size1;j++)
    //     gsl_matrix_set(midpt,i,j,(gsl_matrix_get(inner,i,j)+gsl_matrix_get(outer,i,j))/2);

    return midpt;
}


std::vector<Spline> MidpointGenerator::generate_splines(Eigen::MatrixXd& midpoints){
    std::pair<std::vector<Spline>,std::vector<double>> a= raceline_gen(midpoints,std::rand(),midpoints.cols(),false);
    // auto result =  raceline_gen(midpoints,std::rand(),midpoints->size2,false);
    for (auto &e:a.first){
        cumulated_splines.push_back(e);
    }
    for(auto &e:a.second){
        cumulated_lengths.push_back(e);
    }
    return a.first;
}



Eigen::MatrixXd& MidpointGenerator::generate_points(perceptionsData perceptions_data){ 
    // LEFT ==BLUE
    perceptions_data.bluecones  = sorted_by_norm(perceptions_data.bluecones);
    perceptions_data.yellowcones  = sorted_by_norm(perceptions_data.yellowcones);
    if (perceptions_data.bluecones.size()>0 && perceptions_data.yellowcones.size()==0){
        Eigen::MatrixXd left(2,perceptions_data.bluecones.size());
        Eigen::MatrixXd right(2,perceptions_data.bluecones.size());
        for(int i=0;i<perceptions_data.bluecones.size();i++){
            left(0,i)=perceptions_data.bluecones[i].first;
            left(1,i)=perceptions_data.bluecones[i].second;
            if(i==0){
                right(0,i)=-1*perceptions_data.bluecones[i].first;
                right(1,i)=perceptions_data.bluecones[i].second;
           }
            else{
                double xdiff = perceptions_data.bluecones[i].first-perceptions_data.bluecones[i-1].first;
                double ydiff = perceptions_data.bluecones[i].second-perceptions_data.bluecones[i-1].second;
                right(0,i)=right(0,i-1)+xdiff;
                right(1,i)=right(1,i-1)+ydiff;
            }
        }
        Eigen::MatrixXd& midpoint_mat = midpoint(left,right);
        // gsl_matrix_free(left);
        // gsl_matrix_free(right);
        return midpoint_mat;
    }
    if (perceptions_data.bluecones.size()==0 && perceptions_data.yellowcones.size()>0){
        Eigen::MatrixXd left(2,perceptions_data.yellowcones.size());
        Eigen::MatrixXd right(2,perceptions_data.yellowcones.size());
        for(int i=0;i<perceptions_data.yellowcones.size();i++){
            right(0,i)=perceptions_data.yellowcones[i].first;
            right(1,i)=perceptions_data.yellowcones[i].second;
            if(i==0){
                left(0,i)=-1*perceptions_data.yellowcones[i].first;
                left(1,i)=perceptions_data.yellowcones[i].second;
            }
            else{
                double xdiff = perceptions_data.yellowcones[i].first-perceptions_data.yellowcones[i-1].first;
                double ydiff = perceptions_data.yellowcones[i].second-perceptions_data.yellowcones[i-1].second;
                left(0,i)= left(0,i-1)+xdiff;
                left(1,i)= left(1,i-1)+ydiff;
            }
        }
        Eigen::MatrixXd& midpoint_mat = midpoint(left,right);
        // gsl_matrix_free(left);
        // gsl_matrix_free(right);
        return midpoint_mat;
    }
    double size = std::min(perceptions_data.bluecones.size(),perceptions_data.yellowcones.size());
        Eigen::MatrixXd left(2,size+1);
        Eigen::MatrixXd right(2,size+1);
        for(int i=0;i<perceptions_data.yellowcones.size();i++){
            left(0,i+1)=perceptions_data.bluecones[i].first;
            left(1,i+1)=perceptions_data.bluecones[i].second;
            right(0,i+1)=perceptions_data.yellowcones[i].first;
            right(1,i+1)=perceptions_data.yellowcones[i].second;

        }

        Eigen::MatrixXd& midpoint_mat = midpoint(left,right);
        // gsl_matrix_free(left);
        // gsl_matrix_free(right);
        return midpoint_mat;

}


Eigen::MatrixXd& MidpointGenerator::interpolate_cones(perceptionsData perceptions_data,int interpolation_number){
    return spline_from_cones(perceptions_data).interpolate(interpolation_number,std::make_pair(-1,-1));
}

Spline MidpointGenerator::spline_from_cones(perceptionsData perceptions_data){
    Eigen::MatrixXd midpoints= generate_points(perceptions_data);
    std::vector<Spline> splines = generate_splines(midpoints);
    // gsl_matrix_free(midpoints);
    return splines[0];
}

Eigen::MatrixXd& vector_to_mat(std::vector<std::pair<double,double>> side){
    Eigen::MatrixXd mat(2,side.size());
    for(int i=0;i<side.size();i++){
        mat(0,i)=side[i].first;
        mat(1,i)=side[i].second;
    }

    return mat;
}

Spline MidpointGenerator::spline_from_curve(std::vector<std::pair<double,double>> side){

    Eigen::MatrixXd& side_mat= vector_to_mat(side);
    std::vector<Spline> splines = generate_splines(side_mat);
    // gsl_matrix_free(side_mat);
    return splines[0];
}

// int main(){
//     gsl_matrix *left = gsl_matrix_alloc(2, 5);
//     gsl_matrix *right = gsl_matrix_alloc(2, 3);
//     std::vector<double> left_xcord = {-4.475, -4.45, -4.37, -4.16, -3.78};
//     std::vector<double> left_ycord = {0, 0.424, 0.95, 1.64, 2.39};
//     std::vector<double> right_xcord = {-3, -2.875, -2.68};
//     std::vector<double> right_ycord = {0, 0.857, 1.348};
//     for(int i = 0; i < 5; i++)
//     {
//         gsl_matrix_set(left, 0, i, left_xcord[i]);
//         gsl_matrix_set(left, 1, i, left_ycord[i]);
//     }
//     for(int i = 0; i < 3; i++)
//     {
//         gsl_matrix_set(right, 0, i, right_xcord[i]);
//         gsl_matrix_set(right, 1, i, right_ycord[i]);
//     }

//     gsl_matrix *mid = midpoint(left, right);
//     for(int i = 0; i < mid->size2; i++)
//     {
//         std::cout<<gsl_matrix_get(mid, 0, i) << " " << gsl_matrix_get(mid, 1, i) << "\n";
//     }
// }