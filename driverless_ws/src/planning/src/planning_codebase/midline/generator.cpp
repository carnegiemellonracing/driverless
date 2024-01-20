#include "generator.hpp"
#include <eigen3/Eigen/Dense>

MidpointGenerator::MidpointGenerator(int interpolation_num){
    interpolation_number=interpolation_num;

}

double hypot_x2(std::pair<double,double> a){
    return powf64(a.first,4) + powf64(a.second,2); 
}

//fix sort by norm
bool ptnrm_cmp(std::pair<double,double> a,std::pair<double,double> b){
    //square the x term
    return hypot_x2(a)< hypot_x2(b);
}

std::vector<std::pair<double,double>>  MidpointGenerator::sorted_by_norm(rclcpp::Logger logger,std::vector<std::pair<double,double>> inp){

    std::sort(inp.begin(),inp.end(),ptnrm_cmp);
    
    for(int i = 0; i < inp.size(); i++){
        RCLCPP_INFO(logger, "sorted point %d is (%f, %f)\n", i, inp[i].first, inp[i].second);
    }
    return inp;
}


Eigen::MatrixXd midpoint(rclcpp::Logger logger, Eigen::MatrixXd &left,Eigen::MatrixXd &right){
    int cols = left.cols() +right.cols() -1;
    Eigen::MatrixXd midpt(2,cols);
    
    double lx = left(0,0);
    double rx = right(1,0);
    double ly = left(0,0);
    double ry = right(1,0);

    int l = 0; //index of inner
    int r = 0; //index of outer

    midpt(0,0) = (lx+rx)/2;
    midpt(1,0) = (ly+ry)/2;

    int m = 1; //index of midpoint
    while(l<left.cols()-1 && r < right.cols()-1){
        double lxp1 = left(0,l+1); //x-coordinater of inner[i+1]
        double lyp1 = left(1,l+1); //y-coordinater of inner[i+1]
        double rxp1 = right(0,r+1); //x-coordinater of outer[o+1]
        double ryp1 = right(1,r+1); //y-coordinater of inner[o+1]

        double dist_l_rp1 = pow((rxp1-lx),2) + pow((ryp1-ly),2); //distance between inner[i] and outer[o+1]
        double dist_lp1_r = pow((rx-lxp1),2) + pow((ry-lyp1),2); //distance between inner[i+1] and outer[o]
        if(dist_l_rp1 <= dist_lp1_r){
            r++;
            rx = rxp1;
            ry = ryp1;
            midpt(0,m) = (lx+rx)/2;
            midpt(1,m) = (ly+ry)/2;
        }else{
            l++;
            lx = lxp1;
            ly = lyp1;
            midpt(0,m) = (lx+rx)/2;
            midpt(1,m) = (ly+ry)/2;
        }
        m++;
    }

    if(r == right.cols()-1)
    {
        while(l<left.cols()-1)
        {
            lx = left(0,l); //x-coordinater of inner[i+1]
            ly = left(1,l);
            midpt(0,m) = (lx+rx)/2;
            midpt(1,m) = (ly+ry)/2;
            l++;
            m++;
        }
    }
    else{
        while(r<right.cols()-1)
        {
            rx = left(0,r); //x-coordinater of inner[i+1]
            ry = left(1,r);
            midpt(0,m) = (lx+rx)/2;
            midpt(1,m) = (ly+ry)/2;
            r++;
            m++;
        }
    }
    return midpt;
}



//delaunay triangulation
Eigen::MatrixXd midpoint_new(rclcpp::Logger logger, Eigen::MatrixXd& left,Eigen::MatrixXd& right){
    int cols = left.cols() +right.cols() -1;
    Eigen::MatrixXd midpt(2,cols);

    // RCLCPP_INFO(logger, "left matrix size: %d\n", left.cols());
    // RCLCPP_INFO(logger, "right matrix size: %d\n", right.cols());
    // for(int i = 0 ; i<left.cols(); i ++){
    //     // RCLCPP_INFO(logger, "left matrix: point %d is (%f, %f)\n", i, left(0, i), left(1, i));
    // }
    // for(int i = 0 ; i<right.cols(); i ++){
    //     // RCLCPP_INFO(logger, "right matrix: point %d is (%f, %f)\n", i, right(0, i), right(1, i));
    // }

    double left_x = left(0 , 0); //x-coordinate of first left point
    double left_y = left(1 , 0); //y-coordinate
    double right_x = right(0 , 0); //x-coordinate of first right point
    double right_y = right(1 , 0); //y-coordinate

    midpt(0,0)=(left_x+right_x)/2;
    midpt(1,0)=(left_y+right_y)/2;
    // RCLCPP_INFO(logger, "first midpoint: (%f, %f)\n", midpt(0, i), midpt(1, i));

    int l = 0; //index of left matrix
    int r = 0; //index of right matrix
    int m = 1; //index of midpt

    while(l<left.cols()-1 && r<right.cols()-1){
        double left_xp1 = left(0 , l+1); //x-coordinate of inner[i+1]
        double left_yp1 = left(1 , l+1); //y-coordinate of inner[i+1]
        double right_xp1 = right(0 , r+1); //x-coordinate of outer[o+1]
        double right_yp1 = right(1 , r+1); //y-coordinate of inner[o+1]
        double dist_l_rp1 = pow((right_xp1-left_x),2) + pow((right_yp1-left_y),2); //distance between inner[i] and outer[o+1]
        double dist_lp1_r = pow((right_x-left_xp1),2) + pow((right_y-left_yp1),2); //distance between inner[i+1] and outer[o]
        if(dist_l_rp1 <= dist_lp1_r){
            r++;
            right_x = right_xp1;
            right_y = right_yp1;
            midpt(0,m)=(left_x+right_x)/2;
            midpt(1,m)=(left_y+right_y)/2;
        }else{
            l++;
            left_x = left_xp1;
            left_y = left_yp1;
            midpt(0,m)=(left_x+right_x)/2;
            midpt(1,m)=(left_y+right_y)/2;
        }
        m++;
    }

    if(r == right.cols()-1)
    {
        l++;
        while(l<left.cols())
        {
            // RCLCPP_INFO(logger, "right point: (%f, %f)\n", right_x, right_y);
            left_x = left(0 , l); //x-coordinate of inner[i+1]
            left_y = left(1 , l);
            // RCLCPP_INFO(logger, "left point: (%f, %f)\n", left_x, left_y);
            midpt(0,m)=(left_x+right_x)/2;
            midpt(1,m)=(left_y+right_y)/2;
            l++;
            m++;
        }
    }
    else if (l == left.cols()-1){
        //l == left->size2-1
        r++;
        while(r<right.cols())
        {
            // RCLCPP_INFO(logger, "left point: (%f, %f)\n", left_x, left_y);
            right_x = right(0 , r); 
            right_y = right(1 , r);
            midpt(0,m)=(left_x+right_x)/2;
            midpt(1,m)=(left_y+right_y)/2;
            r++;
            m++;
        }
    }

    // for(int i=0;i<midpt->size1;i++)
    //     for(int j=0;j<midpt->size1;j++)
    //     gsl_matrix_set(midpt,i,j,(gsl_matrix_get(inner,i,j)+gsl_matrix_get(outer,i,j))/2);

    return midpt;
}

std::vector<Spline> MidpointGenerator::generate_splines(rclcpp::Logger logger, Eigen::MatrixXd& midpoints){
    std::pair<std::vector<Spline>,std::vector<double>> a= raceline_gen(logger, midpoints,std::rand(), 4, false);
    // auto result =  raceline_gen(midpoints,std::rand(),midpoints->size2,false);
    for (auto &e:a.first){
        cumulated_splines.push_back(e);
    }
    for(auto &e:a.second){
        cumulated_lengths.push_back(e);
    }
    return a.first;
}

Eigen::MatrixXd MidpointGenerator::generate_points(rclcpp::Logger logger, perceptionsData perceptions_data){ 
    // LEFT ==BLUE
    //Sort cones by dist
    perceptions_data.bluecones  = sorted_by_norm(logger,perceptions_data.bluecones);
    perceptions_data.yellowcones  = sorted_by_norm(logger,perceptions_data.yellowcones);
    
    //Interpolate for blue cones
    if (perceptions_data.bluecones.size()>0 && perceptions_data.yellowcones.size()==0){
        RCLCPP_INFO(logger, "interpolate from blue");

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
        Eigen::MatrixXd midpoint_mat = midpoint(logger, left, right);
        return midpoint_mat;
    }
    //Interpolate for yellow cones
    if (perceptions_data.bluecones.size()==0 && perceptions_data.yellowcones.size()>0){
        RCLCPP_INFO(logger, "interpolate from yellow");

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

        Eigen::MatrixXd midpoint_mat = midpoint(logger, left,right);
        return midpoint_mat;
    }
    // int size = std::min(perceptions_data.bluecones.size(),perceptions_data.yellowcones.size());
    Eigen::MatrixXd left(2,perceptions_data.bluecones.size());
    Eigen::MatrixXd right(2,perceptions_data.yellowcones.size());
    for(int i = 0; i < perceptions_data.bluecones.size(); i++){
        assert(i < left.cols());
        left(0, i) = perceptions_data.bluecones[i].first;
        left(1, i) = perceptions_data.bluecones[i].second;
    }

    for(int i = 0; i < perceptions_data.yellowcones.size(); i++){
        assert(i < right.cols());
        right(0, i) = perceptions_data.yellowcones[i].first;
        right(1, i) = perceptions_data.yellowcones[i].second;
    }

    // RCLCPP_INFO(logger, "leftcones size is %d\n", perceptions_data.bluecones.size());
    // RCLCPP_INFO(logger, "rightcones size is %d\n", perceptions_data.yellowcones.size());
    // RCLCPP_INFO(logger, "left matrix size is %d\n", left.cols());
    // RCLCPP_INFO(logger, "right matrix size is %d\n", right.cols());

    
    RCLCPP_INFO(logger, "no interpolation");

    for(int i = 0 ; i<left.cols(); i ++){
        RCLCPP_INFO(logger, "left point %d is (%f, %f)\n", i, left(0, i), left(1, i));
    }
    
    for(int i = 0 ; i<right.cols(); i ++){
        RCLCPP_INFO(logger, "right point %d is (%f, %f)\n", i, right(0, i), right(1, i));
    }

    Eigen::MatrixXd midpoint_mat = midpoint(logger, left, right);
    Eigen::MatrixXd ret(2, midpoint_mat.cols() + 1);
    ret.col(0).setZero();
    ret.block(0, 1, 2, midpoint_mat.cols()) = midpoint_mat; // equiv to ret.rightCols(midpoint_mat.cols()) = midpoint_mat;
    return ret;

}

Eigen::MatrixXd MidpointGenerator::interpolate_cones(rclcpp::Logger logger, perceptionsData perceptions_data,int interpolation_number){
    return spline_from_cones(logger, perceptions_data).interpolate(interpolation_number,std::make_pair(-1,-1));
}

Spline MidpointGenerator::spline_from_cones(rclcpp::Logger logger, perceptionsData perceptions_data){
    Eigen::MatrixXd midpoints= generate_points(logger, perceptions_data);

    RCLCPP_INFO(logger, "number of points: %d\n", midpoints.cols());
    for(int i = 0 ; i<midpoints.cols(); i ++){
        RCLCPP_INFO(logger, "point %d is (%f, %f)\n", i, midpoints(0, i), midpoints(1, i));
    }
    // RCLCPP_INFO(logger, "first point is (%f, %f)\n", midpoints(0, 0), midpoints(1, 0));
    // RCLCPP_INFO(logger, "second point is (%f, %f)\n", midpoints(0, 1), midpoints(1, 1));

    std::vector<Spline> splines = generate_splines(logger, midpoints);
    if (splines.size() == 0) return Spline(poly_one());
    return (splines[0]);
    
    // Spline newSpline = *(new(Spline));
}

Eigen::MatrixXd vector_to_mat(std::vector<std::pair<double,double>> side){
    Eigen::MatrixXd mat(2,side.size());
    for(int i=0;i<side.size();i++){
        mat(0,i)=side[i].first;
        mat(1,i)=side[i].second;
    }

    return mat;
}

Spline MidpointGenerator::spline_from_curve(rclcpp::Logger logger, std::vector<std::pair<double,double>> side){
    Eigen::MatrixXd side_mat= vector_to_mat(side);
    std::vector<Spline> splines = generate_splines(logger, side_mat);
    return (splines[0]);
}