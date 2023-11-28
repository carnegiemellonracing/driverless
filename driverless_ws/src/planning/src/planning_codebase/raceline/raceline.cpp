#include "raceline.hpp"
#include <Eigen/Dense>
// #include "random.h"

polynomial poly(int deg = 3){
    polynomial inst;
    inst.deg=deg;
    Eigen::VectorXd nums(deg+1);
    nums.setZero();
    inst.nums = nums;
    
	return inst;
}

polynomial poly_one(){
    polynomial p = poly(0);
    p.nums(0)=1;
    return p;
}

polynomial poly_root(double root){
    polynomial p = poly(1);
    p.nums(0) = -root;
    p.nums(1) = 1;
    return p;
}

polynomial polyder(polynomial p){
    if (p.deg ==0) return poly(0);
    polynomial der = poly(p.deg);
    for(int i=0;i<p.deg;i++){
        double coef = p.nums(i+1)*(i+1);
        der.nums(i)=coef;
    }
    p.nums(p.deg)=0;

	return der;
}



polynomial poly_mult(polynomial a,polynomial b){
    polynomial mult = poly(a.deg+b.deg);

    for(int x=0;x<=a.deg;x++){
        for(int y=0;y<=b.deg;y++){
            mult.nums(x+y) = mult.nums(x+y)+a.nums(x)*b.nums(y);

        }
    }
	return mult;
}

double poly_eval(polynomial a, double x){
    double result = 0;
    double xval = 1.0;
    for(int i = 0; i <= a.deg; i++){
        result += a.nums(i) * xval;
        xval*=x;
    }
    return result;
}

Spline::Spline(polynomial interpolation_poly) {
    this->spl_poly = interpolation_poly;
}

Spline::Spline(polynomial interpolation_poly, polynomial first, polynomial second, int path, int sort_ind) {
    this->spl_poly=interpolation_poly;
    this->first_der = first;
    this->second_der = second;
    this->path_id = path_id;
    this->sort_index = sort_ind;
}


Spline::Spline(polynomial interpolation_poly, Eigen::MatrixXd points_mat,Eigen::MatrixXd rotated,Eigen::Matrix2d Q_mat, Eigen::VectorXd translation,polynomial first, polynomial second, int path, int sort_ind)
{
    this->spl_poly=interpolation_poly;
    // Eigen::Matrix<double, 2, 4> points;
    
    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         points(i, j) = points_mat(i, j);
    //     }
    // }
    this->points = points_mat;
    this->rotated_points=rotated;
    this->Q = Q_mat;
    this->translation_vector = translation;
    this->first_der = first;
    this->second_der = second;
    this->path_id = path_id;
    this->sort_index = sort_ind;

}

Spline::~Spline()
{
    // ~spl_poly();
    // polynomial first_der;
    // polynomial second_der;
    
    //No need for this function in Eigen as it frees memory itself

}

double Spline::length(){
    // return 1.0;
    return arclength(first_der, rotated_points(0,0), rotated_points(0, rotated_points.cols()-1));
}

// Eigen::MatrixXd Spline::interpolate(int number, std::pair<float, float> bounds){
//     return interpolate(*this,number,bounds);
// }

bool Spline::operator==(Spline const & other) const{
    return this->sort_index==other.sort_index;
}

bool Spline::operator<(Spline const & other) const{
    return this->sort_index<other.sort_index;
}

polynomial Spline::get_first_der(){
    return this->first_der;
}

polynomial Spline::get_second_der(){
    return this->second_der;
}

Eigen::MatrixXd  Spline::get_points(){
    return points;}

Eigen::MatrixXd  Spline::get_rotated_points(){
    return rotated_points;
}

Eigen::Matrix2d Spline::get_Q(){
    return Q;
}

Eigen::VectorXd Spline::get_translation(){
    return translation_vector;
}

int Spline::get_path_id(){
    return path_id;
}

int Spline::get_sort_index(){
    return sort_index;
}




std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> Spline::along(double progress, double point_index, int precision){
    
    std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> ret;


    double len = this->length();


    double first_x = this->get_rotated_points()(0,0);
    double last_x = this->get_rotated_points()(0,this->get_rotated_points().cols());

    double delta = last_x - first_x;

    std::pair<double,double> boundaries = std::make_pair(first_x,last_x);
    int ratio = progress / len + 1;



        if (ratio >= 2){
            double x = first_x + delta*ratio;
            double shoot = arclength(this->spl_poly,first_x,x);
            
            double lower_bound = first_x + delta * (ratio - 1);
            double upper_bound =  first_x + delta * ratio;

            if (shoot < progress){
                while (shoot < progress){
                    lower_bound = x;
                    //  add approximately one spline length to shoot
                    shoot = arclength(this->spl_poly,first_x,x+delta);
                    x=x+delta;
                }
                upper_bound = x; // upper bound is direct first overshoot (separated by delta from the lower bound)
            }
            else if (shoot >= progress){ // equality not very important
                while (shoot >= progress){
                    upper_bound = x;
                    // # remove approximately one splien length to shoot
                    shoot = arclength(this->spl_poly,first_x,x - delta);
                }
                lower_bound = x; // lower bound is direct first undershoot (separated by delta from the upper bound)
            }    
            std::pair<double,double> boundaries = std::make_pair(lower_bound, upper_bound);
            
        }
        //  Perform a more precise search between the two computed bounds

        std::vector<double> guesses;
        guesses.resize(precision+1);
        for(int i=0;i<=precision;i++){
            guesses[i] = (boundaries.first*i + boundaries.second*(precision-i))/precision;
        }

        //  Evaluate progress along the (extrapolated) spline
        //  As arclength is expensive and cannot take multiple points
        //  at the same time, it is faster to use a for loop
        double past = -1, best_guess = -1, best_length = -1;
        for (double guess : guesses){
            double guess_length = arclength(this->spl_poly, first_x, guess);
            if (abs(progress - guess_length) > abs(progress - past)) //# if we did worst than before
                break;
            best_guess = guess;
            best_length = guess_length;
            past = guess_length;
        }
        Eigen::VectorXd rotated_point(2);
        rotated_point(0)=best_guess;
        
        rotated_point(1)=poly_eval(this->spl_poly,best_guess);
        
        Eigen::MatrixXd rotated_points(2,1);
        rotated_points(0,0)=rotated_point(0);
        rotated_points(0,1)=rotated_point(1);
    
        Eigen::MatrixXd point_mat =reverse_transform(rotated_points,this->Q,this->translation_vector);
        
        Eigen::VectorXd point (2);
        point(0)=point_mat(0);
        point(1)=point_mat(1);

        ret = std::make_tuple(point,best_length,rotated_point,best_guess);

        return ret;
}

double Spline::getderiv(double x){
    Eigen::MatrixXd point_x(1,2);
    point_x(0,0)=x;
    point_x(0,1)=0;

    Eigen::MatrixXd gm= reverse_transform(point_x,this->Q,this->translation_vector);
    return poly_eval(this->first_der,gm.data()[0]);



}

Eigen::MatrixXd Spline::interpolate(int number, std::pair<float,float> bounds){

    if(bounds.first == -1 && bounds.second == -1){
        double bound1 = get_rotated_points()(0,0);
        // MAKE PROPER BOUND 2
        double bound2 = get_rotated_points()(0,get_rotated_points().cols());
        bounds = std::make_pair(bound1,bound2);
    }

    Eigen::MatrixXd points(number,2);
    
    for(int i=0;i<number;i++){
        double x = bounds.first+ (bounds.second-bounds.first)*(i/(number-1));
        points(i,0)=x;
        points(i,1)=poly_eval(get_SplPoly(),x);
    }

    

    Eigen::Matrix2d q = Spline::get_Q();
    Eigen::VectorXd trans = Spline::get_translation();
	Eigen::MatrixXd ret= reverse_transform(points, q, trans);

    return ret;
}

Eigen::Matrix2d rotation_matrix_gen(rclcpp::Logger logger,Eigen::MatrixXd& pnts){
    Eigen::Vector2d beg; beg << pnts.col(0);
    Eigen::Vector2d end; end << pnts.col(pnts.cols()-1);

    Eigen::Vector2d diff = end-beg;

    double norm = diff.norm();

    double cos = diff(0)/norm;
    double sin = diff(1)/norm;

    Eigen::Matrix2d ret;
    ret(0,0)=cos;
    ret(1,0)=sin;
    ret(0,1)=-1*sin;
    ret(1,1)=cos;

    // RCLCPP_INFO(logger, "(sin,cos),(%f, %f)\n", sin,cos);
    // RCLCPP_INFO(logger, "(diff,norm),(%f, %f),%f\n", diff(0),diff(1),norm);
    return ret;
}

Eigen::VectorXd get_translation_vector(Eigen::MatrixXd& group){
    Eigen::Vector2d ret;
    ret(0) = group(0, 0);
    ret(1) = group(1, 0);
    return ret;
}

Eigen::MatrixXd transform_points(rclcpp::Logger logger,Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector){
    Eigen::MatrixXd temp(points.rows(),points.cols());
        // RCLCPP_INFO(logger, "transform points:rotation");
        // RCLCPP_INFO(logger, "first point is (%f, %f)\n", Q(0, 0), Q(0, 1));
        // RCLCPP_INFO(logger, "second point is (%f, %f)\n", Q(1, 0), Q(1, 1));
    
    for(int i=0;i<temp.cols();++i){
        temp(0,i)=points(0,i)-get_translation_vector(0);
        temp(1,i)=points(1,i)-get_translation_vector(1);
    }

    // RCLCPP_INFO(logger, "temp");
    // RCLCPP_INFO(logger, "first point is (%f, %f)\n", temp(0, 0), temp(0, 1));
    // RCLCPP_INFO(logger, "second point is (%f, %f)\n", temp(1, 0),temp(1, 1));


    Eigen::Matrix2d trans = Q.transpose(); 
    // RCLCPP_INFO(logger, "q.trans");
    // RCLCPP_INFO(logger, "first point is (%f, %f)\n", trans(0, 0), trans(0, 1));
    // RCLCPP_INFO(logger, "second point is (%f, %f)\n", trans(1, 0), trans(1, 1));
    Eigen::MatrixXd ret = Q.transpose() * temp;  
    // RCLCPP_INFO(logger, "return");
    // RCLCPP_INFO(logger, "first point is (%f, %f)\n", ret(0, 0), ret(0, 1));
    // RCLCPP_INFO(logger, "second point is (%f, %f)\n", ret(1, 0), ret(1, 1));
    
    return ret;
}

Eigen::MatrixXd reverse_transform(Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector){
    Eigen::MatrixXd temp(points.rows(),points.cols());
    for(int i=0;i<temp.cols();++i){
        temp(0,i)=points(0,i);
        temp(1,i)=points(1,i);
    }

    Eigen::MatrixXd ret = temp*Q;

    for(int i=0;i<temp.cols();++i){
        temp(0,i)= points(0,i)+ get_translation_vector(0);
        temp(1,i)= points(1,i)+ get_translation_vector(1);
    }

    return ret;
}

polynomial lagrange_gen(Eigen::MatrixXd& points){
    polynomial lagrange_poly = poly(points.cols() - 1);


    double x[points.cols()];
    double y[points.cols()];

    for(int i=0;i<points.cols();i++){
        x[i] = points(0,i);
        y[i] = points(1,i);
    }


    for(int i=0;i<points.cols();i++){
        polynomial p = poly_one();
        p.nums(0)=1;
        for(int j=0;j<points.cols();j++){

            if(j != i){
                polynomial pr = poly_root(x[j]);
                polynomial q = poly_mult(p,pr);
                p=q;
            }
        }
        polynomial p1 = poly_one();
        p1.nums(0) = (y[i] / poly_eval(p, x[i]));
        polynomial q = poly_mult(p1,p); // scaling by y_i / sum (x_i - x_j)

        lagrange_poly.nums += q.nums;
        
    }
    
    return lagrange_poly;  

}

double arclength_f(double x, void* params){
    
    polynomial p = *(polynomial*)params;
    double y = poly_eval(p,x);
    return sqrt(y*y+1);
}


// CHECK CORRECTNESS
double arclength(polynomial poly_der1, double x0,double x1){

    gsl_function F;
    F.function = &arclength_f;
    F.params = &poly_der1;

    double result, error;
    size_t neval;
    // gsl_integration_workspace *w 
    //      = gsl_integration_workspace_alloc (100000);

    gsl_integration_qng (&F, x0, x1, 1, 1e-1, &result, &error, &neval);
    // gsl_integration_workspace_free(w); 

    return result;

}

std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id, int points_per_spline,bool loop){

    int n = res.cols();

    std::vector<Spline> splines;

    // Eigen::MatrixXd points=res;

    int shift = points_per_spline-1;
    int group_numbers;

    if (shift == 1){
        group_numbers = n/shift;

        if (loop)
            group_numbers += (int)(n % shift != 0);
    }
    else{
        if (n < 4) group_numbers = 0; // NEED TO MODIFY TO 1 AND DEAL WITH FEWER THAN 4 POINTS
        else group_numbers = n/3;
        // RCLCPP_INFO(logger, "group numbers is %d\n", group_numbers);
    }
    std::vector<std::vector<int>> groups;

    
    std::vector<double> lengths;
    std::vector<double> cumsum;
    // lengths.resize(group_numbers);

    for(int i=0; i<group_numbers; i++){
        // Eigen::MatrixXd group(res,0,group_numbers*shift,2,3);

        // RCLCPP_INFO(logger, "\nnew group beginning\n");

        Eigen::MatrixXd group(2, points_per_spline);
        for(int k = 0; k < group.cols(); k++) {
            for (int j = 0; j < 2; j++) {
                group(j, k) = res(j, i*shift + k);
                // if (j==1) RCLCPP_INFO(logger, "point %d is (%f, %f)\n", k, group(0, k), group(1,k));
            }
        }



        Eigen::Matrix2d Q  = rotation_matrix_gen(logger,group);
        Eigen::VectorXd translation_vector = get_translation_vector(group);
        Eigen::MatrixXd rotated_points = transform_points(logger,group,Q,translation_vector);

        // RCLCPP_INFO(logger, "rotation matrix\n");
        // RCLCPP_INFO(logger, "first point is (%f, %f)\n", Q(0, 0), Q(0, 1));
        // RCLCPP_INFO(logger, "second point is (%f, %f)\n", Q(1, 0), Q(1, 1));

        // RCLCPP_INFO(logger, "Translation vector");
        // RCLCPP_INFO(logger, "(%f, %f)\n", translation_vector(0, 0), translation_vector(0, 1));

        // RCLCPP_INFO(logger, "rotated_points");
        // for (int i = 0; i < rotated_points.cols(); i++) {
        //     RCLCPP_INFO(logger, "point %d is (%f, %f)\n", i, rotated_points(0, i), rotated_points(1, i));

        // }
        // RCLCPP_INFO(logger, "second point is (%f, %f)\n", rotated_points(0, 1), rotated_points(1, 1));


        polynomial interpolation_poly = lagrange_gen(rotated_points);
        polynomial first_der = polyder(interpolation_poly);
        polynomial second_der = polyder(first_der);
        
        
        // Spline* spline = new Spline(interpolation_poly,group,rotated_points,Q,translation_vector,first_der,second_der,path_id,i);

        lengths.emplace_back(0);
        // Spline spline = Spline(interpolation_poly, first_der, second_der, path_id,i);
        Spline spline = Spline(interpolation_poly,group,rotated_points,Q,translation_vector,first_der,second_der,path_id,i);
        splines.emplace_back(spline);

        // lengths.push_back(spline.length());
        if (i == 0) {
            RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
            // RCLCPP_INFO(logger, "spline derivative is %f + %fx + %fx^2 + %fx^3\n", spline.first_der.nums(0), spline.first_der.nums(1), spline.first_der.nums(2), spline.first_der.nums(3));
            cumsum.push_back(splines[0].length());
        } else {
            RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
            cumsum.push_back(cumsum.back()+splines[0].length());
        }

    }

    return std::make_pair(splines,cumsum);



}

