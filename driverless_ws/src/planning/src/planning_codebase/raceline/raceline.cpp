#include "raceline.hpp"
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
// #include "random.h"

// lagrange gen helper function
polynomial poly(int deg = 3){
    polynomial inst;
    inst.deg=deg;
    Eigen::VectorXd nums(deg+1);
    nums.setZero();
    inst.nums = nums;
    
	return inst;
}

// lagrange gen helper function
polynomial poly_one(){
    polynomial p = poly(0);
    p.nums(0)=1;
    return p;
}

// lagrange gen helper function
polynomial poly_root(double root){
    polynomial p = poly(1);
    p.nums(0) = -root;
    p.nums(1) = 1;
    return p;
}

// derivative of a polynomial
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


// multiplies 2 polynomials
polynomial poly_mult(polynomial a,polynomial b){
    polynomial mult = poly(a.deg+b.deg);

    for(int x=0;x<=a.deg;x++){
        for(int y=0;y<=b.deg;y++){
            mult.nums(x+y) = mult.nums(x+y)+a.nums(x)*b.nums(y);

        }
    }
	return mult;
}

// result of passing in x into poly
double poly_eval(polynomial a, double x){
    double result = 0;
    double xval = 1.0;
    for(int i = 0; i <= a.deg; i++){
        result += a.nums(i) * xval;
        xval*=x;
    }
    return result;
}

// Curvature at point(s) `min_x` based on parametric curvature equation
double get_curvature(polynomial fder_x, polynomial fder_y, polynomial sder_x, polynomial sder_y,
                     double min_x) {
  int d2y_dt2 = poly_eval(sder_y, min_x);
  int d2x_dt2 = poly_eval(sder_x, min_x);
  int dy_dt = poly_eval(fder_y, min_x);
  int dx_dt = poly_eval(fder_x, min_x);     
  return fabs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) /
                       pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
}

// spline constructor (not in use)
Spline::Spline(polynomial interpolation_poly) {
    this->spl_poly = interpolation_poly;
}

// main spline function to use
// matrices are (x, unit) and (y, unit), where unit is col indices
Spline::Spline(polynomial poly_x, polynomial poly_y, Eigen::MatrixXd points_x, Eigen::MatrixXd points_y) {
    this->spl_poly_x = poly_x;
    this->spl_poly_y = poly_y;
    this->points_x = points_x;
    this->points_y = points_y;
    this->first_der_x = poly_der(poly_x);
    this->first_der_y = poly_der(poly_y);
    this->second_der_x = poly_der(this->first_der_x);
    this->second_der_y = poly_der(this->first_der_y);
    this->length = this->calculateLength();
}

// calculate length of spline
double Spline::calculateLength(){
    // return 1.0;
    return arclength(this->first_der_x, this->first_der_y, this->points_x(0,0), this->points_x(0, points_x.cols()-1));
}

// Eigen::MatrixXd Spline::interpolate(int number, std::pair<float, float> bounds){
//     return interpolate(*this,number,bounds);
// }

// bool Spline::operator==(Spline const & other) const{
//     return this->sort_index==other.sort_index;
// }

// bool Spline::operator<(Spline const & other) const{
//     return this->sort_index<other.sort_index;
// }

polynomial Spline::get_first_der_x(){
    return this->first_der_x;
}

polynomial Spline::get_first_der_y(){
    return this->first_der_y;
}

polynomial Spline::get_second_der_x(){
    return this->second_der_x;
}

polynomial Spline::get_second_der_y(){
    return this->second_der_y;
}

Eigen::MatrixXd  Spline::get_points_x(){
    return this->points_x;}

Eigen::MatrixXd  Spline::get_points_y(){
    return this->points_y;}

// Eigen::MatrixXd  Spline::get_rotated_points(){
//     return rotated_points;
// }

// Eigen::Matrix2d Spline::get_Q(){
//     return Q;
// }

// Eigen::VectorXd Spline::get_translation(){
//     return translation_vector;
// }

// int Spline::get_path_id(){
//     return path_id;
// }

// int Spline::get_sort_index(){
//     return sort_index;
//}


// TODO change this function to not use rotated points and to consider 2 polynomials
std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> Spline::along(double progress, double point_index, int precision){
    
    std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> ret;


    double len = this->calculateLength();


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

// MIGHT NEED FOR SPLINE ALONG?!
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

// MIGHT NEED FOR SPLINE ALONG?!
Eigen::VectorXd get_translation_vector(Eigen::MatrixXd& group){
    Eigen::Vector2d ret;
    ret(0) = group(0, 0);
    ret(1) = group(1, 0);
    return ret;
}

// MIGHT NEED FOR SPLINE ALONG?!
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

// generates polynomial from points
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

// arclength helper
double arclength_f(double x, void* params){
    
    // polynomial p = *(polynomial*)params;
    std::pair<polynomial, polynomial> p = *(std::pair<polynomial, polynomial>*)params;
    double dx_dt = poly_eval(p.first(),x);
    double dy_dt = poly_eval(p.second(),x);
    return sqrt(dy_dt*dy_dt+dx_dt*dx_dt);
}


// integrating with gsl to get arclength
double arclength(polynomial p1, polynomial p2, double low_bound,double up_bound){

    gsl_function F;
    F.function = &arclength_f;
    F.params = &std::make_pair(p1, p2);

    double result, error;
    size_t neval;
    // gsl_integration_workspace *w 
    //      = gsl_integration_workspace_alloc (100000);

    gsl_integration_qng (&F, low_bound, up_bound, 1, 1e-1, &result, &error, &neval);
    // gsl_integration_workspace_free(w); 

    return result;

}

// given a set of points, returns a vector of splines and a vector of cumulative lengths
std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(rclcpp::Logger logger, Eigen::MatrixXd& points, int points_per_spline,bool loop){
    int n = res.cols(); // number of points passed in

    // construct 2 new matrices of size 2xn (one for index & x, one for index & y)
    Eigen::VectorXd xRow = points.row(0);
    Eigen::VectorXd yRow = points.row(1);

    // NOTE! uses an ordering for the points to generate parametrized polynomials
    // ordering based on vector indices of vector containing points

    // vector of indices (our 3rd dimension, could in future change to time)
    Eigen::VectorXd vector_n = Eigen::VectorXd::LinSpaced(n, 0, n - 1);

    Eigen::Matrix<double, 2, n> xMatrix;
    xMatrix.row(0) = vector_n;
    xMatrix.row(1) = xRow;

    Eigen::Matrix<double, 2, n> yMatrix;
    yMatrix.row(0) = vector_n;
    yMatrix.row(1) = yRow;
    

    std::vector<Spline> splines;

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

    std::vector<double> cumsum;

    for(int i=0; i<group_numbers; i++){

        Eigen::MatrixXd group_x(2, points_per_spline);
        Eigen::MatrixXd group_y(2, points_per_spline);
        for(int k = 0; k < group_x.cols(); k++) {
            for (int j = 0; j < 2; j++) {
                group_x(j, k) = xMatrix(j, i*shift + k);
                group_y(j, k) = yMatrix(j, i*shift + k);
                // if (j==1) RCLCPP_INFO(logger, "point %d is (%f, %f)\n", k, group(0, k), group(1,k));
            }
        }

        // polynomial interpolation_poly = lagrange_gen(rotated_points);
        polynomial poly_x = lagrange_gen(group_x);
        polynomial poly_y = lagrange_gen(group_y);

        Spline spline = Spline(poly_x, poly_y, group_x, group_y);
        splines.emplace_back(spline);

        if (i == 0) {
            // RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
            // RCLCPP_INFO(logger, "spline derivative is %f + %fx + %fx^2 + %fx^3\n", spline.first_der.nums(0), spline.first_der.nums(1), spline.first_der.nums(2), spline.first_der.nums(3));
            cumsum.push_back(splines[0].calculateLength());
        } else {
            // RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
            cumsum.push_back(cumsum.back()+splines[0].calculateLength());
        }

    }

    return std::make_pair(splines, cumsum);
    
}

// @TODO: searchSorted function
/**
 * @brief Finds the indices at which new_vals should be injected into old_vals to maintain sorted order,
 * and clamps the largest possible index to old_vals.size() - 1
 * 
 * This function is a combination of torch.searchsorted and torch.clampmax, using old_vals.size() - 1 as 
 * the max
 * 
 * @param old_vals the original array
 * @param new_vals the new values to inject into the original array
 * @return the indices at which new values should be injected to maintain sorted order
*/
std::vector<int> inject_clamped(std::vector<double> old_vals, std::vector<double> new_vals) {
    std::vector<int> indices;

    int old_idx = 0;
    int new_idx = 0;
    int old_len = old_vals.size();
    int new_len = new_vals.size();

    while (new_idx < new_len){
        if (old_idx >= old_len){
            indices.push_back(old_len-1); // deal with new vals that are greater than all vals in old_vals
            new_idx++;
        }
        else if (new_vals[new_idx] <= old_vals[old_idx]){
            indices.push_back(old_idx);
            new_idx++;
        }
        old_idx++;
    } 

    return indices;
}

/**
 * returns the curvature of the raceline at a given progress
 * @param progress: a sorted vector of progresses along the raceline
 * @param splines: vector of splines that make up the raceline
 * @param cumulated_lengths: vector of the cumulated lengths of the splines
 * @return curvature at progress
*/
std::vector<double> get_curvature_raceline(std::vector<double> progress,std::vector<Spline> splines, std::vector<double> cumulated_lengths) {
    // indices of splines that progress should be on 
    std::vector<int> indices = inject_clamped(cumulated_lengths, progress);

    std::vector<double> curvatures;
    for (int i = 0; i < progress.size(); i++){
        int min_x = progress[i];
        int index = indices[i];
        if (index > 0){
            min_x -= cumulated_lengths[index-1];
        }
        
        double curvature = 
        get_curvature(
            splines[index].get_first_der_x(),
            splines[index].get_first_der_y(),
            splines[index].get_second_der_x(),
            splines[index].get_second_der_y(),
            min_x
        );

        curvatures.push_back(curvature);
    }

    return curvatures;
}

/** replicating the searchSorted function from numpy
 * since target is just 1 value, we can use a binary search to find the index of the target
 * @param arr: a sorted vector of doubles
 * @param target: the value to search for
*/
int searchSorted (std::vector<double> arr, double target) {
    int left = 0;
    int right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}

/** 
 * returns the point on the raceline at a given progress
 * @param progress: a singel progress along the raceline
 * @param splines: vector of splines that make up the raceline
 * @param cumulated_lengths: vector of the cumulated lengths of the splines
 * @param previous_index: index of spline where progress begins (?????)
 * NOT USING PREVIOUS_INDEX HERE, but it is in the orginal interpolate_raceline function in python-19a
 * @param precision: number of points used to get approximation for a specific spline
 * @return tuple representing point on the raceline at progress
*/

std::pair<double, double> interpolate_raceline(double progress, std::vector<Spline> splines, 
                                        std::vector<double> cumulated_lengths, int precision = 20) {
    int index = searchSorted(cumulated_lengths, progress);

    Spline curr = splines[index];

    double delta = 0;
    
    if (index == 0) {
        delta = progress;
    } else {
        delta = progress - cumulated_lengths[index-1];
    }

    std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> result = 
        curr.along(delta, 0, precision);
    
    return result.first;
}