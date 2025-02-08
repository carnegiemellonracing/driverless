#include "raceline.hpp"
#include <eigen3/Eigen/Dense>
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

// Curvature at point(s) `min_x` based on 2d curvature equation
// https://mathworld.wolfram.com/Curvature.html
double get_curvature(polynomial poly_der_1, polynomial poly_der_2, double min_x) {
  return (poly_eval(poly_der_2, min_x) /
         pow(1 + pow(poly_eval(poly_der_1, min_x), 2), 3 / 2));

}

Spline::Spline(polynomial interpolation_poly) {
    this->spl_poly = interpolation_poly;
}

Spline::Spline(polynomial interpolation_poly, polynomial first, polynomial second, 
                polynomial third, int path, int sort_ind) {
    this->spl_poly=interpolation_poly;
    this->first_der = first;
    this->second_der = second;
    this->third_der = third;
    this->path_id = path_id;
    this->sort_index = sort_ind;
}


Spline::Spline(polynomial interpolation_poly, Eigen::MatrixXd points_mat, polynomial first, polynomial second, polynomial third, int path, int sort_ind, bool calcLength)
    : spl_poly(interpolation_poly),points(points_mat),first_der(first),second_der(second),third_der(third),path_id(path_id),sort_index(sort_ind)
{}

Spline::Spline(){}

Spline::~Spline(){}



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

int Spline::get_path_id(){
    return path_id;
}

int Spline::get_sort_index(){
    return sort_index;
}

ParameterizedSpline::ParameterizedSpline(Spline spline_x, Spline spline_y) {
    this->spline_x = spline_x;
    this->spline_y = spline_y;
}

// dy/dx = dy/dt / dx/dt
double ParameterizedSpline::get_first_der(double t) {
    // handle infinity
    double first_der_x = poly_eval(spline_x.first_der, t);
    if (first_der_x == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return poly_eval(spline_y.first_der, t) / first_der_x;
}

// dy2/d2x = (dy/dt / dx/dt)/dt * dt/dx = (x'y''-y'x'')/(x')^3
double ParameterizedSpline::get_second_der(double t) {
    double first_der_x = poly_eval(spline_x.first_der, t);
    if (first_der_x == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double first_der_y = poly_eval(spline_y.first_der, t);
    double second_der_x = poly_eval(spline_x.second_der, t);
    double second_der_y = poly_eval(spline_y.second_der, t);
    return (first_der_x * second_der_y - first_der_y * second_der_x) / std::pow(first_der_x, 3);
}

double ParameterizedSpline::get_third_der(double t) {
    double first_der_x = poly_eval(spline_x.first_der, t);
    if (first_der_x == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double first_der_y = poly_eval(spline_y.first_der, t);
    double second_der_x = poly_eval(spline_x.second_der, t);
    double second_der_y = poly_eval(spline_y.second_der, t);
    double third_der_x = poly_eval(spline_x.third_der, t);
    double third_der_y = poly_eval(spline_y.third_der, t);
    return ((first_der_x * first_der_x * third_der_y) - 
           (first_der_x * first_der_y * third_der_x) -
           (3 * first_der_x * second_der_x * second_der_y) + 
           (3 * first_der_y * second_der_x * second_der_x)) / std::pow(first_der_x, 5);
}

polynomial catmull_rom(const Eigen::MatrixXd& points) {
    double P0 = points(0);
    double P1 = points(1);
    double P2 = points(2);
    double P3 = points(3);

    double T1 = 0.5 * (P2 - P0);
    double T2 = 0.5 * (P3 - P1);

    Eigen::VectorXd coeffs(4);

    coeffs(3) = 2*P1 - 2*P2 + T1 + T2;
    coeffs(2) = -3*P1 + 3*P2 - 2*T1 - T2;
    coeffs(1) = T1;
    coeffs(0) = P1;

    polynomial spline_poly = poly(3);
    spline_poly.nums = coeffs;

    return spline_poly;
}

double arclength_f(double t, void* params){
    polynomial px = (*(std::pair<polynomial, polynomial>*)params).first;
    polynomial py = (*(std::pair<polynomial, polynomial>*)params).second;
    double x = poly_eval(px,t);
    double y = poly_eval(py,t);
    return sqrt(x*x+y*y);
}


// CHECK CORRECTNESS
double arclength(std::pair<polynomial, polynomial> poly_der, double x0,double x1){

    gsl_function F;
    F.function = &arclength_f;
    F.params = &poly_der;

    double result, error;
    size_t neval;
    // gsl_integration_workspace *w 
    //      = gsl_integration_workspace_alloc (100000);

    gsl_integration_qng (&F, x0, x1, 1, 1e-1, &result, &error, &neval);
    // gsl_integration_workspace_free(w); 

    return result;

}

std::pair<std::vector<ParameterizedSpline>,std::vector<double>> parameterized_spline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id, int points_per_spline,bool loop){    
    std::vector<ParameterizedSpline> splines;

    std::vector<double> lengths;
    std::vector<double> cumsum;

    // assume add last point to beginning, and first point to end in make_splines_vector

    for(int i=0; i < res.cols()-3; i++){
        // Eigen::MatrixXd group(res,0,group_numbers*shift,2,3);
        Eigen::MatrixXd group(3, 4);

        for(int k = 0; k < group.cols(); k++) {
            for (int j = 0; j < group.rows(); j++) {
                group(j, k) = res(j, i + k); 
            }
        }

        // not rotating here because doing parametrized spline
        polynomial interpolation_poly_x = catmull_rom(group.row(0));
        polynomial first_der_x = polyder(interpolation_poly_x);
        polynomial second_der_x = polyder(first_der_x);
        polynomial third_der_x = polyder(second_der_x);

        polynomial interpolation_poly_y = catmull_rom(group.row(1));
        polynomial first_der_y = polyder(interpolation_poly_y);
        polynomial second_der_y = polyder(first_der_y);
        polynomial third_der_y = polyder(second_der_y);

        lengths.emplace_back(0);

        // TODO delete spline rotated points and translation vector
        Spline spline_x = Spline(interpolation_poly_x,first_der_x,second_der_x,third_der_x,path_id,i);
        Spline spline_y = Spline(interpolation_poly_y,first_der_y,second_der_y,third_der_y,path_id,i);
        ParameterizedSpline currParamSpline = ParameterizedSpline(spline_x, spline_y);
        currParamSpline.start_cone_id = group(2, 1);

        splines.emplace_back(currParamSpline);

        // lengths.push_back(spline.calculateLength());
        if (i == 0) {
            // RCLCPP_INFO(logger, "spline x is %f + %fx + %fx^2 + %fx^3\n", spline_x.spl_poly.nums(0), spline_x.spl_poly.nums(1), spline_x.spl_poly.nums(2), spline_x.spl_poly.nums(3));
            // RCLCPP_INFO(logger, "spline y is %f + %fx + %fx^2 + %fx^3\n", spline_y.spl_poly.nums(0), spline_y.spl_poly.nums(1), spline_y.spl_poly.nums(2), spline_y.spl_poly.nums(3));
            // //RCLCPP_INFO(logger, "spline derivative is %f + %fx + %fx^2 + %fx^3\n", spline.first_der.nums(0), spline.first_der.nums(1), spline.first_der.nums(2), spline.first_der.nums(3));
            cumsum.push_back(arclength(std::make_pair(spline_x.first_der, spline_y.first_der), 0, 1));
        } else {
            // RCLCPP_INFO(logger, "spline x is %f + %fx + %fx^2 + %fx^3\n", spline_x.spl_poly.nums(0), spline_x.spl_poly.nums(1), spline_x.spl_poly.nums(2), spline_x.spl_poly.nums(3));
            // RCLCPP_INFO(logger, "spline y is %f + %fx + %fx^2 + %fx^3\n", spline_y.spl_poly.nums(0), spline_y.spl_poly.nums(1), spline_y.spl_poly.nums(2), spline_y.spl_poly.nums(3));
            cumsum.push_back(cumsum.back()+arclength(std::make_pair(spline_x.first_der, spline_y.first_der), 0, 1));
        }
    }
    return std::make_pair(splines, cumsum);
}

/**
 * Makes a vector of splines from a vector of x-y points.
 * 
 * @param points The points to make splines from.
 * @return Vector of splines, vector of their cumulative lengths. 
 */
std::pair<std::vector<ParameterizedSpline>,std::vector<double>> make_splines_vector(std::vector<std::tuple<double,double,int>> points) {
    Eigen::MatrixXd pointMatrix(3, points.size() + 3);
    // Eigen::MatrixXd pointMatrix(2, points.size());
    for(int i = 0; i < points.size(); i++){
        assert((i + 1) < pointMatrix.cols());
        pointMatrix(0, i + 1) = std::get<0>(points[i]);
        pointMatrix(1, i + 1) = std::get<1>(points[i]);
        pointMatrix(2, i + 1) = std::get<2>(points[i]);
    }
    // add first point at end, add last point at beginning
    // uncomment with cycle tests
    pointMatrix(0, 0) = std::get<0>(points[points.size()-1]);
    pointMatrix(1, 0) = std::get<1>(points[points.size()-1]);
    pointMatrix(2, 0) = std::get<2>(points[points.size()-1]);
    pointMatrix(0, points.size() + 1) = std::get<0>(points[0]);
    pointMatrix(1, points.size() + 1) = std::get<1>(points[0]);
    pointMatrix(2, points.size() + 1) = std::get<2>(points[0]);
    pointMatrix(0, points.size() + 2) = std::get<0>(points[1]);
    pointMatrix(1, points.size() + 2) = std::get<1>(points[1]);
    pointMatrix(2, points.size() + 2) = std::get<2>(points[1]);

    auto dummy_logger = rclcpp::get_logger("du");
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> res = parameterized_spline_gen(dummy_logger, pointMatrix, std::rand(), 4, false);
    return res;
}