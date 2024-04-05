#include <gsl/gsl_integration.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <cmath>
//#include </opt/ros/foxy/include/rclcpp/rclcpp.hpp>
#include <rclcpp/rclcpp.hpp>

#ifndef RACELINE
#define RACELINE

const int prefered_degree = 3,overlap = 0;
struct polynomial
{
    int deg;
    Eigen::VectorXd nums;
};

polynomial poly(int deg);

polynomial poly_one();

polynomial poly_root(double root);

polynomial polyder(polynomial p);


polynomial poly_mult(polynomial a,polynomial b);

double poly_eval(polynomial a,double x);


// TODO (4/3): restructuring spline class to account for ordering of points
class Spline
{
public:
    int path_id; // what is this?
    int sort_index; // what is this?

    polynomial spl_poly_x; // original poly outputting x
    polynomial first_der_x;
    polynomial second_der_x
    
    Eigen::MatrixXd points_x; // matrix of passed in points (order & x)

    polynomial spl_poly_y; // original poly outputting y
    polynomial first_der_y;
    polynomial second_der_y;
    
    Eigen::MatrixXd points_y; // matrix of passed in points (order & y)

    // TODO: rewrite function to find length with 
    double length; // length of the spline

    polynomial get_SplPoly(){ return spl_poly;} // original polynomial
    
    void set_SplPoly(polynomial p){ // setting polynomial
        spl_poly.deg = p.deg;
        spl_poly.nums = p.nums;
        first_der = polyder(spl_poly);
        second_der = polyder(first_der);
    }


    polynomial get_first_der();
    polynomial get_second_der();

    Eigen::MatrixXd get_points();
    void set_points(Eigen::MatrixXd newpoints);

    // Eigen::MatrixXd get_rotated_points();
    // void set_rotated_points(Eigen::MatrixXd newpoints);

    // Eigen::Matrix2d get_Q();
    // void set_Q(Eigen::Matrix2d new_Q);

    // Eigen::VectorXd get_translation();
    // void set_translation(Eigen::VectorXd new_trans);

    int get_path_id();
    void set_path_id(int new_id);

    int get_sort_index();
    void set_sort_index(int new_sort);
    double calculateLength();
    
    // Eigen::MatrixXd interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1));
    
    // Eigen::MatrixXd interpolate(int number,std::pair<float,float> bounds);
    // Eigen::MatrixXd interpolate(int number, std::pair<float,float> bounds = std::make_pair(0,1));


    bool operator==(Spline const & other) const;
    bool operator<(Spline const & other) const;

    std::tuple<Eigen::VectorXd  ,double, Eigen::VectorXd  ,double> along(double progress, double point_index=0, int precision=20);
    // double getderiv(double x);

    // std::pair<double, double> along(double progress) const;

    Spline(polynomial interpolation_poly);
    Spline(polynomial interpolation_poly, polynomial first, polynomial second, int path, int sort_ind);
    // Spline(polynomial interpolation_poly,Eigen::MatrixXd points_mat,Eigen::MatrixXd rotated,Eigen::Matrix2d Q_mat, Eigen::VectorXd translation,polynomial first, polynomial second, int path, int sort_ind);
    Spline(polynomial interpolation_poly, Eigen::MatrixXd points_mat,Eigen::MatrixXd rotated,Eigen::Matrix2d Q_mat, Eigen::VectorXd translation,polynomial first, polynomial second, int path, int sort_ind, bool calcLength = false);

    Spline();
    ~Spline();
};

// new functions translated from python (path_optimization.py)
std::vector<double> get_curvature_raceline(std::vector<double> progress,std::vector<Spline> splines, std::vector<double> cumulated_lengths);
// interpolate_raceline Tuple[np.ndarray[float], Spline, float]
std::pair<double, double> interpolate_raceline(double progress, std::vector<Spline> splines, 
                                        std::vector<double> cumulated_lengths, int precision);


Eigen::Matrix2d rotation_matrix_gen(rclcpp::Logger logger,Eigen::MatrixXd& pnts);
Eigen::VectorXd get_translation_vector(Eigen::MatrixXd& group);

Eigen::MatrixXd transform_points(rclcpp::Logger logger,Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector);

Eigen::MatrixXd reverse_transform(Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector);

polynomial lagrange_gen(Eigen::MatrixXd& points);

double arclength_f(double, void* params);

double arclength(polynomial poly, double x0,double x1);

std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id =std::rand(), int points_per_spline = prefered_degree+1,bool loop = true);

#endif
