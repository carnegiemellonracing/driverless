#include <gsl/gsl_integration.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
#include <tuple>
#include <cmath>
#include <rclcpp/rclcpp.hpp>

#ifndef RACELINE
#define RACELINE

// TODO make separate file for poly things here and in frenet

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

double get_curvature(polynomial poly_der_1, polynomial poly_der_2, double min_x);

class Spline
{
public:
    int path_id;
    int sort_index;

    polynomial spl_poly;
    polynomial first_der;
    polynomial second_der;
    polynomial third_der; // TODO add third der constructor
    
    Eigen::MatrixXd points;
    // Eigen::MatrixXd rotated_points;

    // Eigen::Matrix2d Q;
    // Eigen::VectorXd translation_vector;
    double length;


    polynomial get_SplPoly(){ return spl_poly;}
    void set_SplPoly(polynomial p){
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

    // Eigen::VectorXd get_translation();
    // void set_translation(Eigen::VectorXd new_trans);

    int get_path_id();
    void set_path_id(int new_id);

    int get_sort_index();
    void set_sort_index(int new_sort);
    double calculateLength();
    
    // Eigen::MatrixXd interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1));
    
    // Eigen::MatrixXd interpolate(int number,std::pair<float,float> bounds);
    Eigen::MatrixXd interpolate(int number, std::pair<float,float> bounds = std::make_pair(0,1));


    bool operator==(Spline const & other) const;
    bool operator<(Spline const & other) const;

    double getderiv(double x);

    // std::pair<double, double> along(double progress) const;

    Spline(polynomial interpolation_poly);
    Spline(polynomial interpolation_poly, polynomial first, polynomial second, polynomial third, int path, int sort_ind);
    // Spline(polynomial interpolation_poly,Eigen::MatrixXd points_mat,Eigen::MatrixXd rotated,Eigen::Matrix2d Q_mat, Eigen::VectorXd translation,polynomial first, polynomial second, int path, int sort_ind);
    Spline(polynomial interpolation_poly, Eigen::MatrixXd points_mat,polynomial first, polynomial second, polynomial third, int path, int sort_ind, bool calcLength = false);

    Spline();
    ~Spline();
};

class ParameterizedSpline
{
public:
    Spline spline_x;
    Spline spline_y;

    ParameterizedSpline(Spline spline_x, Spline spline_y);

    double get_first_der(double t);
    double get_second_der(double t);
    double get_third_der(double t);
};

double arclength_f(double, void* params);

double arclength(std::pair<polynomial, polynomial> poly_der, double x0,double x1);

std::pair<std::vector<ParameterizedSpline>,std::vector<double>> parameterized_spline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id, int points_per_spline,bool loop);

std::pair<std::vector<ParameterizedSpline>,std::vector<double>> make_splines_vector(std::vector<std::pair<double,double>> points);

#endif
