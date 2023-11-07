#include <gsl/gsl_integration.h>
#include <vector>
#include <Eigen/Dense>
#include<Eigen/Core>
#include <vector>
#include <tuple>
#include <cmath>


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

class Spline
{
private:
    polynomial spl_poly;
    
    Eigen::MatrixXd points;
    Eigen::MatrixXd rotated_points;

    Eigen::MatrixXd Q;
    Eigen::VectorXd translation_vector;

    polynomial first_der;
    polynomial second_der;
    

    int path_id;
    int sort_index;

public:
    polynomial get_SplPoly(){ return spl_poly;}
    void set_SplPoly(polynomial p){
        spl_poly.deg = p.deg;
        spl_poly.nums = p.nums;
        first_der = polyder(spl_poly);
        second_der = polyder(first_der);
    }


    polynomial get_first_der();
    polynomial get_second_der();

    Eigen::MatrixXd& get_points();
    void set_points(Eigen::MatrixXd newpoints);

    Eigen::MatrixXd& get_rotated_points();
    void set_rotated_points(Eigen::MatrixXd newpoints);

    Eigen::MatrixXd& get_Q();
    void set_Q(Eigen::MatrixXd new_Q);

    Eigen::VectorXd&  get_translation();
    void set_translation(Eigen::VectorXd new_trans);

    int get_path_id();
    void set_path_id(int new_id);

    int get_sort_index();
    void set_sort_index(int new_sort);
    double length();
    
    // Eigen::MatrixXd interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1));
    
    Eigen::MatrixXd& interpolate(int number,std::pair<float,float> bounds);

    bool operator==(Spline const & other) const;
    bool operator<(Spline const & other) const;

    std::tuple<Eigen::VectorXd  ,double, Eigen::VectorXd  ,double> along(double progress, double point_index=0, int precision=20);
    double getderiv(double x);

    std::pair<double, double> along(double progress) const;

    Spline(polynomial interpolation_poly,Eigen::MatrixXd& points_mat,Eigen::MatrixXd& rotated,Eigen::MatrixXd& Q_mat, Eigen::VectorXd& translation,polynomial first, polynomial second, int path, int sort_ind);

    Spline() {};

    ~Spline();
};



Eigen::MatrixXd& interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1));

Eigen::MatrixXd& rotation_matrix_gen(Eigen::MatrixXd pnts);
Eigen::VectorXd& get_translation_vector(Eigen::MatrixXd group);

Eigen::MatrixXd& transform_points(Eigen::MatrixXd points, Eigen::MatrixXd Q, Eigen::VectorXd get_translation_vector);

Eigen::MatrixXd& reverse_transform(Eigen::MatrixXd points, Eigen::MatrixXd Q, Eigen::VectorXd get_translation_vector);

polynomial lagrange_gen(Eigen::MatrixXd   points);

double arclength_f(double, void* params);

double arclength(polynomial poly, double x0,double x1);

std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(Eigen::MatrixXd& res,int path_id =std::rand() ,int points_per_spline = prefered_degree+1,bool loop = true);

#endif
