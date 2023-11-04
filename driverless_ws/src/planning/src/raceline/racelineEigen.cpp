#include <gsl/gsl_integration.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Polynomial>
#include "racelineEigen.hpp"
// #include "random.h"

polynomial poly(int deg = 3){
    polynomial inst;
    inst.deg=deg;
    Eigen::VectorXd nums(deg+1);
    inst.nums = nums;
    
	return inst;
}

polynomial poly_one(){
    polynomial p = poly(1);
    p.nums(0)=1;
    return p;
}

polynomial poly_root(double root){
    polynomial p = poly(1);
    p.nums(0)= -root;
    p.nums(1)=1;
    return p;
}

polynomial polyder(polynomial p){
    if (p.deg ==0) return poly(0);
    polynomial der = poly(p.deg-1);
    for(int i=0;i<p.deg;i++){
        double coef = p.nums(i+1)*(i+1);
        der.nums(i)=coef;
    }

	return der;
}

polynomial polyint(polynomial p){
    polynomial antider = poly(p.deg+1);
    for(int i=0;i<p.deg;i++){
        antider.nums(i+1)=p.nums(i)/(i+1);
    }

	return antider;
}

polynomial poly_mult(polynomial a,polynomial b){
    polynomial mult = poly(a.deg+b.deg);

    for(int x=0;x<=a.deg;x++){
        for(int y=0;y<=b.deg;y++){
            mult.nums(x+y)=mult.nums(x+y)+a.nums(x)*b.nums(y);

        }
    }
	return mult;
}

Spline::Spline(polynomial interpolation_poly,Eigen::MatrixXd& points_mat,Eigen::MatrixXd& rotated,Eigen::MatrixXd& Q_mat, Eigen::MatrixXd& translation,polynomial first, polynomial second, int path, int sort_ind)
{
    spl_poly=interpolation_poly;
    points = points_mat;
    rotated_points=rotated;
    Q = Q_mat;
    translation_vector = translation;
    first_der = first;
    second_der = second;
    path = path_id;
    sort_index = sort_ind;

}

Spline::~Spline()
{
    //No need for this function in Eigen as it frees memory itself

}

double Spline::length(){
    return arclength(this->spl_poly,this->rotated_points(0,0),this->rotated_points(0,this->rotated_points.cols()-1));
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

Eigen::MatrixXd&  Spline::get_points(){
    return points;}

Eigen::MatrixXd&  Spline::get_rotated_points(){
    return rotated_points;
}

Eigen::MatrixXd&  Spline::get_Q(){
    return Q;
}

Eigen::VectorXd&  Spline::get_translation(){
    return translation_vector;
}

int Spline::get_path_id(){
    return path_id;
}

int Spline::get_sort_index(){
    return sort_index;
}




std::tuple<Eigen::VectorXd&,double, Eigen::VectorXd&,double> Spline::along(double progress, double point_index, int precision){
    std::tuple<Eigen::VectorXd&,double, Eigen::VectorXd&,double> ret;


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
        Eigen::PolynomialSolver solver(this->spl_poly.nums.data())
        rotated_point(1)=solver(best_guess);
        
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
    Eigen::Polynomial solver(this->first_der.nums.data())
    return solver(gm.data()[0]);



}

Eigen::MatrixXd& Spline::interpolate(int number, std::pair<float,float> bounds){

    if(bounds.first == -1 && bounds.second == -1){
        double bound1 = get_rotated_points()(0,0);
        // MAKE PROPER BOUND 2
        double bound2 = get_rotated_points()(0,get_rotated_points().cols());
        bounds = std::make_pair(bound1,bound2);
    }

    Eigen::MatrixXd points(number,2);
    
    for(int i=0;i<number;i++){
        double x = bounds.first+ (bounds.second-bounds.first)*(i/(number-1));
        Eigen::PolynomialSolver solver(get_SplPoly().nums.data())
        double y = solver(x);
        points(i,0)=x;
        points(i,1)=y;
    }

    


	Eigen::MatrixXd ret= reverse_transform(points,get_Q(),get_translation());

    return ret;
}

Eigen::MatrixXd rotation_matrix_gen(Eigen::MatrixXd pnts){
    Eigen::Vector2d beg; beg << pnts.col(0);
    Eigen::Vector2d end; end << pnts.col(pnts.cols()-1);

    end = end-beg;

    double norm = end.norm();

    double cos = end(0)/norm;
    double sin = end(1)/norm;

    Eigen::MatrixXd ret(2,2);
    ret(0,0)=cos;
    ret(1,0)=-sin;
    ret(0,1)=sin;
    ret(1,1)=cos;


    return ret;
}

Eigen::VectorXd& get_translation_vector(Eigen::MatrixXd group){
    Eigen::VectorXd ret; ret << group.col(0);
    return ret;
}

Eigen::MatrixXd& transform_points(Eigen::MatrixXd points, Eigen::MatrixXd Q, Eigen::VectorXd get_translation_vector){
    Eigen::MatrixXd temp(points.rows(),points.cols());
    for(int i=0;i<temp.cols();++i){
        temp(0,i)=points(0,i)-get_translation_vector(0);
        temp(1,i)=points(1,i)-get_translation_vector(1);
    }

    Q = Q.transpose();

    Eigen::MatrixXd ret (points.rows(),points.cols());
    ret = temp*Q;
    // gsl_linalg_matmult(temp,Q,ret);
    // gsl_matrix_free(temp);    
    Q =Q.transpose();

    return ret;
}

Eigen::MatrixXd& reverse_transform(Eigen::MatrixXd points, Eigen::MatrixXd Q, Eigen::VectorXd get_translation_vector){
    Eigen::MatrixXd temp(points->size1,points->size2);
    for(int i=0;i<temp->size2;++i){
        temp(0,i)=points(0,i);
        temp(1,i)=points(1,i);
    }

    Eigen::MatrixXd ret = temp*Q;

    for(int i=0;i<temp->size2;++i){
        temp(0,i)= points(0,i)+ get_translation_vector(0);
        temp(1,i)= points(1,i)+ get_translation_vector(1);
    }

    return ret;
}

polynomial lagrange_gen(Eigen::MatrixXd points){
    polynomial lagrange_poly = poly(3);

    for(int col = 0;col <points.cols();col++){


    }
    double x[points.cols()];
    double y[points.cols()];
    for(int i=0;i<points.cols();i++){
        x[i] = points(i,0);
        y[i] = points(i,1);
    }


    for(int i=0;i<points->size2;i++){
        polynomial p = poly_one();
        for(int j=0;j<points.cols();j++){
            if(j!=i){
                polynomial pr =poly_root(x[j]);
                polynomial q =poly_mult(p,pr);
                // gsl_vector_free(p.nums);
                // gsl_vector_free(pr.nums);
                p=q;
            }
        }
        polynomial p1 = poly_one();
        Eigen::PolynomialSolver solver(p.nums.data());
        p1.nums(0)=1/solver(x[i]);
        polynomial q = poly_mult(p1,p);
        // gsl_vector_free(p.nums);
        // gsl_vector_free(p1.nums);

        lagrange_poly.nums+=q.nums;
        // gsl_vector_free(q.nums);
        
    }
    
    return lagrange_poly;    

}

double arclength_f(double, void* params){
    
    polynomial p = *(polynomial*)params;
    Eigen::PolynomialSolver solver(p);
    double x = p(x);
    return math.sqrt(x*x+1);
}

double arclength(polynomial poly, double x0,double x1){

    gsl_function F;
    F.function = &arclength_f;
    F.params = &poly;

    double result, error;
    gsl_integration_workspace * w 
         = gsl_integration_workspace_alloc (1000);

    gsl_integration_qags (&F, x0, x1, 0, 1e-7, 1000,
                             w, &result, &error);  
    gsl_integration_workspace_free(w); 

    return result;

}

std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(Eigen::MatrixXd& res,int path_id ,int points_per_spline,bool loop){

    int n = res->size2;

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
        group_numbers = n;
    }
    std::vector<std::vector<int>> groups;
    
    std::vector<double> lengths;
    std::vector<double> cumsum;
    lengths.resize(group_numbers);

    for(int i=0;i<group_numbers;i++){
        Eigen::MatrixXd group(res,0,group_numbers*shift,2,3);

        Eigen::MatrixXd Q  = rotation_matrix_gen(group);
        Eigen::VectorXd translation_vector = get_translation_vector(group);
        Eigen::MatrixXd rotated_points = transform_points(group,Q,translation_vector);

        polynomial interpolation_poly = lagrange_gen(rotated_points);
        polynomial first_der = polyder(interpolation_poly);
        polynomial second_der = polyder(first_der);
        
        
        Spline spline = Spline(interpolation_poly,group,rotated_points,Q,translation_vector,first_der,second_der,path_id,i);
        
        
        splines.push_back(spline);
        lengths.push_back(spline.length());

        cumsum.push_back(cumsum.back()+spline.length());

    }

    return std::make_pair(splines,cumsum);



}

