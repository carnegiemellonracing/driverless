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
    // std::cout << "poly eval" << std::endl;
    // std::cout << a.nums << std::endl;
    // std::cout << x << std::endl;
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

/*
 * @brief Gets the sign of the concavity using the 2nd derivative at input_x
 */
Concavity get_concavity_sign(polynomial poly_der_2, double input_x) {
    double value = poly_eval(poly_der_2, input_x);

    if (value < -STRAIGHT_CONCAVITY_TH) {
        return Concavity::NEG;
    } else if (value > STRAIGHT_CONCAVITY_TH) {
        return Concavity::POS;
    } else {
        return Concavity::STRAIGHT;
    }

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
{

}

Spline::~Spline()
{
    // ~spl_poly();
    // polynomial first_der;
    // polynomial second_der;
    
    //No need for this function in Eigen as it frees memory itself

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

// TODO finish this
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


// /**
//  * @TODO: fix reverse transform
//  * @TODO: even though guesses make sense, final interpolated points are still all the same
//  * @TODO: update get_curvature_raceline to use interpolation
//  */
// std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> Spline::along(double progress, double point_index, int precision){
    
//     // std::cout << "progress: " << progress << std::endl;
   
//     std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> ret;


//     double len = this->calculateLength();


//     double first_x = this->get_rotated_points()(0,0);
//     // std::cout << "first x: " << first_x << std::endl;
//     double last_x = this->get_rotated_points()(0,this->get_rotated_points().cols()-1);

//     //std::cout << "SPLINE ALONG 1" << std::endl;

//     double delta = last_x - first_x;

//     std::pair<double,double> boundaries = std::make_pair(first_x,last_x);
//     int ratio = progress / len + 1;

//     if (ratio >= 2){
//         double x = first_x + delta*ratio;
//         double shoot = arclength(this->first_der,first_x,x);
        
//         double lower_bound = first_x + delta * (ratio - 1);
//         double upper_bound =  first_x + delta * ratio;

//         if (shoot < progress){
//             while (shoot < progress){
//                 // std::cout << "." << std::endl;
//                 lower_bound = x;
//                 //  add approximately one spline length to shoot
//                 shoot = arclength(this->first_der,first_x,x+delta);
//                 x=x+delta;
//             }
//             upper_bound = x; // upper bound is direct first overshoot (separated by delta from the lower bound)
//         }
//         else if (shoot >= progress){ // equality not very important
//             while (shoot >= progress){
//                 //std::cout << ", " << shoot << " " << progress << std::endl;
//                 upper_bound = x;
//                 // # remove approximately one splien length to shoot
//                 shoot = arclength(this->first_der,first_x,x - delta);
//                 x -= delta;
//             }
//             lower_bound = x; // lower bound is direct first undershoot (separated by delta from the upper bound)
//         }    
//         std::pair<double,double> boundaries = std::make_pair(lower_bound, upper_bound);
        
//     }
//     //  Perform a more precise search between the two computed bounds
//     //std::cout << "SPLINE ALONG 2" << std::endl;

//         std::vector<double> guesses;
//         guesses.resize(precision+1);
//         for(int i=0;i<=precision;i++){
//             guesses[i] = (boundaries.first*i + boundaries.second*(precision-i))/precision;
//         }


//         // std::cout << "number of guesses: " << guesses.size() << std::endl;
//         // for (double guess : guesses){
//         //     std::cout << guess << std::endl;
//         // }

//         //std::cout << "SPLINE ALONG 3" << std::endl;

//         //  Evaluate progress along the (extrapolated) spline
//         //  As arclength is expensive and cannot take multiple points
//         //  at the same time, it is faster to use a for loop
//         double past = double(INT64_MAX), best_guess = 0.1, best_length = 0;
//         int i = 0;
//         for (double guess : guesses){
//             // std::cout << "guess " << i << ": "<< guess << std::endl;
//             // std::cout << "first x " << i << ": "<< first_x << std::endl;
//             // std::cout << "poly " << i << ": "<< this->spl_poly.nums << std::endl;
//             double guess_length = arclength(this->first_der, first_x, guess);
//             // std::cout << "progress " << progress << std::endl;
//             // std::cout << "guess length " << i << ": "<< guess_length << std::endl;
//             // std::cout << "past " << past << std::endl;
//             if (abs(progress - guess_length) > abs(progress - past)) //# if we did worst than before
//                 break;
//             best_guess = guess;
//             best_length = guess_length;
//             past = guess_length;
//             i++;
//         }
//         Eigen::VectorXd rotated_point(2);
//         rotated_point(0)=best_guess;
//         // std::cout << "best guess: " << best_guess << std::endl;

//         //std::cout << "SPLINE ALONG 4" << std::endl;
        
//         rotated_point(1)=poly_eval(this->spl_poly,best_guess);
//         // std::cout << "spline poly: " << this->spl_poly.num << std::endl;

//         //std::cout << "SPLINE ALONG 4.1" << std::endl;
        
//         Eigen::MatrixXd rotated_points(2,1);
//         //std::cout << "SPLINE ALONG 4.2" << std::endl;
//         rotated_points(0,0)=rotated_point(0);
//         //std::cout << "SPLINE ALONG 4.3" << std::endl;
//         rotated_points(1,0)=rotated_point(1);

//         // std::cout << "rotated points: " << rotated_points << std::endl;

//         //std::cout << "SPLINE ALONG 5" << std::endl;
    
//         Eigen::MatrixXd point_mat = reverse_transform(rotated_points,this->Q,this->translation_vector);
//         // std::cout << "SPLINE ALONG 5" << std::endl;

//         Eigen::VectorXd point (2);
//         point(0)=point_mat(0);
//         point(1)=point_mat(1);

//         // std::cout << "point mat: \n" << point_mat << std::endl ;

//         ret = std::make_tuple(point,best_length,rotated_point,best_guess);

//         //std::cout << "SPLINE ALONG 6" << std::endl;

//         return ret;
// }

// double Spline::getderiv(double x){
//     Eigen::MatrixXd point_x(1,2);
//     point_x(0,0)=x;
//     point_x(0,1)=0;

//     Eigen::MatrixXd gm= reverse_transform(point_x,this->Q,this->translation_vector);
//     return poly_eval(this->first_der,gm.data()[0]);



// }

// Eigen::MatrixXd Spline::interpolate(int number, std::pair<float,float> bounds){

//     if(bounds.first == -1 && bounds.second == -1){
//         double bound1 = get_rotated_points()(0,0);
//         // MAKE PROPER BOUND 2
//         double bound2 = get_rotated_points()(0,get_rotated_points().cols());
//         bounds = std::make_pair(bound1,bound2);
//     }

//     Eigen::MatrixXd points(number,2);
    
//     for(int i=0;i<number;i++){
//         double x = bounds.first+ (bounds.second-bounds.first)*(i/(number-1));
//         points(i,0)=x;
//         points(i,1)=poly_eval(get_SplPoly(),x);
//     }

    

//     Eigen::Matrix2d q = Spline::get_Q();
//     Eigen::VectorXd trans = Spline::get_translation();
// 	Eigen::MatrixXd ret= reverse_transform(points, q, trans);

//     return ret;
// }

// Eigen::Matrix2d rotation_matrix_gen(rclcpp::Logger logger,Eigen::MatrixXd& pnts){
//     Eigen::Vector2d beg; beg << pnts.col(0);
//     Eigen::Vector2d end; end << pnts.col(pnts.cols()-1);

//     Eigen::Vector2d diff = end-beg;

//     double norm = diff.norm();

//     double cos = diff(0)/norm;
//     double sin = diff(1)/norm;

//     Eigen::Matrix2d ret;
//     ret(0,0)=cos;
//     ret(1,0)=sin;
//     ret(0,1)=-1*sin;
//     ret(1,1)=cos;

//     // //RCLCPP_INFO(logger, "(sin,cos),(%f, %f)\n", sin,cos);
//     // //RCLCPP_INFO(logger, "(diff,norm),(%f, %f),%f\n", diff(0),diff(1),norm);
//     return ret;
// }

// Eigen::VectorXd get_translation_vector(Eigen::MatrixXd& group){
//     Eigen::Vector2d ret;
//     ret(0) = group(0, 0);
//     ret(1) = group(1, 0);
//     return ret;
// }

// Eigen::MatrixXd transform_points(rclcpp::Logger logger,Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector){
//     Eigen::MatrixXd temp(points.rows(),points.cols());
//         // //RCLCPP_INFO(logger, "transform points:rotation");
//         // //RCLCPP_INFO(logger, "first point is (%f, %f)\n", Q(0, 0), Q(0, 1));
//         // //RCLCPP_INFO(logger, "second point is (%f, %f)\n", Q(1, 0), Q(1, 1));
    
//     for(int i=0;i<temp.cols();++i){
//         temp(0,i)=points(0,i)-get_translation_vector(0);
//         temp(1,i)=points(1,i)-get_translation_vector(1);
//     }

//     // //RCLCPP_INFO(logger, "temp");
//     // //RCLCPP_INFO(logger, "first point is (%f, %f)\n", temp(0, 0), temp(0, 1));
//     // //RCLCPP_INFO(logger, "second point is (%f, %f)\n", temp(1, 0),temp(1, 1));


//     Eigen::Matrix2d trans = Q.transpose(); 
//     // //RCLCPP_INFO(logger, "q.trans");
//     // //RCLCPP_INFO(logger, "first point is (%f, %f)\n", trans(0, 0), trans(0, 1));
//     // //RCLCPP_INFO(logger, "second point is (%f, %f)\n", trans(1, 0), trans(1, 1));
//     Eigen::MatrixXd ret = trans * temp;  
//     // //RCLCPP_INFO(logger, "return");
//     // //RCLCPP_INFO(logger, "first point is (%f, %f)\n", ret(0, 0), ret(0, 1));
//     // //RCLCPP_INFO(logger, "second point is (%f, %f)\n", ret(1, 0), ret(1, 1));
    
//     return ret;
// }

// // TODO: use eigen functions instead of for loops
// Eigen::MatrixXd reverse_transform(Eigen::MatrixXd& points, Eigen::Matrix2d& Q, Eigen::VectorXd& get_translation_vector){
//     //std::cout << Q << std::endl;
//     Eigen::MatrixXd temp(points.rows(),points.cols()); // TODO: is it necessary to copy points here? pass by value
//     for(int i=0;i<temp.cols();++i){
//         temp(0,i)=points(0,i);
//         temp(1,i)=points(1,i);
//     }

//     Eigen::MatrixXd ret = Q*temp;

//     for(int i=0;i<ret.cols();++i){
//         ret(0,i) += get_translation_vector(0);
//         ret(1,i) += get_translation_vector(1);
//     }

//     return ret;
// }

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
    // std::cout << "deriv: " << poly_der1.nums << std::endl;

    double result, error;
    size_t neval;
    // gsl_integration_workspace *w 
    //      = gsl_integration_workspace_alloc (100000);

    gsl_integration_qng (&F, x0, x1, 1, 1e-1, &result, &error, &neval);
    // gsl_integration_workspace_free(w); 

    return result;

}

std::pair<std::vector<ParameterizedSpline>,std::vector<double>> parameterized_spline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id, int points_per_spline,bool loop){

    ////std::cout << "Number of rows:" << res.rows() << std::endl;
    ////std::cout << "Number of cols:" << res.cols() << std::endl;
    
    int n = res.cols();

    std::vector<ParameterizedSpline> splines;

    // Eigen::MatrixXd points=res;

    // TODO: make sure that the group numbers are being calculated properly
    // [2, 3, 4, 3, 5, 3, 5, ]

    int shift = points_per_spline-1; //3
    int group_numbers;

    if (shift == 1){
        group_numbers = n/shift;

        if (loop)
            group_numbers += (int)(n % shift != 0);
    }
    else{
        if (n < 4) group_numbers = 0; // NEED TO MODIFY TO 1 AND DEAL WITH FEWER THAN 4 POINTS
        else group_numbers = ((n-2)/3) + 1;

        //RCLCPP_INFO(logger, "group numbers is %d\n", group_numbers);
    }

    // for loop through group numbers
    // extra points if mod 3 != 1, in this case we take last 4 points and make spline and change the start point
    // by going back by 2 if mod3 = 0, go back by 0 if mod3 = 1, go back by 1 if mod3 = 2

    //If there are is leftover, 

    // std::vector<std::vector<int>> groups; not used anywhere else 

    std::vector<double> lengths;
    std::vector<double> cumsum;
    // lengths.resize(group_numbers);

    //RCLCPP_INFO(logger, "points:%d, group numbers: %d\n",n,group_numbers);

    int flag = 0;

    // Define the t values
    Eigen::RowVector4d t_values(0, 0.33, 0.66, 1);

    for(int i=0; i<group_numbers; i++){

        // Eigen::MatrixXd group(res,0,group_numbers*shift,2,3);
        Eigen::MatrixXd group(2, points_per_spline);

        // if last group, set flag to (0, 1, or 2) depending on mod3 as stated above
        // @TODO make remaining points wrap around the front of the track to close the loop
        if (i == (group_numbers - 1)) {
            ////std::cout << "LAST GROUP" << std::endl;
            // flag =  (n - 1) % 3;
            flag = points_per_spline - (n - i*shift);
        }

        for(int k = 0; k < group.cols(); k++) {
            for (int j = 0; j < 2; j++) {
                ////std::cout << "Curr row:" << j << std::endl;
                ////std::cout << "Curr col:" << i*shift + k - flag << std::endl;
                ////std::cout << "Curr flag:" << flag << std::endl;

                group(j, k) = res(j, i*shift + k - flag); // ERROR index out of bound error
                // if (j==1) RCLCPP_INFO(logger, "raceline point %d is (%f, %f)\n", k, group(0, k), group(1,k));
            }
        }

        // shave off overlapping points from the spline if last group for og matrix
        group = group.block(0, flag, 2, group.cols()-flag);

        // now have a 2x4 matrix of x and y, make matrices for t and x, t and y
        // make a t vector 0, 0.33, 0.66, 1

        // Create 2x4 matrices for t and x, t and y
        //TODO
        Eigen::MatrixXd t_and_x(2, 4);
        Eigen::MatrixXd t_and_y(2, 4);

        // Fill the matrices
        t_and_x.row(0) = t_values;
        t_and_x.row(1) = group.row(0); 

        t_and_y.row(0) = t_values;
        t_and_y.row(1) = group.row(1);

        // Print the matrices
        // std::cout << "Matrix for t and x:\n" << t_and_x << "\n\n";
        // std::cout << "Matrix for t and y:\n" << t_and_y << "\n";

        // not rotating here because doing parametrized spline
        polynomial interpolation_poly_x = lagrange_gen(t_and_x);
        polynomial first_der_x = polyder(interpolation_poly_x);
        polynomial second_der_x = polyder(first_der_x);
        polynomial third_der_x = polyder(second_der_x);

        polynomial interpolation_poly_y = lagrange_gen(t_and_y);
        polynomial first_der_y = polyder(interpolation_poly_y);
        polynomial second_der_y = polyder(first_der_y);
        polynomial third_der_y = polyder(second_der_y);

        lengths.emplace_back(0);

        // TODO delete spline rotated points and translation vector
        Spline spline_x = Spline(interpolation_poly_x,t_and_x,first_der_x,second_der_x,third_der_x,path_id,i);
        Spline spline_y = Spline(interpolation_poly_x,t_and_y,first_der_y,second_der_y,third_der_y,path_id,i);
        splines.emplace_back(ParameterizedSpline(spline_x, spline_y));

        // lengths.push_back(spline.calculateLength());
        if (i == 0) {
            RCLCPP_INFO(logger, "spline x is %f + %fx + %fx^2 + %fx^3\n", spline_x.spl_poly.nums(0), spline_x.spl_poly.nums(1), spline_x.spl_poly.nums(2), spline_x.spl_poly.nums(3));
            RCLCPP_INFO(logger, "spline y is %f + %fx + %fx^2 + %fx^3\n", spline_y.spl_poly.nums(0), spline_y.spl_poly.nums(1), spline_y.spl_poly.nums(2), spline_y.spl_poly.nums(3));
            // //RCLCPP_INFO(logger, "spline derivative is %f + %fx + %fx^2 + %fx^3\n", spline.first_der.nums(0), spline.first_der.nums(1), spline.first_der.nums(2), spline.first_der.nums(3));
            // TODO rewrite calcLength
            cumsum.push_back(arclength(std::make_pair(spline_x.first_der, spline_y.first_der), 0, 1));
        } else {
            RCLCPP_INFO(logger, "spline x is %f + %fx + %fx^2 + %fx^3\n", spline_x.spl_poly.nums(0), spline_x.spl_poly.nums(1), spline_x.spl_poly.nums(2), spline_x.spl_poly.nums(3));
            RCLCPP_INFO(logger, "spline y is %f + %fx + %fx^2 + %fx^3\n", spline_y.spl_poly.nums(0), spline_y.spl_poly.nums(1), spline_y.spl_poly.nums(2), spline_y.spl_poly.nums(3));
            cumsum.push_back(cumsum.back()+arclength(std::make_pair(spline_x.first_der, spline_y.first_der), 0, 1));
        }
        // std::cout << "i: " << i << std::endl;
    }

    // std::cout << "cum length: " << std::endl;
    // for (auto l : cumsum) {
    //     // std::cout << l << std::endl;
    // }

    return std::make_pair(splines, cumsum);
}

// std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(rclcpp::Logger logger, Eigen::MatrixXd& res,int path_id, int points_per_spline,bool loop){

//     ////std::cout << "Number of rows:" << res.rows() << std::endl;
//     ////std::cout << "Number of cols:" << res.cols() << std::endl;
    
//     int n = res.cols();

//     std::vector<Spline> splines;

//     // Eigen::MatrixXd points=res;

//     // TODO: make sure that the group numbers are being calculated properly
//     // [2, 3, 4, 3, 5, 3, 5, ]

//     int shift = points_per_spline-1; //3
//     int group_numbers;

//     if (shift == 1){
//         group_numbers = n/shift;

//         if (loop)
//             group_numbers += (int)(n % shift != 0);
//     }
//     else{
//         if (n < 4) group_numbers = 0; // NEED TO MODIFY TO 1 AND DEAL WITH FEWER THAN 4 POINTS
//         else group_numbers = ((n-2)/3) + 1;

//         //RCLCPP_INFO(logger, "group numbers is %d\n", group_numbers);
//     }

//     // for loop through group numbers
//     // extra points if mod 3 != 1, in this case we take last 4 points and make spline and change the start point
//     // by going back by 2 if mod3 = 0, go back by 0 if mod3 = 1, go back by 1 if mod3 = 2

//     //If there are is leftover, 

//     // std::vector<std::vector<int>> groups; not used anywhere else 

//     std::vector<double> lengths;
//     std::vector<double> cumsum;
//     // lengths.resize(group_numbers);

//     //RCLCPP_INFO(logger, "points:%d, group numbers: %d\n",n,group_numbers);

//     int flag = 0;

//     for(int i=0; i<group_numbers; i++){

//         // Eigen::MatrixXd group(res,0,group_numbers*shift,2,3);
//         Eigen::MatrixXd group(2, points_per_spline);

//         // if last group, set flag to (0, 1, or 2) depending on mod3 as stated above
//         // @TODO make remaining points wrap around the front of the track to close the loop
//         if (i == (group_numbers - 1)) {
//             ////std::cout << "LAST GROUP" << std::endl;
//             // flag =  (n - 1) % 3;
//             flag = points_per_spline - (n - i*shift);
//         }

//         for(int k = 0; k < group.cols(); k++) {
//             for (int j = 0; j < 2; j++) {
//                 ////std::cout << "Curr row:" << j << std::endl;
//                 ////std::cout << "Curr col:" << i*shift + k - flag << std::endl;
//                 ////std::cout << "Curr flag:" << flag << std::endl;

//                 group(j, k) = res(j, i*shift + k - flag); // ERROR index out of bound error
//                 // if (j==1) RCLCPP_INFO(logger, "raceline point %d is (%f, %f)\n", k, group(0, k), group(1,k));
//             }
//         }

//         ////std::cout << "Exited inner loop" << std::endl;

//         Eigen::Matrix2d Q  = rotation_matrix_gen(logger,group);
//         Eigen::VectorXd translation_vector = get_translation_vector(group);
//         Eigen::MatrixXd rotated_points = transform_points(logger,group,Q,translation_vector);

//         ////std::cout << "Checkpoint 1" << std::endl;


//         // //RCLCPP_INFO(logger, "rotation matrix\n");
//         // //RCLCPP_INFO(logger, "first point is (%f, %f)\n", Q(0, 0), Q(0, 1));
//         // //RCLCPP_INFO(logger, "second point is (%f, %f)\n", Q(1, 0), Q(1, 1));

//         // //RCLCPP_INFO(logger, "Translation vector");
//         // //RCLCPP_INFO(logger, "(%f, %f)\n", translation_vector(0, 0), translation_vector(0, 1));

//         RCLCPP_INFO(logger, "rotated_points");
//         for (int i = 0; i < rotated_points.cols(); i++) {
//             RCLCPP_INFO(logger, "point %d is (%f, %f)\n", i, rotated_points(0, i), rotated_points(1, i));
//             RCLCPP_INFO(logger, "regular point %d is (%f, %f)\n", i, group(0, i), group(1, i));

//         }
//         RCLCPP_INFO(logger, "second point is (%f, %f)\n", rotated_points(0, 1), rotated_points(1, 1));

//         // not rotating here because doing parametrized spline
//         polynomial interpolation_poly = lagrange_gen(group);

//         polynomial first_der = polyder(interpolation_poly);
//         polynomial second_der = polyder(first_der);

//         ////std::cout << "Checkpoint 2" << std::endl;

//         ////std::cout << group.rows() << group.cols()  << flag << std::endl;
//         ////std::cout << rotated_points.rows() << rotated_points.cols() << flag << std::endl;
//         // shave off overlapping points from the spline if last group for og matrix and rotated matrix
//         // group = group.block(0, flag, 2, group.cols());
//         group = group.block(0, flag, 2, group.cols()-flag);
//         ////std::cout << "Checkpoint 2.5" << std::endl;
//         rotated_points = rotated_points.block(0, flag, 2, rotated_points.cols()-flag);

//         ////std::cout << "Checkpoint 3" << std::endl;
        
//         // Spline* spline = new Spline(interpolation_poly,group,rotated_points,Q,translation_vector,first_der,second_der,path_id,i);

//         lengths.emplace_back(0);
//         // Spline spline = Spline(interpolation_poly, first_der, second_der, path_id,i);
//         Spline spline = Spline(interpolation_poly,group,rotated_points,Q,translation_vector,first_der,second_der,path_id,i);
//         splines.emplace_back(spline);

//         ////std::cout << "Checkpoint 4" << std::endl;

//         // TODO if last group, then shave off overlaping points from the spline (in points and rotated point), then get length

//         // lengths.push_back(spline.calculateLength());
//         if (i == 0) {
//             RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
//             // //RCLCPP_INFO(logger, "spline derivative is %f + %fx + %fx^2 + %fx^3\n", spline.first_der.nums(0), spline.first_der.nums(1), spline.first_der.nums(2), spline.first_der.nums(3));
//             cumsum.push_back(splines[0].calculateLength());
//         } else {
//             RCLCPP_INFO(logger, "spline is %f + %fx + %fx^2 + %fx^3\n", spline.spl_poly.nums(0), spline.spl_poly.nums(1), spline.spl_poly.nums(2), spline.spl_poly.nums(3));
//             cumsum.push_back(cumsum.back()+splines[i].calculateLength());
//         }
//         // std::cout << "i: " << i << std::endl;
//     }

//     // std::cout << "cum length: " << std::endl;
//     // for (auto l : cumsum) {
//     //     // std::cout << l << std::endl;
//     // }

//     return std::make_pair(splines, cumsum);
// }

/**
 * Makes a vector of splines from a vector of x-y points.
 * 
 * @param points The points to make splines from.
 * @return Vector of splines, vector of their cumulative lengths. 
 */
std::pair<std::vector<ParameterizedSpline>,std::vector<double>> make_splines_vector(std::vector<std::pair<double,double>> points) {
    Eigen::MatrixXd pointMatrix(2, points.size() + 2);
    // Eigen::MatrixXd pointMatrix(2, points.size());
    for(int i = 0; i < points.size(); i++){
        assert(i < pointMatrix.cols());
        pointMatrix(0, i) = points[i].first;
        pointMatrix(1, i) = points[i].second;
    }
    // add first point again at the end to make cycle with splines
    // uncomment with cycle tests
    pointMatrix(0, points.size()) = points[0].first;
    pointMatrix(1, points.size()) = points[0].second;
    pointMatrix(0, points.size() + 1) = points[1].first;
    pointMatrix(1, points.size() + 1) = points[1].second;

    ////std::cout << pointMatrix << std::endl;

    auto dummy_logger = rclcpp::get_logger("du");
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> res = parameterized_spline_gen(dummy_logger, pointMatrix, std::rand(), 4, false);
    return res;
}

/**
 * Finds the indices at which new_vals should be injected into old_vals to maintain sorted order,
 * and clamps the largest possible index to old_vals.size() - 1.
 * 
 * This function is a combination of torch.searchsorted and torch.clampmax, using old_vals.size() - 1 as 
 * the max.
 * 
 * @param old_vals The original array.
 * @param new_vals The new values to inject into the original array.
 * 
 * @return The indices at which new values should be injected to maintain sorted order.
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

// /**
//  * TODO: get_curvature_raceline will return the sign of the concavity of the current chunk
//  * Returns the curvature of the raceline at a given progress.
//  * 
//  * @param progress A sorted vector of progresses along the raceline. (Progress in meters)
//  * @param splines A vector of splines that make up the raceline.
//  * @param cumulated_lengths A vector of the cumulated lengths of the splines. (meters)
//  * 
//  * @return The curvature at the given progress.
//  */
// Concavity get_curvature_raceline(std::vector<double> progress, std::vector<Spline> splines, std::vector<double> cumulated_lengths) {
//     //TODO: progress and splines currently are singletons, they should be just doubles
//     // We have these as vectors because inject_clamped expects vectors

//     // indices of splines that progress should be on 
//     /* 
//      * 3 components; modifying the components for each chunk
//      * 
//      * 1.) Each chunk has an accumulator for curvature average
//      * - Stores information about the chunk's "curvature" (in this case, identified by the concavity)
//      * 2.) Function that updates the curvature
//      * 3.) Function to check if we need to create a new chunk; happens if the signs change
//      */
//     std::vector<int> indices = inject_clamped(cumulated_lengths, progress);

//     //for (int i = 0; i < progress.size(); i++){
//     int prog = progress[0];
//     int index = indices[0];
//     if (index > 0){
//         prog -= cumulated_lengths[index-1];
//     }
        
//         //double curvature = get_curvature(
//         //    splines[index].get_first_der(),
//         //    splines[index].get_second_der(),
//         //    min_x
//         //);
//     std::pair<double, double> xy = interpolate_raceline(prog, splines, cumulated_lengths, 200);
//     Concavity cur_concavity = get_concavity_sign(splines[index].get_second_der(), xy.first);

//     std::cout << "x: " << xy.first << std::endl;
//     std::cout << "y: " << xy.second << std::endl;
//     return cur_concavity;
// }

/** 
 * Replicates the searchSorted function from numpy.
 * 
 * @param arr A sorted vector of doubles.
 * @param target The value to search for.
 * 
 * @return Returns the index of target.
 */
int searchSorted (std::vector<double> arr, double target) {
    int left = 0;
    int right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) { // TODO what about duplicates
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

// /** 
//  * Returns the x-y point on the raceline at a given progress. 
//  * 
//  * @param progress A single progress along the raceline.
//  * @param splines A vector of splines that make up the raceline.
//  * @param cumulated_lengths A vector of the cumulated lengths of the splines.
//  * @param precision A number of points used to get approximation for a specific spline.
//  * 
//  * @return A tuple representing point on the raceline at the given progress.
//  */
// std::pair<double, double> interpolate_raceline(double progress, std::vector<Spline> splines, 
//                                                std::vector<double> cumulated_lengths, int precision) {
//     // std::cout << "real progress: " << progress << std::endl;
//     int index = searchSorted(cumulated_lengths, progress) + 1; //TODO: use std::binary_search
//     //std::cout << "searchsorted" << std::endl;
//     //std::cout << index << std::endl;
//     //std::cout << splines.size() << std::endl;
//     Spline curr = splines[index]; // +1 because otherwise if less than first spline lenght, returns 0
//     double delta = 0;
//     //std::cout << "get curr and delta" << std::endl;

//     // std::cout << "cum length again: " << std::endl;
//     for (auto l : cumulated_lengths) {
//         // std::cout << l << std::endl;
//     }
    
//     if (index == 0) {
//         //std::cout << "if1" << std::endl;
//         delta = progress;
//         //std::cout << "if" << std::endl;
//     } else {
//         //std::cout << "else1" << std::endl;
//         delta = progress - cumulated_lengths[index-1];
//         // std::cout << "cumulated index: " << index << std::endl;
//         // std::cout << "cumulated length: " << cumulated_lengths[index-1] << std::endl;
//     }
//     //std::cout << "before curr along" << std::endl;
//     std::tuple<Eigen::VectorXd,double, Eigen::VectorXd,double> result =
//         curr.along(delta, 0, precision);
//     Eigen::VectorXd point = std::get<0>(result);
//     //std::cout << "after point" << std::endl;
//     return std::make_pair(point(0), point(1));
// }

