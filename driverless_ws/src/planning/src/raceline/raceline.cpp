#include <gsl/gsl_poly.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <vector>
#include "raceline.hpp"
// #include "random.h"

polynomial poly(int deg = 3){
    polynomial inst;
    inst.deg=deg;
    inst.nums = gsl_vector_alloc(deg+1);
	return inst;
}

polynomial poly_one(){
    polynomial p = poly(1);
    gsl_vector_set(p.nums,0,1);
    return p;
}

polynomial poly_root(double root){
    polynomial p = poly(1);
    gsl_vector_set(p.nums,0,-root);
    gsl_vector_set(p.nums,1,1);
    return p;
}

polynomial polyder(polynomial p){
    if (p.deg ==0) return poly(0);
    polynomial der = poly(p.deg-1);
    for(int i=0;i<p.deg;i++){
        double coef = gsl_vector_get(p.nums,i+1)*(i+1);
        gsl_vector_set(der.nums,i,coef);
    }

	return der;
}

polynomial poly_mult(polynomial a,polynomial b){
    polynomial mult = poly(a.deg+b.deg);

    for(int x=0;x<=a.deg;x++){
        for(int y=0;y<=b.deg;y++){
            gsl_vector_set(mult.nums,x+y,gsl_vector_get(mult.nums,x+y)+gsl_vector_get(a.nums,x)*gsl_vector_get(b.nums,y));

        }
    }
	return mult;
}

Spline::Spline(polynomial interpolation_poly,gsl_matrix *points_mat,gsl_matrix *rotated,gsl_matrix *Q_mat, gsl_vector *translation,polynomial first, polynomial second, int path, int sort_ind)
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
    gsl_vector_free(spl_poly.nums);
    gsl_vector_free(first_der.nums);
    gsl_vector_free(second_der.nums);
    gsl_vector_free(translation_vector);
    gsl_matrix_free(Q);
    gsl_matrix_free(points);
    gsl_matrix_free(rotated_points);

}

double Spline::length(){
    return arclength(this->spl_poly,gsl_matrix_get(this->rotated_points,0,0),gsl_matrix_get(this->rotated_points,0,this->rotated_points->size2-1));
}

// gsl_matrix* Spline::interpolate(int number, std::pair<float, float> bounds){
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

gsl_matrix* Spline::get_points(){
    return points;}

gsl_matrix* Spline::get_rotated_points(){
    return rotated_points;
}

gsl_matrix* Spline::get_Q(){
    return Q;
}

gsl_vector* Spline::get_translation(){
    return translation_vector;
}

int Spline::get_path_id(){
    return path_id;
}

int Spline::get_sort_index(){
    return sort_index;
}




std::tuple<gsl_vector*,double, gsl_vector*,double> Spline::along(double progress, double point_index, int precision){
    std::tuple<gsl_vector*,double, gsl_vector*,double> ret;


    double len = this->length();


    double first_x = gsl_matrix_get(this->get_rotated_points(),0,0);
    double last_x = gsl_matrix_get(this->get_rotated_points(),0,this->get_rotated_points()->size2);

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
        gsl_vector *rotated_point = gsl_vector_alloc(2);
        gsl_vector_set(rotated_point,0,best_guess);
        gsl_vector_set(rotated_point,1,gsl_poly_eval(this->spl_poly.nums->data,this->spl_poly.deg,best_guess));
        
        gsl_matrix *rotated_points = gsl_matrix_alloc(2,1);
        gsl_matrix_set(rotated_points,0,0,rotated_point->data[0]);
        gsl_matrix_set(rotated_points,0,1,rotated_point->data[1]);
        
        gsl_matrix *point_mat =reverse_transform(rotated_points,this->Q,this->translation_vector);
        
        gsl_vector *point = gsl_vector_alloc(2);
        gsl_vector_set(point,0,point_mat->data[0]);
        gsl_vector_set(point,0,point_mat->data[1]);

        ret = std::make_tuple(point,best_length,rotated_point,best_guess);

        return ret;
}

double Spline::getderiv(double x){
    gsl_matrix *point_x = gsl_matrix_alloc(1,2);
    gsl_matrix_set(point_x,0,0,x);
    gsl_matrix_set(point_x,0,1,0);

    gsl_matrix *gm= reverse_transform(point_x,this->Q,this->translation_vector);

    return gsl_poly_eval(this->first_der.nums->data,this->first_der.deg,gm->data[0]);



}

gsl_matrix* Spline::interpolate(int number, std::pair<float,float> bounds){

    if(bounds.first == -1 && bounds.second == -1){
        double bound1 = gsl_matrix_get(get_rotated_points(),0,0);
        // MAKE PROPER BOUND 2
        double bound2 = gsl_matrix_get(get_rotated_points(),0,get_rotated_points()->size2);
        bounds = std::make_pair(bound1,bound2);
    }

    gsl_matrix *points = gsl_matrix_alloc(number,2);
    
    for(int i=0;i<number;i++){
        double x = bounds.first+ (bounds.second-bounds.first)*(i/(number-1));
        double y = gsl_poly_eval(get_SplPoly().nums->data,get_SplPoly().deg,x);
        gsl_matrix_set(points,i,0,x);
        gsl_matrix_set(points,i,1,y);
    }

    


	gsl_matrix *ret= reverse_transform(points,get_Q(),get_translation());

    return ret;
}

gsl_matrix* rotation_matrix_gen(gsl_matrix *pnts){
    gsl_vector *beg= gsl_vector_alloc_col_from_matrix(pnts,0);
    gsl_vector *end = gsl_vector_alloc_col_from_matrix(pnts,pnts->size2-1);

    gsl_vector_sub(end,beg);
    gsl_vector_free(beg);

    double norm = gsl_blas_dnrm2(end);

    double cos = gsl_vector_get(end,0)/norm;
    double sin = gsl_vector_get(end,1)/norm;

    gsl_vector_free(end);

    gsl_matrix *ret = gsl_matrix_alloc(2,2);
    gsl_matrix_set(ret,0,0,cos);
    gsl_matrix_set(ret,1,0,-sin);
    gsl_matrix_set(ret,0,1,sin);
    gsl_matrix_set(ret,1,1,cos);


    return ret;
}

gsl_vector *get_translation_vector(gsl_matrix *group){
    return gsl_vector_alloc_col_from_matrix(group,0);
}

gsl_matrix *transform_points(gsl_matrix *points, gsl_matrix *Q, gsl_vector *get_translation_vector){
    gsl_matrix *temp = gsl_matrix_alloc(points->size1,points->size2);
    for(int i=0;i<temp->size2;++i){
        gsl_matrix_set(temp,0,i,gsl_matrix_get(points,0,i)-gsl_vector_get(get_translation_vector,0));
        gsl_matrix_set(temp,1,i,gsl_matrix_get(points,1,i)-gsl_vector_get(get_translation_vector,1));
    }

    gsl_matrix_transpose(Q);

    gsl_matrix *ret = gsl_matrix_alloc(points->size1,points->size2);
    gsl_linalg_matmult(temp,Q,ret);
    gsl_matrix_free(temp);    
    gsl_matrix_transpose(Q);

    return ret;
}

gsl_matrix *reverse_transform(gsl_matrix *points, gsl_matrix *Q, gsl_vector *get_translation_vector){
    gsl_matrix *temp = gsl_matrix_alloc(points->size1,points->size2);
    for(int i=0;i<temp->size2;++i){
        gsl_matrix_set(temp,0,i,gsl_matrix_get(points,0,i));
        gsl_matrix_set(temp,1,i,gsl_matrix_get(points,1,i));
    }

    gsl_matrix *ret = gsl_matrix_alloc(points->size1,points->size2);
    gsl_linalg_matmult(temp,Q,ret);
    gsl_matrix_free(temp);    

    for(int i=0;i<temp->size2;++i){
        gsl_matrix_set(temp,0,i,gsl_matrix_get(points,0,i)+gsl_vector_get(get_translation_vector,0));
        gsl_matrix_set(temp,1,i,gsl_matrix_get(points,1,i)+gsl_vector_get(get_translation_vector,1));
    }

    return ret;
}

polynomial lagrange_gen(gsl_matrix* points){
    polynomial lagrange_poly = poly(3);

    for(int col = 0;col <points->size2;col++){


    }
    double x[points->size2];
    double y[points->size2];
    for(int i=0;i<points->size2;i++){
        x[i] = gsl_matrix_get(points,i,0);
        y[i] = gsl_matrix_get(points,i,1);
    }


    for(int i=0;i<points->size2;i++){
        polynomial p = poly_one();
        for(int j=0;j<points->size2;j++){
            if(j!=i){
                polynomial pr =poly_root(x[j]);
                polynomial q =poly_mult(p,pr);
                gsl_vector_free(p.nums);
                gsl_vector_free(pr.nums);
                p=q;
            }
        }
        polynomial p1 = poly_one();
        gsl_vector_set(p1.nums,0,1/gsl_poly_eval(p.nums->data,p.deg+1,x[i]));
        polynomial q = poly_mult(p1,p);
        gsl_vector_free(p.nums);
        gsl_vector_free(p1.nums);

        gsl_vector_add(lagrange_poly.nums,q.nums);
        gsl_vector_free(q.nums);
        
    }
    
    return lagrange_poly;    

}

double arclength_f(double, void* params){
    
    polynomial p = *(polynomial*)params;

    double x = gsl_poly_eval(p.nums->data,p.deg+1,x);
    return x*x+1;
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

std::pair<std::vector<Spline>,std::vector<double>> raceline_gen(gsl_matrix *res,int path_id ,int points_per_spline,bool loop){

    int n = res->size2;

    std::vector<Spline> splines;

    // gsl_matrix *points=res;

    int shift = points_per_spline-1;

    int group_numbers = n/shift;


    if (loop)
        group_numbers += (int)(n % shift != 0);

    std::vector<std::vector<int>> groups;
    
    std::vector<double> lengths;
    std::vector<double> cumsum;
    lengths.resize(group_numbers);

    for(int i=0;i<group_numbers;i++){
        gsl_matrix *group = gsl_matrix_alloc_from_matrix(res,0,group_numbers*shift,2,3);

        gsl_matrix *Q  = rotation_matrix_gen(group);
        gsl_vector *translation_vector = get_translation_vector(group);
        gsl_matrix *rotated_points = transform_points(group,Q,translation_vector);

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

