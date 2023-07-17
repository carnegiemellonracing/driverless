#include <gsl/gsl_poly.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <vector>

int prefered_degree=3,overlap =0;

struct polynomial
{
    int deg;
    gsl_vector *nums;
};

polynomial poly(int deg = 3){
    polynomial inst;
    inst.deg=deg;
    inst.nums = gsl_vector_alloc(deg+1);
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


}

polynomial poly_mult(polynomial a,polynomial b){
    polynomial mult = poly(a.deg+b.deg);

    for(int x=0;x<=a.deg;x++){
        for(int y=0;y<=b.deg;y++){
            gsl_vector_set(mult.nums,x+y,gsl_vector_get(mult.nums,x+y)+gsl_vector_get(a.nums,x)*gsl_vector_get(b.nums,y));

        }
    }
}

class Spline
{
private:
    polynomial spl_poly;
    
    gsl_matrix *points;
    gsl_matrix *rotated_points;

    gsl_matrix *Q;
    gsl_vector *translation_vector;

    polynomial first_der;
    polynomial second_der;
    

    int path_id;
    int sort_index;
    double length;

public:
    polynomial get_SplPoly(){ return spl_poly;}
    void set_SplPoly(polynomial p){
        spl_poly.deg = p.deg;
        spl_poly.nums = p.nums;
        first_der = polyder(spl_poly);
        second_der = polyder(first_der);
    }

    gsl_matrix* get_points(){return points;}
    void set_points(gsl_matrix *newpoints){points = newpoints;}

    gsl_matrix* get_rotated_points(){return rotated_points;}
    void set_points(gsl_matrix *newpoints){rotated_points = newpoints;}

    gsl_matrix* get_Q(){return Q;}
    void set_Q(gsl_matrix *new_Q){Q=new_Q;}

    gsl_vector* get_translation(){return translation_vector;}
    void set_translation(gsl_vector *new_trans){translation_vector = new_trans;}

    int get_path_id(){return path_id;}
    void set_path_id(int new_id){path_id = new_id;}

    int get_sort_index(){return sort_index;} 
    void set_sort_index(int new_sort){sort_index = new_sort;}

    double get_length(){return length;}
    
    std::vector<float> interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1)){
        interpolate(spline,number,bounds);
    }
    



    Spline(polynomial interpolation_poly,gsl_matrix *points_mat,gsl_matrix *rotated,gsl_matrix *Q_mat, gsl_vector *translation,polynomial first, polynomial second, int path, int sort_ind);
    
    
    ~Spline();
};



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

    length = arclength(interpolation_poly,gsl_matrix_get(points_mat,0,0),gsl_matrix_get(points_mat,0,points_mat->size2-1));
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

std::vector<float> interpolate(Spline spline,int number, std::pair<float,float> bounds = std::make_pair(-1,-1)){

    if(bounds.first == -1 && bounds.second == -1){
        double bound1 = gsl_matrix_get(spline.get_rotated_points(),1,0);
        // MAKE PROPER BOUND 2
        double bound2 = gsl_matrix_get(spline.get_rotated_points(),1,0);
        bounds = std::make_pair(bound1,bound2);
    }

    
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

    gsl_matrix *temp = gsl_matrix_alloc(Q->size1,Q->size2);
    for(int i=0;i<Q->size2;++i){
        gsl_matrix_set(temp,0,i,gsl_matrix_get(Q,0,i)+gsl_vector_get(get_translation_vector,0));
        gsl_matrix_set(temp,1,i,gsl_matrix_get(Q,1,i)+gsl_vector_get(get_translation_vector,1));
    }


    gsl_matrix *ret = gsl_matrix_alloc(points->size1,points->size2);
    gsl_linalg_matmult(points,temp,ret);
    gsl_matrix_free(temp);
    

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

}

std::pair<std::vector<Spline>,std::vector<int>> raceline_gen(gsl_matrix *res,int path_id,int points_per_spline = prefered_degree+1,bool loop = true){

    int n = res->size2;

    std::vector<Spline> splines;

    gsl_matrix *points=res;

    int shift = points_per_spline-1;

    int group_numbers = n/shift;


    if (loop)
        group_numbers += (int)(n % shift != 0);

    std::vector<std::vector<int>> groups;
    
    std::vector<int> lengths;
    std::vector<int> cumsum;
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
        lengths.push_back(spline.get_length());

        cumsum.push_back(cumsum.back()+spline.get_length());

    }

    return std::make_pair(splines,cumsum);



}