std_msgs/Header header

float64[4] spl_poly_coefs
float64[4] first_der_coefs
float64[4] second_der_coefs

#dimensions are 2xn
geometry_msgs/Point[] points # z field is unused
geometry_msgs/Point[] rotated_points # z field is unused

float64[4] q # [q(0,0), q(0, 1), q(1, 0), q(1, 1)]
float64[2] translation_vector # [dx, dy]

int64 path_id
int64 sort_index
float64 length