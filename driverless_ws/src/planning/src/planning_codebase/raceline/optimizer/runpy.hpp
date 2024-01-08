#include <Python.h>
#include <vector>
#include <string>

_IO_FILE *pythonfile;

PyObject *double_list_Py(std::vector<double> x);
// void AddToDict(PyObject* dict, std::string k,PyObject *v){

// }

std::vector<double> PyList_DoubleVec(PyObject* );

void runlto(std::vector<double> x,std::vector<double> y,std::vector<double> wl,std::vector<double> wr);
// std::vector<std::pair<double,double>> runlto(std::vector<double> x,std::vector<double> y,std::vector<double> wl,std::vector<double> wr);