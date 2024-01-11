#include "runpy.hpp"

PyObject *double_list_Py(std::vector<double> x){
    PyObject *ret = PyList_New(x.size());
    for(long i=0;i<x.size();i++)
    PyList_SetItem(ret,i,PyFloat_FromDouble(x[i]));
    return ret;
}

// void AddToDict(PyObject* dict, std::string k,PyObject *v){

// }

std::vector<double> PyList_DoubleVec(PyObject *DL,int i){
    PyObject* L=PyList_GetItem(DL,i);
    ssize_t sze = PyList_GET_SIZE(L);
    std::vector<double> ret;
    for(ssize_t i =0;i<sze;++i){
        ret.push_back(PyFloat_AsDouble(PyList_GetItem(L,i)));
        
    }
    return ret;

};


/*

std::vector<std::pair<double,double>> runlto(std::vector<double> x,std::vector<double> y,std::vector<double> wl,std::vector<double> wr){

        Py_Initialize();
        PyRun_SimpleString("import os\n");
        
    PyObject *obj = Py_BuildValue("s", "/optimizer/lto/global_racetrajectory_optimization/main_globaltraj.py");
    FILE *file = _Py_fopen_obj(obj, "r+");
    // PyRun_SimpleString("import os\n");
    // PyRun_SimpleString("import moduler\n");
    // PyRun_SimpleString("print (os.getcwd())");



    // PyRun_SimpleFile(file,"pyshit.py");
    // PyRun_StringFlags(myString,0,NULL,NULL,NULL);
    



    PyObject *py_x,*py_y,*py_wl,*py_wr;
    py_x=double_list_Py(x);
    py_y=double_list_Py(y);
    py_wl=double_list_Py(wl);
    py_wr=double_list_Py(wr);

    PyObject *globals =PyDict_New();
    PyDict_SetItemString(globals,"x",py_x);
    PyDict_SetItemString(globals,"y",py_y);
    PyDict_SetItemString(globals,"wl",py_wl);
    PyDict_SetItemString(globals,"wr",py_wr);

    PyObject *ret=  PyRun_SimpleFile(file,"pyshit.py");

    std::vector<double> xpts =PyList_DoubleVec(ret,0);
    std::vector<double> ypts =PyList_DoubleVec(ret,1);

    assert(xpts.size()==ypts.size());


    std::vector<std::pair<double,double>> values;
    for(int i=0;i<xpts.size();i++){
        values.push_back(std::make_pair(xpts[i],ypts[i]));
    }
    return values;
    
    
    Py_Finalize();

    // PyList_DoubleVec(y_pts);

}
*/


void runlto(std::vector<double> x,std::vector<double> y,std::vector<double> wl,std::vector<double> wr){

        Py_Initialize();
        PyRun_SimpleString("import os\n");
        
    PyObject *obj = Py_BuildValue("s", "/optimizer/lto/global_racetrajectory_optimization/main_globaltraj.py");
    FILE *file = _Py_fopen_obj(obj, "r+");
    // PyRun_SimpleString("import os\n");
    // PyRun_SimpleString("import moduler\n");
    // PyRun_SimpleString("print (os.getcwd())");



    // PyRun_SimpleFile(file,"pyshit.py");
    // PyRun_StringFlags(myString,0,NULL,NULL,NULL);
    



    PyObject *py_x,*py_y,*py_wl,*py_wr;
    py_x=double_list_Py(x);
    py_y=double_list_Py(y);
    py_wl=double_list_Py(wl);
    py_wr=double_list_Py(wr);

    PyObject *globals =PyDict_New();
    PyObject *local =PyDict_New();
    PyDict_SetItemString(globals,"x",py_x);
    PyDict_SetItemString(globals,"y",py_y);
    PyDict_SetItemString(globals,"wl",py_wl);
    PyDict_SetItemString(globals,"wr",py_wr);

    PyRun_File(file,"stuff",0,globals,local);
    
    
    Py_Finalize();

    // PyList_DoubleVec(y_pts);

}