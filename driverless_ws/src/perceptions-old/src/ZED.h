#include <stdio.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

class ZEDSDK{
    public:
        void open();
        void close();
        void set_runtime_params();
}