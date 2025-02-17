#include <iostream>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include "environment.hpp"
#include "env_node.cpp"
#include "env_node.hpp"

namespace controls{


    class RacecarObservationSpace : public ObservationSpace {
        public: 
            RacecarObservationSpace(controls::env::EnvNode environmentDynamics) : ObservationSpace()
            {
                
            }
        private:

    };

}




