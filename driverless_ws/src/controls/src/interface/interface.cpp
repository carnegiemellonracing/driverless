#include "interface.hpp"
#include "cpu_environment.hpp"

namespace controls {
    namespace interface {
        static std::unique_ptr<environment> environment::create_environment(device dev) {
            switch (dev) {
                case device::cpu:
                    return std::make_shared<environment>()
            }
        }
    }
}