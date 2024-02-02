### `mppi` namespace

Files:
- `mppi.hpp`: public interface (host functions _only_)
- `mppi.cuh`: declarations for `mppi.cu` (includes `mppi.hpp`)
- `mppi.cu`: implementation

Reasoning: we need a public header for the ROS node to call, hence `mppi.hpp`. But it's
also nice to separate private declarations from implementations to get a better quick look at how
the code is organized, hence `mppi.cuh`. 