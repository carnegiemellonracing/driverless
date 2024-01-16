#pragma once

#define IDX_2D_PITCHED(tensor, idx, pitch) ((decltype(tensor))((char*)tensor + idx.y * pitch + idx.x * sizeof(std::remove_pointer<decltype(tensor)>::type))
#define IDX_3D_PITCHED(tensor, dims, idx, pitch) ((decltype(tensor))((char*)tensor + idx.z * dims.y * pitch + idx.y * pitch + idx.x * sizeof(std::remove_pointer<decltype(tensor)>::type)))
#define IDX_3D(tensor, dims, idx) (&tensor[idx.z * dims.y * dims.x + idx.y * dims.x + idx.x])

#define CUDA_CALL(x) (cuda_assert(x, __FILE__, __LINE__))

#define CURAND_CALL(x) (curand_assert(x, __FILE__, __LINE__))

struct cudaTensor3D {
    cudaPitchedPtr pitched_ptr;
    cudaExtent extent;
};

static const char* curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"cuda assert: %s. Location: %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static void curand_assert(curandStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"curand assert: %s. Location: %s:%d\n", curandGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}