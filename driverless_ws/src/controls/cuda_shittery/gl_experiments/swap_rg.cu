#include <cassert>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <cuda_utils.cuh>

constexpr dim3 block_dim {32, 32, 1};

__global__ void view_tex(cudaTextureObject_t texture) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        const float4 col = tex2D<float4>(texture, 0.5, 0.5);
        printf("Color: %f %f %f %f", col.x, col.y, col.z, col.w);
    }
}

void cuda_test(GLuint rbo, uint width, uint height) {
    cudaGraphicsResource_t img_rsc;
    CUDA_CALL(cudaGraphicsGLRegisterImage(&img_rsc, rbo, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsNone));
    CUDA_CALL(cudaGraphicsMapResources(1, &img_rsc));

    cudaResourceDesc img_rsc_desc {};
    img_rsc_desc.resType = cudaResourceTypeMipmappedArray;
    CUDA_CALL(cudaGraphicsResourceGetMappedMipmappedArray(&img_rsc_desc.res.mipmap.mipmap, img_rsc));

    cudaTextureDesc img_tex_desc {};
    img_tex_desc.addressMode[0] = cudaAddressModeClamp;
    img_tex_desc.addressMode[1] = cudaAddressModeClamp;
    img_tex_desc.filterMode = cudaFilterModeLinear;
    img_tex_desc.readMode = cudaReadModeElementType;
    img_tex_desc.normalizedCoords = true;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &img_rsc_desc, &img_tex_desc, nullptr);

    view_tex<<<dim3(width / block_dim.x, height / block_dim.y, 1), block_dim>>>(tex);

    CUDA_CALL(cudaGraphicsUnmapResources(1, &img_rsc));
}