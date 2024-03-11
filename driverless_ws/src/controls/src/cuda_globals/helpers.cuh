#pragma once

#include "cuda_globals.cuh"


namespace controls {
    namespace cuda_globals {
        __device__ static void sample_curv_state(float world_pose[3], float curv_pose[3], bool& out_of_bounds) {
            const float x = world_pose[0];
            const float y = world_pose[1];
            const float yaw = world_pose[2];

            const float xmin = curv_frame_lookup_tex_info.xcenter - curv_frame_lookup_tex_info.width / 2;
            const float ymin = curv_frame_lookup_tex_info.ycenter - curv_frame_lookup_tex_info.width / 2;

            const float u = (x - xmin) / curv_frame_lookup_tex_info.width;
            const float v = (y - ymin) / curv_frame_lookup_tex_info.width;

            const float4 parallel_pose = tex2D<float4>(curv_frame_lookup_tex, u, v);
            curv_pose[0] = parallel_pose.x;
            curv_pose[1] = parallel_pose.y;
            curv_pose[2] = yaw - parallel_pose.z;

            out_of_bounds = parallel_pose.w < 0;
        }
    }
}