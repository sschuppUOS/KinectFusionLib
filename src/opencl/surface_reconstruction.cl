typedef struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;
}CameraParameters;

// Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
// Vec2ida = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;

__kernel void update_tsdf_kernel(
    __global uchar* depth_image, __global uchar* color_image,
    __global uchar* tsdf_volume, __global uchar* color_volume,
    __global const int3* volume_size//, float voxel_scale,
    // CameraParameters cam_params, const float truncation_distance,
    // __global float* rotation, __global float* translation
) {

}


// __kernel void negaposi(
// 	__global uchar* input,
// 	/*int input_step, int input_offset,*/
// 	__global uchar* result//,
// 	/*int result_step, int result_offset,
// 	int height, int width*/
// ) {
//     // // Get the index of the current element to be processed
//     int x = get_global_id(0);
//     int y = get_global_id(1);
//     // // Do the operation
//     // B[x] = 1;
//     result[x] = 1;
// }