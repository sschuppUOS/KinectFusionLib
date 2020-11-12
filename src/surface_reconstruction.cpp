#include <kinectfusion.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MEM_SIZE (128)//suppose we have a vector with 128 elements
#define MAX_SOURCE_SIZE (0x100000)

using namespace cv;


namespace kinectfusion {
    namespace internal {
        namespace opencl {

            cv::ocl::Context ctx;
            cv::ocl::Kernel kernel;

            void surface_reconstruction(const cv::UMat& depth_image, const cv::UMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params, const float truncation_distance,
                                        const Eigen::Matrix4f& model_view)
            {
                if(kernel.empty())
                    compile_surface_reconstruction_kernel();
                
                /** THIS WORKS
                uint length = depth_image.total() * depth_image.channels();
                cv::UMat flat = depth_image.reshape(1, length);
                if(!depth_image.isContinuous()) {
                    flat = flat.clone(); // O(N),
                }
                // flat.data is your array pointer
                auto * ptr = flat.u; // usually, its uchar*

                int i;
                const int LIST_SIZE = ptr->size/4;
                int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
                int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
                for(i = 0; i < LIST_SIZE; i++) {
                    B[i] = LIST_SIZE - i;
                }
                cl_int ret;
                // Create memory buffers on the device for each vector 
                cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                        ptr->size, NULL, &ret);
                cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                        LIST_SIZE * sizeof(int), NULL, &ret);
                cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        LIST_SIZE * sizeof(int), NULL, &ret);
                cl_mem i_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        LIST_SIZE * sizeof(int), NULL, &ret);
            
                // Copy the lists A and B to their respective memory buffers
                ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                        LIST_SIZE * sizeof(int), ptr, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
                        LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
                ret = clEnqueueWriteBuffer(command_queue, i_mem_obj, CL_TRUE, 0, 
                        LIST_SIZE * sizeof(int), ptr, 0, NULL, NULL);
            
                // Set the arguments of the kernel
                ret = clSetKernelArg(surface_reconstruction_kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
                ret = clSetKernelArg(surface_reconstruction_kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
                ret = clSetKernelArg(surface_reconstruction_kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
                ret = clSetKernelArg(surface_reconstruction_kernel, 3, sizeof(cl_mem), (void *)&i_mem_obj);
            
                // Execute the OpenCL kernel on the list
                size_t global_item_size = LIST_SIZE; // Process the entire lists
                size_t local_item_size = 64; // Divide work items into groups of 64
                ret = clEnqueueNDRangeKernel(command_queue, surface_reconstruction_kernel, 1, NULL, 
                        &global_item_size, &local_item_size, 0, NULL, NULL);
            
                // Read the memory buffer C on the device to the local variable C
                int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
                ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
                        LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
                std::cout << ret << std::endl;

                ret = clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0, 
                        LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
                std::cout << ret << std::endl;
            
                // Display the result to the screen
                for(i = 0; i < LIST_SIZE; i++)
                    printf("%d + %d = %d\n", A[i], B[i], C[i]);
            
                // Clean up
                ret = clFlush(command_queue);
                ret = clFinish(command_queue);
                ret = clReleaseMemObject(a_mem_obj);
                ret = clReleaseMemObject(b_mem_obj);
                ret = clReleaseMemObject(c_mem_obj);
                
                free(A);
                free(B);
                free(C);


                THIS WORKS
                THIS WORKS **/




                // const dim3 threads(32, 32);
                // const dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                //                   (volume.volume_size.y + threads.y - 1) / threads.y);

                // cl_int ret;
                // cl_mem color_image_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, color_image.elemSize() * sizeof(int), NULL, &ret);
                // std::cout << ret << std::endl;
                // // cl_mem dst_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, color_image.elemSize() * sizeof(int), NULL, &ret);

                // // clSetKernelArg(surface_reconstruction_kernel, 0, depth_image.elemSize(), (void *)depth_image);
                // clSetKernelArg(surface_reconstruction_kernel, 0, sizeof(cl_mem), color_image_buf);
                // // clSetKernelArg(surface_reconstruction_kernel, 2, volume.tsdf_volume.elemSize(), volume.tsdf_volume);
                // // clSetKernelArg(surface_reconstruction_kernel, 3, volume.color_volume.elemSize(), volume.color_volume);
                // // clSetKernelArg(surface_reconstruction_kernel, 4, sizeof(int3), volume.volume_size);
                // // clSetKernelArg(surface_reconstruction_kernel, 5, sizeof(float), volume.voxel_scale);
                // // clSetKernelArg(surface_reconstruction_kernel, 6, sizeof(cam_params), cam_params);
                // // clSetKernelArg(surface_reconstruction_kernel, 7, sizeof(float), truncation_distance);
                // // clSetKernelArg(surface_reconstruction_kernel, 8, model_view.block(0, 0, 3, 3).size(), model_view.block(0, 0, 3, 3));
                // // clSetKernelArg(surface_reconstruction_kernel, 9, model_view.block(0, 3, 3, 1).size(), model_view.block(0, 3, 3, 1));
                
                // size_t global_item_size = color_image.elemSize();
                // size_t local_item_size = 32;

                cv::UMat d_result(
                    2,
                    2,
                    cv::ACCESS_WRITE,
                    cv::USAGE_ALLOCATE_DEVICE_MEMORY
                );

                std::cout << "Setting kernel arguments" << std::endl;
                // std::cout << volume.volume_size << std::endl;
                kernel.set(0, depth_image);
                kernel.set(1, color_image);
                kernel.set(2, volume.tsdf_volume);
                kernel.set(3, volume.color_volume);
                kernel.set(4, volume.volume_size);
                // kernel.set(5, volume.voxel_scale);
                // kernel.set(6, cam_params);
                // kernel.set(7, truncation_distance);
                // kernel.set(8, model_view.block(0, 3, 3, 1).data());
                // kernel.set(9, model_view.block(0, 3, 3, 1).data());
                
                size_t dimThreads[3] = {
                    32,
                    32,
                    1
                };
                
                if (!kernel.run(3, dimThreads, NULL, true)) {
                    std::cerr
                        << "Failed to execute kernel. "
                        << std::endl;
                }
                // std::cout << d_result << std::endl;
                //! [Define kernel parameters and run]
            }

            void compile_surface_reconstruction_kernel() {

                if (!cv::ocl::haveOpenCL()) {
                    std::cerr << "OpenCL is not available." << std::endl;
                    for(;;);
                }
                
                // create opencl context of gpu
                if (!ctx.create(cv::ocl::Device::TYPE_GPU)) {
                    std::cerr 
                        << "Failed to create GPU context. " 
                        << std::endl;
                    for(;;);
                }
                if (!ctx.ptr())
                {
                    std::cerr << "OpenCL is not available" << std::endl;
                    for(;;);
                }
                cv::ocl::Device device = ctx.device(0);
                if (!device.compilerAvailable())
                {
                    std::cerr << "OpenCL compiler is not available" << std::endl;
                    for(;;);
                }

                FILE* source_file = fopen("/home/student/s/sschupp/KinectFusionApp/KinectFusionLib/src/opencl/surface_reconstruction.cl", "r");
                if (!source_file) {
                    fprintf(stderr, "Failed to load surface_reconstruction_kernel.\n");
                    exit(1);
                }
                char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
                size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, source_file);
                fclose( source_file );

                // // // std::cout << source_str << std::endl;
                // load_file("opencl/include/common.h");
                cv::String module_name;
                cv::ocl::ProgramSource source(module_name, "update_tsdf_kernel", source_str, "");

                cv::String errmsg;
                cv::ocl::Program program(source, "", errmsg);
                if (!errmsg.empty())
                {
                    std::cout << "Compile Error has occurred:" << std::endl << errmsg << std::endl;
                }
                //! [Compile/build OpenCL for current OpenCL device]



                // //! [Get OpenCL kernel by name]
                kernel.create("update_tsdf_kernel", program);
                if (kernel.empty())
                {
                    std::cerr << "Can't get OpenCL kernel" << std::endl;
                    for(;;);
                }
                //! [Define kernel parameters and run]


                /*
                cl_int ret = 0;
                // Load the kernel source code into the array source_str
                FILE *fp;
                char *source_str;
                size_t source_size;
            
                fp = fopen("/home/student/s/sschupp/KinectFusionApp/KinectFusionLib/src/opencl/surface_reconstruction.cl", "r");
                if (!fp) {
                    fprintf(stderr, "Failed to load surface_reconstruction_kernel.\n");
                    exit(1);
                }
                source_str = (char*)malloc(MAX_SOURCE_SIZE);
                source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
                fclose( fp );
            
                // Get platform and device information
                cl_platform_id platform_id = NULL;
                cl_device_id device_id = NULL;   
                cl_uint ret_num_devices;
                cl_uint ret_num_platforms;
                ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
                ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
                        &device_id, &ret_num_devices);
            
                // Create an OpenCL context
                context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
            
                // Create a command queue
                command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
            
                // Create a program from the kernel source
                cl_program program = clCreateProgramWithSource(context, 1, 
                        (const char **)&source_str, (const size_t *)&source_size, &ret);
            
                // Build the program
                ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
            
                // Create the OpenCL kernel
                surface_reconstruction_kernel = clCreateKernel(program, "vecAdd", &ret);

                ret = clReleaseProgram(program);**/
            }

            void surface_reconstruction_cleanup() {
                // std::cout << "surface_reconstruction_cleanup" << std::endl;
                // /* ret = */ clReleaseKernel(surface_reconstruction_kernel);
                // /* ret = */ clReleaseCommandQueue(command_queue);
                // /* ret = */ clReleaseContext(context);
            }
        }
    }
}