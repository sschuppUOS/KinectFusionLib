cmake_minimum_required(VERSION 3.0)

project(KinectFusionLib)

# Use modern C++
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------

## CUDA
find_package(CUDA 10.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    # Optional: Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52 --expt-relaxed-constexpr)
ENDIF ()

## OpenCV
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
# SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")
find_package(OpenCV 4.2 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)


# ------------------------------------------------
#                     Library
# ------------------------------------------------

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
file(GLOB KinectFusion_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")

include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include" "${PROJECT_BINARY_DIR}")

cuda_add_library(KinectFusion SHARED "${KinectFusion_SRCS}")

add_library(kinfulib SHARED ${KinectFusion_SRCS})
target_link_libraries(kinfulib ${OpenCV_LIBS} ${CUDA_LIBRARIES})

cuda_add_library(KinectFusion_static STATIC "${KinectFusion_SRCS}")

add_library(kinfulib_static STATIC ${KinectFusion_SRCS})
target_link_libraries(kinfulib_static ${OpenCV_LIBS} ${CUDA_LIBRARIES})


# ------------------------------------------------
#                 install Library
# ------------------------------------------------

install(TARGETS kinfulib kinfulib_static
        EXPORT kinfulibTargets
        ARCHIVE DESTINATION lib/static
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include ${CMAKE_INSTALL_INCLUDEDIR})


# ------------------------------------------------
#               install Cmake stuff
# ------------------------------------------------

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
set(SYSCONFIG_INSTALL_DIR etc/kinfulib/)

install(EXPORT kinfulibTargets
        FILE kinfulibTargets.cmake
        NAMESPACE kinfulib::
        DESTINATION ${LIB_INSTALL_DIR}/cmake/kinfulib)

configure_package_config_file(kinfulibConfig.cmake
                             ${CMAKE_CURRENT_BINARY_DIR}/kinfulibConfig.cmake
                             INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/hyperlib)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/kinfulibConfig.cmake
                                COMPATIBILITY SameMajorVersion
                                VERSION 0.0.1)

# INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kinfulibConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/kinfulibConfigVersion.cmake
#         DESTINATION ${LIB_INSTALL_DIR}/cmake/kinfulib)
