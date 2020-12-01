include("${CMAKE_CURRENT_LIST_DIR}/KinfulibTargets.cmake")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules")

set(KINFU_INCLUDE_DIRS
    @KINFU_INSTALL_INCLUDE_DIRS@
)

set(KINFU_DEFINITIONS "")
set(KINFU_LIBRARIES kinectfusion::kinfulib)
set(KINFU_STATIC_LIBRARIES
    kinectfusion::kinfulib_static
)

set(KINFU_FOUND TRUE)