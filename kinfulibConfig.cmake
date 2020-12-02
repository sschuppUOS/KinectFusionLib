include("/usr/local/lib/cmake/kinfulib/kinfulibTargets.cmake")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules")

set(KINFULIB_INCLUDE_DIRS
    "@PROJECT_INCLUDE_DIR@"
)

set(KINFU_DEFINITIONS "")
set(KINFU_LIBRARIES kinfulib::kinfulib)
set(KINFU_STATIC_LIBRARIES
    kinfulib::kinfulib_static
)

set(KINFU_FOUND TRUE)
