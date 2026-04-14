cmake_minimum_required (VERSION 3.5)
project (ACMP)

find_package(CUDA 6.0 REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.0")
set(CUDA_TOOLKIT_ROOT_NOVersion_DIR "/usr/local/cuda")

get_filename_component(CMAKE_MODULE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(MVS_WORK_PATH "${CMAKE_MODULE_DIR}" DIRECTORY)
get_filename_component(LF_WORK_PATH "${MVS_WORK_PATH}" DIRECTORY)

set(EIGEN3_INCLUDE_DIR ${LF_WORK_PATH}/ThirdPartyLib/Eigen3.3.7/include/eigen3)
set(yaml-cpp_DIR ${LF_WORK_PATH}/ThirdPartyLib/yaml-cpp/lib/cmake/yaml-cpp)

find_library(CUDA_CUDART_LIBRARY
        NAMES cudart
        PATHS
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_NOVersion_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_NOVersion_DIR}/lib
)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(OpenCV_DIR  ${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.04/Debug/lib/cmake/opencv4)
    set(libpleno_DIR ${LF_WORK_PATH}/Common/LibPleno/lib/Debug/cmake/pleno)
    set(Boost_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/debug/lib/cmake/Boost-1.71.0)
else ()
    set(OpenCV_DIR ${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.04/Release/lib/cmake/opencv4)
    set(libpleno_DIR ${LF_WORK_PATH}/Common/LibPleno/lib/Release/cmake/pleno)
    set(Boost_DIR  ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/release/lib/cmake/Boost-1.71.0)
endif ()

find_package(libpleno REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

include_directories(${OpenCV_INCLUDE_DIRS}
        ${LIBPLENO_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/include
        ${LF_WORK_PATH}/ThirdPartyLib/cmdLine
        .)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80)

link_directories(${Boost_LIBRARY_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++17)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    add_definitions(-DPLENO_DISABLE_VIEWER)
    add_definitions(-D_USE_EIGEN)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

cuda_add_executable(
        ACMP
        Common/CommonCUDA.h
        MVStereo/ACMP.h
        MVStereo/ACMP.cpp
        MVStereo/ACMP.cu
        Main.cpp
        MVStereo/LFDepthInfo.h
        DepthSolverManager.h
        DepthSolverManager.cpp
        MVStereo/LFDepthInfo.cpp
        Common/Common.h
        Common/QuadTree.h
        Common/QuadTree.cpp
        Common/Common.cpp
        Util/tinyxml2.cpp
        Util/tinyxml2.h
        Util/ImageQualityEstimate.h
        Util/ImageQualityEstimate.cpp
        MINeighbor/SelectNeighbors.cpp
        MINeighbor/SelectNeighbors.h
        FocalStack/LFFusion.cpp
        FocalStack/LFFusion.h
        FocalStack/LFRefocus.cpp
        FocalStack/LFRefocus.h
        DepthSolver.cpp
        DepthSolver.h
        Util/MISimilarityMeasure.h
        Util/MISimilarityMeasure.cpp
        MIStereo/MIStereoMatch.h
        MIStereo/MIStereoMatch.cpp
        Common/CommonUtil.cpp
        Common/CommonUtil.h
        MIStereo/AdaptMIPM.cpp
        MIStereo/AdaptMIPM.h
        Common/CommonCUDA.cpp
        MVStereo/LFACMP.cpp
        MVStereo/LFACMP.h
        MIStereo/AdaptMIPM.cu
        MIStereo/CudaUtil.cu
        MIStereo/CudaUtil.h
        MIStereo/AdaptMIPMUtil.cu
        MIStereo/AdaptMIPMUtil.cuh
        MIStereo/AdaaptMIPM_EdgewareUtil.cu
        MIStereo/AdaaptMIPM_EdgewareUtil.cuh
        MIStereo/AdaptMIPMPFPGR.cpp
        MIStereo/AdaptMIPMPFPGR.h
        MIStereo/AdaptMIPMPFPGR.cu
        MIStereo/AdaptMIPMPFPGRUtil.cu
        MIStereo/AdaptMIPMPFPGRUtil.cuh

        # ===== 新增：ACMM 风格多尺度整帧微图像匹配内核 =====
        MIStereo/AdaptMIPMFrameACMM.h
        MIStereo/AdaptMIPMFrameACMM.cpp
        MIStereo/AdaptMIPMFrameACMM.cu

        Util/Logger.cpp
        Util/Logger.h
        FocalStack/LFRefocus.h
        Util/Distribution_analyzer.cpp
        Util/Distribution_analyzer.h
        Util/LightFieldPlotter.cpp
        Util/LightFieldPlotter.h
        CheckModule/MICycleCheck.cpp
        CheckModule/MICycleCheck.h
        Common/PTotEstimator.cpp
        Common/PTotEstimator.h
        FocalStack/VirtualToRealDepthFunc.cpp
        FocalStack/VirtualToRealDepthFunc.h
        missing_symbols.c
        CommonMVS/AABB.h
        CommonMVS/AABB.inl
        CommonMVS/AutoEstimator.h
        CommonMVS/AutoPtr.h
        CommonMVS/Common.h
        CommonMVS/Common.cpp
        CommonMVS/Config.h
        CommonMVS/ConfigTable.h
        CommonMVS/ConfigTable.cpp
        CommonMVS/CriticalSection.h
        CommonMVS/CUDA.h
        CommonMVS/CUDA.cpp
        CommonMVS/EventQueue.h
        CommonMVS/EventQueue.cpp
        CommonMVS/FastDelegate.h
        CommonMVS/FastDelegateBind.h
        CommonMVS/FastDelegateCPP11.h
        CommonMVS/File.h
        CommonMVS/Filters.h
        CommonMVS/HalfFloat.h
        CommonMVS/Hash.h
        CommonMVS/HTMLDoc.h
        CommonMVS/LinkLib.h
        CommonMVS/List.h
        CommonMVS/Log.h
        CommonMVS/Log.cpp
        CommonMVS/MemFile.h
        CommonMVS/OBB.h
        CommonMVS/OBB.inl
        CommonMVS/Octree.h
        CommonMVS/Octree.inl
        CommonMVS/OctreeFusion.h
        CommonMVS/OctreeFusion.cpp
        CommonMVS/OctreeFusionVoxel.h
        CommonMVS/OctreeFusionVoxel.cpp
        CommonMVS/Plane.h
        CommonMVS/Plane.inl
        CommonMVS/Queue.h
        CommonMVS/Random.h
        CommonMVS/Ray.h
        CommonMVS/Ray.inl
        CommonMVS/Rotation.h
        CommonMVS/Rotation.inl
        CommonMVS/Sampler.inl
        CommonMVS/Semaphore.h
        CommonMVS/SharedPtr.h
        CommonMVS/SML.h
        CommonMVS/SML.cpp
        CommonMVS/Sphere.h
        CommonMVS/Sphere.inl
        CommonMVS/Streams.h
        CommonMVS/Strings.h
        CommonMVS/Thread.h
        CommonMVS/Timer.h
        CommonMVS/Timer.cpp
        CommonMVS/Types.h
        CommonMVS/Types.cpp
        CommonMVS/Types.inl
        CommonMVS/Util.h
        CommonMVS/Util.cpp
        CommonMVS/Util.inl
        CheckModule/MIDisparityFilter.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")

set_target_properties(ACMP PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

get_filename_component(BUILD_OUTPUT_DIR ${CMAKE_BINARY_DIR} ABSOLUTE)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(LIBV_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/libv/Debug/lib)
    set(OPENCV_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.04/Debug/lib)
    set(BOOST_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/debug/lib)
    set(LIBPLENO_LIB_PATH ${LF_WORK_PATH}/Common/LibPleno/lib/Debug)
else ()
    set(LIBV_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/libv/Release/lib)
    set(OPENCV_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.04/Release/lib)
    set(BOOST_LIB_PATH ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/release/lib)
    set(LIBPLENO_LIB_PATH ${LF_WORK_PATH}/Common/LibPleno/lib/Release)
endif ()

target_link_libraries(ACMP
        ${LIBV_LIB_PATH}/liblibv-graphic.so
        ${LIBV_LIB_PATH}/liblibv-geometry.so
        ${LIBV_LIB_PATH}/liblibv-core.so
        ${OpenCV_LIBS}
        ${LIBPLENO_NOGUI_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${Boost_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
)

file(GLOB LIBV_SO_FILES        "${LIBV_LIB_PATH}/liblibv-*.so*")
file(GLOB YAMLCPP_SO_FILES     "${LF_WORK_PATH}/ThirdPartyLib/yaml-cpp/lib/libyaml-cpp.so*")
file(GLOB OPENCV_SO_FILES      "${OPENCV_LIB_PATH}/libopencv_*.so*")
file(GLOB BOOST_SYSTEM_FILES   "${BOOST_LIB_PATH}/libboost_system.so*")
file(GLOB BOOST_FILESYSTEM_FILES "${BOOST_LIB_PATH}/libboost_filesystem.so*")
file(GLOB LIBPLENO_SO_FILES    "${LIBPLENO_LIB_PATH}/libpleno_nogui.so*")

get_filename_component(_CUDA_CUDART_DIR "${CUDA_CUDART_LIBRARY}" DIRECTORY)
file(GLOB CUDA_CUDART_SO_FILES "${_CUDA_CUDART_DIR}/libcudart.so*")

if (LIBV_SO_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBV_SO_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
if (YAMLCPP_SO_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YAMLCPP_SO_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
if (OPENCV_SO_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_SO_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
if (BOOST_SYSTEM_FILES OR BOOST_FILESYSTEM_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${BOOST_SYSTEM_FILES} ${BOOST_FILESYSTEM_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
if (LIBPLENO_SO_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBPLENO_SO_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
if (CUDA_CUDART_SO_FILES)
    add_custom_command(TARGET ACMP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDA_CUDART_SO_FILES} ${CMAKE_BINARY_DIR}/
    )
endif ()
