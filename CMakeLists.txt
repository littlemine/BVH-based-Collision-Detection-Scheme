cmake_minimum_required(VERSION 3.19)

project(efficient_bvhcd CXX CUDA)
find_package(assimp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(bvhcd)
target_link_libraries(bvhcd PRIVATE assimp::assimp)
target_link_libraries(bvhcd PRIVATE CUDA::cuda_driver CUDA::cudart)

target_compile_features(bvhcd PRIVATE cuda_std_17 cxx_std_17)
set_target_properties(bvhcd
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(bvhcd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Externals/inc)
target_include_directories(bvhcd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Externals/inc/gProximity)
target_include_directories(bvhcd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Externals/inc/cuda)
target_include_directories(bvhcd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Project/Mine)

target_compile_options(bvhcd
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-allow-expensive-optimizations=true>

  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=loop_not_reachable --diag_suppress=2642">
)
set_target_properties(bvhcd
  PROPERTIES # LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(bvhcd INTERFACE CMAKE_GENERATOR_PLATFORM=x64)
target_link_options(bvhcd
  PRIVATE $<DEVICE_LINK:-dlto>
)

#
target_sources(bvhcd PRIVATE
  Source/Project/Mine/entry_cudaproject.cpp
  Source/Project/Mine/world/Scene.cpp
  Source/Project/Mine/utility/CudaDeviceUtils.cu
  Source/Project/Mine/utility/CudaKernelUtils.cu
  Source/Project/Mine/system/CudaDevice/CudaDevice.cu
  Source/Project/Mine/system/CudaDevice/CudaKernelLauncher.cu
  Source/Project/Mine/setting/CDBenchmarkSettings.cpp
  Source/Project/Mine/Frameworks/AppBase/AppBase.cpp
  Source/Project/Mine/Frameworks/AppBase/Main.cpp
  Source/Project/Mine/Frameworks/CudaProject/CudaProjectApp.cpp
  Source/Project/Mine/Frameworks/CudaProject/CudaProjectModuleRegister.cpp

  Source/Project/Mine/collision/auxiliary/BvhRestrLog.cpp
  Source/Project/Mine/collision/auxiliary/BvhRestrLog.cu
  Source/Project/Mine/collision/auxiliary/FlOrderLog.cpp
  Source/Project/Mine/collision/auxiliary/FlOrderLog.cu

  Source/Project/Mine/collision/bvtt_front/BvhCDKernels.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontGenerateKernels.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontKeepKernels.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontLooseInter.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontLooseIntra.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontPruneSproutKernels.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontReorderKernels.cu
  Source/Project/Mine/collision/bvtt_front/BvttFrontRestructureKernels.cu

  Source/Project/Mine/collision/lbvh/BvhBV.cu
  Source/Project/Mine/collision/lbvh/BvhExtNode.cu
  Source/Project/Mine/collision/lbvh/BvhIntNode.cu
  Source/Project/Mine/collision/lbvh/BvhPrimitiveNode.cu
  Source/Project/Mine/collision/lbvh/LBvhBuildKernels.cu
  Source/Project/Mine/collision/lbvh/LBvhFixedDeformable.cu
  Source/Project/Mine/collision/lbvh/LBvhRestructureKernels.cu
  Source/Project/Mine/collision/lbvh/LBvhRigid.cu
  Source/Project/Mine/collision/lbvh/LBvhUpdateKernels.cu

  Source/Project/Mine/collision/narrow_phase/narrow_phase.cu
)
