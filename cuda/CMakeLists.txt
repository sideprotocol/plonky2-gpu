cmake_minimum_required(VERSION 3.16)
project(PLONKY2_CUDA CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(SRC_LIST
        plonky2_gpu.cu
        test.cu
#        plonky2_gpu.cu
)
#add_definitions(-D__CUDA_ARCH__)

#add_compile_options(-G)

#add_library(plonky2_gpu SHARED ${SRC_LIST})
add_executable(cutest ${SRC_LIST})

