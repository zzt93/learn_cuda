cmake_minimum_required(VERSION 3.27)
project(learn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(learn_cuda main.cu
        sum.cu
        transpose.cu
        softmax.cu)

set_target_properties(learn_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
