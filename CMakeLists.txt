cmake_minimum_required(VERSION 3.3)



set(CMAKE_CXX_STANDARD 17)
# find cuda compiler
execute_process(
    COMMAND where.exe nvcc
    OUTPUT_VARIABLE NVCC_PATH
    RESULT_VARIABLE NVCC_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NVCC_RESULT EQUAL 0)
    message(FATAL_ERROR "nvcc not found. Please install CUDA Toolkit or add nvcc to PATH.")
endif()
string(REPLACE "\\" "/" NVCC_PATH "${NVCC_PATH}")
string(REGEX REPLACE "\r?\n.*" "" NVCC_PATH "${NVCC_PATH}")
set(CMAKE_CUDA_COMPILER "${NVCC_PATH}" CACHE FILEPATH "CUDA compiler")
message(STATUS "Found nvcc: ${CMAKE_CUDA_COMPILER}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        message(STATUS "Running __nvcc_device_query for CMAKE_CUDA_ARCHITECHTURES")
        execute_process(COMMAND "__nvcc_device_query"
                        RESULT_VARIABLE NVCC_DEVICE_QUERY_RESULT
                        OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECHTURES
                        OUTPUT_STRIP_TRAILING_WHILESPACE)
        if(NVCC_DEVICE_QUERY_RESULT EQUAL 0)
            message(STATUS "CMAKE_CUDA_ARCHITECHTURES set to ${CMAKE_CUDA_ARCHITECHTURES}")
        else()
            message(WARNING "CMAKE_CUDA_ARCHITECHTURES not defined and __nvcc_device_query failed")
            unset(CMAKE_CUDA_ARCHITECTURES)
        endif()
    endif()       
endif()

project(morden_cuda LANGUAGES CXX CUDA)

file(GLOB SOURCES "*.cpp" "*.cu")
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
target_link_libraries(${PROJECT_NAME} PRIVATE cusparse cublas)
