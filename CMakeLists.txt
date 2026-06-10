cmake_minimum_required(VERSION 3.24)

project(RAIF
    VERSION 1.0.0
    DESCRIPTION "Reliable AI Inference Fabric - local C++/CUDA inference kernels"
    LANGUAGES CXX
)

option(RAIF_ENABLE_CUDA "Build CUDA kernels" ON)
option(RAIF_ENABLE_CUBLAS "Build optional cuBLAS benchmark baseline when available" ON)
set(RAIF_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architecture list. RTX 2060 uses 75.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE RAIF_PYBIND11_STATUS
    OUTPUT_VARIABLE RAIF_PYBIND11_CMAKE_DIR
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT RAIF_PYBIND11_STATUS EQUAL 0)
    message(FATAL_ERROR "pybind11 was not found for ${Python3_EXECUTABLE}. Run: python -m pip install pybind11")
endif()

set(pybind11_DIR "${RAIF_PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake package directory")
find_package(pybind11 CONFIG REQUIRED)

add_library(raif_cpu STATIC
    src/cpu/gelu_cpu.cpp
    src/cpu/layernorm_cpu.cpp
    src/cpu/softmax_cpu.cpp
    src/cpu/gemm_cpu.cpp
    src/cpu/benchmark_cpu.cpp
)

target_include_directories(raif_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(raif_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(raif_cpu PUBLIC RAIF_VERSION="${PROJECT_VERSION}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(raif_cpu PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(raif_cpu PUBLIC RAIF_WITH_OPENMP=1)
endif()

if(MSVC)
    target_compile_options(raif_cpu PRIVATE /W4 /permissive-)
else()
    target_compile_options(raif_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(RAIF_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(raif_cuda STATIC
        src/cuda/gelu_cuda.cu
        src/cuda/layernorm_cuda.cu
        src/cuda/softmax_cuda.cu
        src/cuda/gemm_cuda.cu
    )

    target_include_directories(raif_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(raif_cuda PUBLIC CUDA::cudart)
    target_compile_definitions(raif_cuda PUBLIC RAIF_WITH_CUDA=1)

    set(RAIF_HAS_CUBLAS OFF)
    if(RAIF_ENABLE_CUBLAS AND TARGET CUDA::cublas)
        target_link_libraries(raif_cuda PUBLIC CUDA::cublas)
        target_compile_definitions(raif_cuda PUBLIC RAIF_WITH_CUBLAS=1)
        set(RAIF_HAS_CUBLAS ON)
    endif()

    set_target_properties(raif_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${RAIF_CUDA_ARCHITECTURES}"
    )

    if(MSVC)
        target_compile_options(raif_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
    else()
        target_compile_options(raif_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    endif()
endif()

pybind11_add_module(raif MODULE
    bindings/python/raif_bindings.cpp
)

target_include_directories(raif PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(raif PRIVATE raif_cpu)

if(RAIF_ENABLE_CUDA)
    target_link_libraries(raif PRIVATE raif_cuda)
    target_compile_definitions(raif PRIVATE RAIF_WITH_CUDA=1)
    if(RAIF_HAS_CUBLAS)
        target_compile_definitions(raif PRIVATE RAIF_WITH_CUBLAS=1)
    endif()
endif()

set_target_properties(raif PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

foreach(CONFIG_TYPE DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(raif PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE} "${CMAKE_BINARY_DIR}/python"
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE} "${CMAKE_BINARY_DIR}/python"
    )
endforeach()
