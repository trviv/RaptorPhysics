cmake_minimum_required(VERSION 2.8)

project(ParticlePhysics)

find_package(CUDA REQUIRED)

#code to get subdirectories
macro(SUBDIRLIST RESULT CURRENT_DIRECTORY)
  file(GLOB children RELATIVE ${CURRENT_DIRECTORY} ${CURRENT_DIRECTORY}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${CURRENT_DIRECTORY}/${child})
      set(dirlist ${dirlist} ${child})
    endif(IS_DIRECTORY ${CURRENT_DIRECTORY}/${child})
  endforeach(child ${children})
  set(${RESULT} ${dirlist})
endmacro(SUBDIRLIST RESULT CURRENT_DIRECTORY)

set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

message(${CMAKE_BINARY_DIR})

if(CUDA_FOUND)

  message("CUDA found generating GPU code")
  #set(NVCC_VERBOSE 1)

  set(PROJECT_STRING Freeglut)
  add_subdirectory(Freeglut)

  get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
  link_directories(${CUDA_LIB_DIR})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003")

  #set(CUDA_SEPARABLE_COMPILATION ON)

  #set(CUDA_BUILD_CUBIN ON)
  #set(CUDA_CUSTOM_NVCC_FLAGS -keep -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -rdc=true)
  set(CUDA_CUSTOM_NVCC_FLAGS -keep -gencode arch=compute_30,code=sm_30)

  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(External/GLEW/include/GL/)
  link_directories(${CMAKE_BINARY_DIR})
  link_directories("${CMAKE_BINARY_DIR}/../../External/GLEW/lib/Release/Win32/")

  set(ADDITIONAL_INCLUDE_DIR Freeglut)
  set(PROJECT_STRING Core)
  add_subdirectory(Core)

  #set(CUDA_CUSTOM_NVCC_FLAGS -keep -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -rdc=true)
  #set(CUDA_CUSTOM_NVCC_FLAGS -keep -gencode arch=compute_30,code=sm_30 -rdc=true)
  set(CUDA_ADDITIONAL_LIBRARIES -L${CMAKE_BINARY_DIR}/Release;-lCudas)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut)
  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING ParticlePhysics)
  add_subdirectory(ParticlePhysics)

  add_dependencies(Core Freeglut)
  add_dependencies(ParticlePhysics Core)

else(CUDA_FOUND)

  message("CUDA not found")

endif(CUDA_FOUND)