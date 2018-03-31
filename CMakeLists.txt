cmake_minimum_required(VERSION 2.8)

project(ParticlePhysics)

find_package(OpenCL REQUIRED)

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

if(OPENCL_FOUND)

  message("OpenCL found generating GPU code")

  set(PROJECT_STRING Freeglut)
  add_subdirectory(Freeglut)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003")

  include_directories("External/GLEW/include/GL/")
  include_directories("${OpenCL_INCLUDE_DIR}")

  link_directories("${CMAKE_BINARY_DIR}")
  link_directories("${CMAKE_SOURCE_DIR}/External/GLEW/lib/Release/Win32/")
  link_directories("${OPENCL_INCLUDE_DIR}")

  set(ADDITIONAL_INCLUDE_DIR Freeglut Core)
  set(ADDITIONAL_LIBRARY Freeglut)
  set(PROJECT_STRING Core)
  add_subdirectory(Core)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  set(ADDITIONAL_LIBRARY Core Freeglut glew32)
  set(PROJECT_STRING ParticlePhysics)
  add_subdirectory(ParticlePhysics)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  set(ADDITIONAL_LIBRARY ParticlePhysics Core Freeglut glew32)
  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING App)
  add_subdirectory(App)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  set(ADDITIONAL_LIBRARY Core Freeglut glew32)
  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING Tests)
  add_subdirectory(Tests)

else(OPENCL_FOUND)

  message("OpenCL not found")

endif(OPENCL_FOUND)