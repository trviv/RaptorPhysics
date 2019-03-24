cmake_minimum_required(VERSION 2.8)

project(ParticlePhysics)

find_package(OpenCL REQUIRED)
find_package(Vulkan)

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

macro(GROUPFILES CURRENT_DIRECTORY)
  subdirlist(SUBDIRS ${CURRENT_DIRECTORY})
  #code to include subdirectories to the project
  foreach(SUBDIR ${SUBDIRS})
    file(GLOB_RECURSE dir_files ${SUBDIR}/*.h ${SUBDIR}/*.cu ${SUBDIR}/*.cpp ${SUBDIR}/*.shader)
    foreach(filename ${dir_files})
      string(LENGTH ${CURRENT_DIRECTORY} len)
      string(LENGTH ${SUBDIR} len2)
      MATH(EXPR len "${len}+${len2}+2")
      string(SUBSTRING ${filename} ${len} 5 files2)
      string(FIND ${files2} "." len)
      if (len)
        string(SUBSTRING ${files2} 0 ${len} files2)
      endif (len)
      string(FIND ${files2} "/" len)
      if (len)
        string(SUBSTRING ${files2} 0 ${len} files2)
      endif (len)
      source_group(${SUBDIR}\\${files2} FILES ${filename})
    endforeach(filename ${dir_files})
  endforeach(SUBDIR ${SUBDIRS})
endmacro(GROUPFILES CURRENT_DIRECTORY)

set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

message(${CMAKE_BINARY_DIR})

if(OPENCL_FOUND)

  message("OpenCL found generating GPU code")

  set(PROJECT_STRING Freeglut)
  add_subdirectory(Freeglut)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003")

  include_directories("External/GLEW/include/GL/")
  include_directories("${OpenCL_INCLUDE_DIR}")
  if(Vulkan_FOUND)
    include_directories("${Vulkan_INCLUDE_DIR}")
  endif(Vulkan_FOUND)

  link_directories("${CMAKE_BINARY_DIR}")
  link_directories("${CMAKE_SOURCE_DIR}/External/GLEW/lib/Release/Win32/")
  link_directories("${OPENCL_INCLUDE_DIR}")
  if(Vulkan_FOUND)
    link_directories("${Vulkan_LIBRARY}")
  #set(OpenCL_LIBRARY ${OpenCL_LIBRARY} ${Vulkan_LIBRARY_FILE})
  #link_directories("${CMAKE_SOURCE_DIR}/External/Vulkan/Lib")
  endif(Vulkan_FOUND)

  set(ADDITIONAL_INCLUDE_DIR Freeglut Core)
  set(ADDITIONAL_LIBRARY Freeglut)
  set(PROJECT_STRING Core)
  add_subdirectory(Core)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  set(ADDITIONAL_LIBRARY Core Freeglut glew32)
  set(PROJECT_STRING ParticlePhysics)
  add_subdirectory(ParticlePhysics)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  if(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY ParticlePhysics Core Freeglut glew32 vulkan-1)
  else(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY ParticlePhysics Core Freeglut glew32)
  endif(Vulkan_FOUND)

  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING App)
  add_subdirectory(App)

  set(ADDITIONAL_INCLUDE_DIR Core Freeglut ParticlePhysics)
  if(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY Core Freeglut glew32 vulkan-1)
  else(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY Core Freeglut glew32)
  endif(Vulkan_FOUND)
  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING Tests)
  add_subdirectory(Tests)

else(OPENCL_FOUND)

  message("OpenCL not found")

endif(OPENCL_FOUND)