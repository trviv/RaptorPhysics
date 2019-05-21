cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0015 NEW)

project(ParticlePhysics)

if (APPLE)
  set(CMAKE_CXX_STANDARD 14)
endif (APPLE)

find_package(OpenCL REQUIRED)
find_package(Vulkan)

# code to get subdirectories
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
  # code to include subdirectories to the project
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

# code to add a framework
macro(ADD_FRAMEWORK fwname appname framework_path)
  find_library(FRAMEWORK_${fwname}
    NAMES ${fwname}
    PATHS ${framework_path}
    NO_DEFAULT_PATH)
  if (${FRAMEWORK_${fwname}} STREQUAL FRAMEWORK_${fwname}-NOTFOUND)
    MESSAGE(ERROR ": Framework ${fwname} not found")
  else (${FRAMEWORK_${fwname}} STREQUAL FRAMEWORK_${fwname}-NOTFOUND)
    target_link_libraries(${appname} ${FRAMEWORK_${fwname}})
      MESSAGE(STATUS "Framework ${fwname} found at ${FRAMEWORK_${fwname}}")
  endif (${FRAMEWORK_${fwname}} STREQUAL FRAMEWORK_${fwname}-NOTFOUND)
endmacro(ADD_FRAMEWORK)

set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")

message(${CMAKE_BINARY_DIR})

if(OPENCL_FOUND)

  message("OpenCL found generating GPU code")

  # include Freeglut and GLEW on Windows
  if (WIN32)
    include_directories("External/GLEW/include/GL/")
    link_directories("${CMAKE_SOURCE_DIR}/External/GLEW/lib/Release/Win32/")

    set(PROJECT_STRING Freeglut)
    add_subdirectory(Freeglut)

    set(ADDITIONAL_INCLUDE_DIR Freeglut)
    set(ADDITIONAL_LIBRARY Freeglut glew32)
  endif (WIN32)

  if (Vulkan_FOUND)
    include_directories("${Vulkan_INCLUDE_DIR}")
    link_directories("${Vulkan_LIBRARY}")
  endif (Vulkan_FOUND)

  include_directories("${OpenCL_INCLUDE_DIR}")
  link_directories("${OPENCL_INCLUDE_DIR}")
  link_directories("${CMAKE_BINARY_DIR}")

  set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} Core)

  set(PROJECT_STRING Core)
  add_subdirectory(Core)

  if (APPLE)
    add_framework(Foundation Core "/System/Library/Frameworks")
    add_framework(OpenCL Core "/System/Library/Frameworks")
    add_framework(OpenGL Core "/System/Library/Frameworks")
    add_framework(SDL2 Core "${CMAKE_SOURCE_DIR}/External/SDL/MacOS")
    set_target_properties(Core PROPERTIES LINK_FLAGS "-F${CMAKE_SOURCE_DIR}/External/SDL/MacOS/")
  endif (APPLE)

  set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} ParticlePhysics)
  set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} Core)
  set(PROJECT_STRING ParticlePhysics)
  add_subdirectory(ParticlePhysics)

  if(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} ParticlePhysics vulkan-1)
  else(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} ParticlePhysics)
  endif(Vulkan_FOUND)

  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING App)
  add_subdirectory(App)

  if(Vulkan_FOUND)
    set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} vulkan-1)
  endif(Vulkan_FOUND)
  set(PROJECT_EXECUTABLE TRUE)
  set(PROJECT_STRING Tests)
  add_subdirectory(Tests)

else(OPENCL_FOUND)

  message("OpenCL not found")

endif(OPENCL_FOUND)
