cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0015 NEW)

set(ADDITIONAL_INCLUDE_DIR)
set(ADDITIONAL_LIBRARY)
set(PROJECT_EXECUTABLE FALSE)

#===============================================================================
# Apple Specific
#===============================================================================
if (APPLE)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "-x objective-c++")
endif (APPLE)

# Apple specific script for code signing and branching projects
if (APPLE)
  set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "Z3XJW97K73")
  set(CREATE_IOS_PROJECT "" CACHE BOOL "Create project for iOS")

  if (CREATE_IOS_PROJECT)
    set(PROJECT_PLATFORM_SUFFIX "_ios")
  else (CREATE_IOS_PROJECT)
    set(PROJECT_PLATFORM_SUFFIX "_macos")
  endif (CREATE_IOS_PROJECT)
endif (APPLE)
#===============================================================================
# End Apple Specific
#===============================================================================

project(ParticlePhysics${PROJECT_PLATFORM_SUFFIX})

set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")
link_directories("${CMAKE_BINARY_DIR}")

#===============================================================================
# Macros
#===============================================================================
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
    MESSAGE(ERROR ": Framework ${fwname} not found in " ${framework_path})
  else (${FRAMEWORK_${fwname}} STREQUAL FRAMEWORK_${fwname}-NOTFOUND)
    target_link_libraries(${appname} ${FRAMEWORK_${fwname}})
      MESSAGE(STATUS "Framework ${fwname} found at ${FRAMEWORK_${fwname}}")
  endif (${FRAMEWORK_${fwname}} STREQUAL FRAMEWORK_${fwname}-NOTFOUND)
endmacro(ADD_FRAMEWORK)

# code to handle additional include directories
macro(INCLUDE_ADDITIONAL_DIR ADDITIONAL_INCLUDE_DIR)
  message("Depends on: ${ADDITIONAL_INCLUDE_DIR}")
  #get the root directory
  get_directory_property(PARENT PARENT_DIRECTORY)
  foreach(ADDITIONAL_INCLUDE ${ADDITIONAL_INCLUDE_DIR})
    #go inside additional directory
    set(ADD_INCLUDE_DIRECTORY "${PARENT}/${ADDITIONAL_INCLUDE}")
    #get the list of sub directories of the additional include
    subdirlist(ADDITIONAL_INCLUDE_SUBDIRS ${ADD_INCLUDE_DIRECTORY})
    include_directories("${PARENT}/${ADDITIONAL_INCLUDE}")
  endforeach(ADDITIONAL_INCLUDE ${ADDITIONAL_INCLUDE_DIR})
endmacro(INCLUDE_ADDITIONAL_DIR ADDITIONAL_INCLUDE_DIR)

# script for sub directories
macro(CREATE_LIB_AND_EXE)
  file(GLOB_RECURSE SOURCES_C "*.c" "*.cpp" "*.h" "../*.xml")
  file(GLOB_RECURSE SOURCES_S "*.shader")

  set_source_files_properties(${SOURCES_S} PROPERTIES LANGUAGE METAL)

  #code to get current directory name
  get_directory_property(CURRENT_DIRECTORY PARENT_DIRECTORY)
  set(CURRENT_DIRECTORY "${CURRENT_DIRECTORY}/${PROJECT_STRING}")

  #get subdirectories
  subdirlist(SUBDIRS ${CURRENT_DIRECTORY})
  groupfiles(${CURRENT_DIRECTORY})

  #if some directory is to be included
  include_additional_dir("${ADDITIONAL_INCLUDE_DIR}")

  message("List of subdirectories: ${SUBDIRS}")

  if (APPLE)
    # group resources and frameworks
    source_group(Resources FILES ${SOURCES_SHADER})
    source_group(Frameworks FILES ${SOURCES_FRAMEWORK})

    if (PROJECT_EXECUTABLE)
      add_executable(${PROJECT_STRING} MACOSX_BUNDLE ${SOURCES_S} ${SOURCES_C}
        ${SOURCES_SHADER} ${SOURCES_FRAMEWORK} ${APPLE_RESOURCES}
        "${CMAKE_SOURCE_DIR}/Shared/plist.in")
      # using custom info.plist embedded with story board
      set_target_properties(${PROJECT_STRING} PROPERTIES
        XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/../Frameworks"
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/Shared/plist.in"
        RESOURCE "${APPLE_RESOURCES}")
      if (CREATE_IOS_PROJECT)
        add_custom_command(TARGET ${PROJECT_STRING} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SOURCES_SHADER} $<TARGET_FILE_DIR:${PROJECT_STRING}>)
        add_custom_command(TARGET ${PROJECT_STRING} POST_BUILD COMMAND /usr/bin/codesign ARGS --deep --force --sign ${SIGNING_IDENTITY} --timestamp=none $<TARGET_FILE_DIR:${PROJECT_STRING}>)
      else (CREATE_IOS_PROJECT)
        add_custom_command(TARGET ${PROJECT_STRING} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SOURCES_SHADER} $<TARGET_FILE_DIR:${PROJECT_STRING}>/../Resources/)
        add_custom_command(TARGET ${PROJECT_STRING} POST_BUILD COMMAND /usr/bin/codesign ARGS --deep --force --sign ${SIGNING_IDENTITY} --timestamp=none ${CMAKE_CURRENT_BINARY_DIR}/\${CONFIGURATION}/${PROJECT_STRING}.app)
      endif (CREATE_IOS_PROJECT)
      set_source_files_properties(${SOURCES_FRAMEWORK} PROPERTIES MACOSX_PACKAGE_LOCATION Frameworks)
      set_source_files_properties(${APPLE_RESOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    else (PROJECT_EXECUTABLE)
      add_library(${PROJECT_STRING} ${SOURCES_S} ${SOURCES_C})
      set_target_properties(${PROJECT_STRING} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif (PROJECT_EXECUTABLE)

  else (APPLE)
    if (PROJECT_EXECUTABLE)
      add_executable(${PROJECT_STRING} ${SOURCES_S} ${SOURCES_C})
    else (PROJECT_EXECUTABLE)
      add_library(${PROJECT_STRING} ${SOURCES_S} ${SOURCES_C})
      set_target_properties(${PROJECT_STRING} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif (PROJECT_EXECUTABLE)
    add_custom_command(TARGET ${PROJECT_STRING} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SOURCES_SHADER} $<TARGET_FILE_DIR:${PROJECT_STRING}>)

  endif (APPLE)

  target_link_libraries(${PROJECT_STRING} ${ADDITIONAL_LIBRARY})
  set(ADDITIONAL_FILES ${ADDITIONAL_FILES} ${SOURCES_S} ${SOURCES_C} PARENT_SCOPE)

  set(SOURCES_SHADER)
endmacro(CREATE_LIB_AND_EXE)
#===============================================================================
# End Macros
#===============================================================================

# include Freeglut and GLEW on Windows
if (WIN32)
  include_directories("External/GLEW/include/GL/")
  link_directories("${CMAKE_SOURCE_DIR}/External/GLEW/lib/Release/Win32/")

  set(PROJECT_STRING Freeglut)
  add_subdirectory(Freeglut)

  set(ADDITIONAL_INCLUDE_DIR Freeglut)
  set(ADDITIONAL_LIBRARY Freeglut glew32)
endif (WIN32)

# include SDL on Apple
if (APPLE)
  if (CREATE_IOS_PROJECT)
    set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} "External/SDL/iOS/")
    set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} "External/SDL/iOS/SDL2/")
    set(APPLE_RESOURCES "${CMAKE_SOURCE_DIR}/Shared/LaunchScreen.storyboard"
      "${CMAKE_SOURCE_DIR}/Shared/Main.storyboard")
  else (CREATE_IOS_PROJECT)
    set(SOURCES_FRAMEWORK "${CMAKE_SOURCE_DIR}/External/SDL/MacOS/SDL2.framework")
    set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} "External/SDL/MacOS/SDL2.framework/Headers/")
  endif (CREATE_IOS_PROJECT)
  set(APPLE_RESOURCES "${APPLE_RESOURCES}" "${CMAKE_SOURCE_DIR}/Shared/Default-568h@2x.png")
endif (APPLE)

find_package(Vulkan)
if (Vulkan_FOUND)
  include_directories("${Vulkan_INCLUDE_DIR}")
  link_directories("${Vulkan_LIBRARY}")
  set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} vulkan-1)
endif (Vulkan_FOUND)

find_package(OpenCL)
if (OPENCL_FOUND)
  include_directories("${OpenCL_INCLUDE_DIR}")
  link_directories("${OPENCL_INCLUDE_DIR}")
  set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} ${OpenCL_LIBRARY})
endif (OPENCL_FOUND)

set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} Core)
set(PROJECT_STRING Core)
add_subdirectory(Core)

# Apple frameworks
if (APPLE)
  set(APPLE_FRAMEWORK_PATH ${CMAKE_OSX_SYSROOT})

  if (CREATE_IOS_PROJECT)
    # Setup iOS developer location
    execute_process(COMMAND xcode-select -print-path OUTPUT_VARIABLE XCODE_COMPILER_PRINT_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_IOS_DEVELOPER_ROOT ${CMAKE_IOS_DEVELOPER_ROOT} CACHE PATH "Location of iOS Platform")
    set(CMAKE_IOS_DEVELOPER_ROOT "${XCODE_COMPILER_PRINT_PATH}/Platforms/iPhoneOS.platform/Developer")

    # Find and use the most recent iOS sdk
    file(GLOB _CMAKE_IOS_SDKS "${CMAKE_IOS_DEVELOPER_ROOT}/SDKs/*")
    if (_CMAKE_IOS_SDKS)
      list(SORT _CMAKE_IOS_SDKS)
      list(REVERSE _CMAKE_IOS_SDKS)
      list(GET _CMAKE_IOS_SDKS 0 CMAKE_IOS_SYSROOT)
    else (_CMAKE_IOS_SDKS)
      message(FATAL_ERROR "No iOS SDK's found in default search path ${CMAKE_IOS_DEVELOPER_ROOT}. Manually set CMAKE_IOS_SYSROOT or install the iOS SDK.")
    endif (_CMAKE_IOS_SDKS)

    message (STATUS "Toolchain using default iOS SDK: ${CMAKE_IOS_SYSROOT}")

    set(CMAKE_IOS_SYSROOT ${CMAKE_IOS_SYSROOT} CACHE PATH "Location of the selected iOS SDK")
    set(APPLE_FRAMEWORK_PATH ${CMAKE_IOS_SYSROOT})

  endif (CREATE_IOS_PROJECT)

  # common frameworks
  add_framework(Foundation Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
  add_framework(Metal Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")

  # mac frameworks
  if (NOT CREATE_IOS_PROJECT)
    add_framework(GLKit Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(OpenCL Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(OpenGL Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(SDL2 Core "${CMAKE_SOURCE_DIR}/External/SDL/MacOS/")
    set_target_properties(Core PROPERTIES LINK_FLAGS "-F${CMAKE_SOURCE_DIR}/External/SDL/MacOS/")

  else (NOT CREATE_IOS_PROJECT)
    add_framework(AudioToolbox Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(AVFoundation Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(CoreAudio Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(CoreGraphics Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(CoreMotion Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(CoreBluetooth Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(GameController Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(OpenGLES Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(QuartzCore Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    add_framework(UiKit Core "${APPLE_FRAMEWORK_PATH}/System/Library/Frameworks")
    target_link_libraries(Core "${CMAKE_SOURCE_DIR}/External/SDL/iOS/SDL2/libSDL2.a")

  endif (NOT CREATE_IOS_PROJECT)

endif (APPLE)

set(ADDITIONAL_INCLUDE_DIR ${ADDITIONAL_INCLUDE_DIR} ParticlePhysics)
set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} Core)
set(PROJECT_STRING ParticlePhysics)
add_subdirectory(ParticlePhysics)

set(PROJECT_EXECUTABLE TRUE)
set(ADDITIONAL_LIBRARY ${ADDITIONAL_LIBRARY} ParticlePhysics)

set(PROJECT_STRING App)
add_subdirectory(App)
set_target_properties(App PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER "org.vivekprojects.particlephysics.app")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Mac Developer")

set(PROJECT_STRING Tests)
add_subdirectory(Tests)
set_target_properties(Tests PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER "org.vivekprojects.particlephysics.tests")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer")
