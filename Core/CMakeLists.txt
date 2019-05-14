file(GLOB_RECURSE SOURCES_C "*.c" "*.cpp" "*.h" "*.shader" "../*.xml")

#code to get current directory name
get_directory_property(CURRENT_DIRECTORY PARENT_DIRECTORY)
set(CURRENT_DIRECTORY "${CURRENT_DIRECTORY}/${PROJECT_STRING}")

#get subdirectories
subdirlist(SUBDIRS ${CURRENT_DIRECTORY})
groupfiles(${CURRENT_DIRECTORY})

#if some directory is to be included
if(ADDITIONAL_INCLUDE_DIR)

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

endif(ADDITIONAL_INCLUDE_DIR)

message("List of subdirectories: ${SUBDIRS}")

if(CUSTOM_DEFINITIONS)
  add_definitions(${CUSTOM_DEFINITIONS})
endif(CUSTOM_DEFINITIONS)

if(PROJECT_EXECUTABLE)

  add_executable(${PROJECT_STRING} ${SOURCES_CU} ${SOURCES_C})

else(PROJECT_EXECUTABLE)

  add_library(${PROJECT_STRING} ${SOURCES_CU} ${SOURCES_C})
  set_target_properties(${PROJECT_STRING} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif(PROJECT_EXECUTABLE)
target_link_libraries(${PROJECT_STRING} ${ADDITIONAL_LIBRARY} ${OpenCL_LIBRARY})
set(ADDITIONAL_FILES ${ADDITIONAL_FILES} ${SOURCES_CU} ${SOURCES_C} PARENT_SCOPE)
