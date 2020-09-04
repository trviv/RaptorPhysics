set (SOURCES_SHADER
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Compute/ComputeShared.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Compute/ComputeUtilsShared.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../ParticlePhysics/Common/ConstrainStruct.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../ParticlePhysics/Common/ParticleStruct.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/ComputeHeader.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/ComputeUtils.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/PrefixScan.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/Reduce.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/ComplexReduce.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/RadixSort.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/TestCSV.csv")

create_lib_and_exe()
