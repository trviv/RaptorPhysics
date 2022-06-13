set (SOURCES_SHADER
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Compute/ComputeShared.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Compute/ComputeUtilsShared.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../ParticlePhysics/Common/ConstrainStruct.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../ParticlePhysics/Common/ParticleStruct.h"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/ComputeHeader.shader"
  "${PROJECT_SOURCE_DIR}/${PROJECT_STRING}/../Core/Shaders/ComputeUtils.shader")

create_lib_and_exe()
