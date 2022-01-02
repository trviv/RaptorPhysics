#ifndef PRIMITIVE_ACCELERATION_DATA_STRUCTURE_H
#define PRIMITIVE_ACCELERATION_DATA_STRUCTURE_H

#include <Entities/RayTracingEntity.h>
#include "BoundingVolumeHierarchyADS.h"

/*!
@class Class to create acceleration data structure for a primitive.
*/
class PrimitiveAccelerationDataStruct : public BoundingVolumeHierarchyADS
{
  const RayTracingEntity* primitiveEntity;

public:

  PrimitiveAccelerationDataStruct();

  ~PrimitiveAccelerationDataStruct();

  void bindEntity(const RayTracingEntity* primitiveEntity);
};

#endif
