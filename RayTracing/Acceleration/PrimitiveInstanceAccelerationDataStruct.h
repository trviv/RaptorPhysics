#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCTURE_H
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCTURE_H

#include "PrimitiveAccelerationDataStruct.h"

/*!
@class Class to create acceleration data structure for a primitive instances.
*/
class PrimitiveInstanceAccelerationDataStruct : public BoundingVolumeHierarchyADS
{
  vector<const PrimitiveAccelerationDataStruct*> primitiveInstances;

  vector<const RayTracingEntity*> entityInstances;

public:

  PrimitiveInstanceAccelerationDataStruct();

  ~PrimitiveInstanceAccelerationDataStruct();

  void registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS);

  void registerPrimitiveInstance(const RayTracingEntity* entityInstance);
};

#endif
