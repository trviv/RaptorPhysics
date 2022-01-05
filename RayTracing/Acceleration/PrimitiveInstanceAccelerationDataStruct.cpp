#include "PrimitiveInstanceAccelerationDataStruct.h"

PrimitiveInstanceAccelerationDataStruct::PrimitiveInstanceAccelerationDataStruct()
{
}

PrimitiveInstanceAccelerationDataStruct::~PrimitiveInstanceAccelerationDataStruct()
{
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS)
{
  if (primitiveInstances.count(primitiveEntityADS) == 0)
  {
    primitiveInstances[primitiveEntityADS] = vector<const RayTracingEntity*>();
  }
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveInstance(const RayTracingEntity* entityInstance)
{
  const auto instanceIdentity = getRayTracingEntityId(entityInstance->getIdentity());
  for (auto& prim : primitiveInstances)
  {
    if (getRayTracingEntityId(prim.first->primitiveEntity->getIdentity()) == instanceIdentity)
    {
      prim.second.push_back(entityInstance);
    }
  }
}
