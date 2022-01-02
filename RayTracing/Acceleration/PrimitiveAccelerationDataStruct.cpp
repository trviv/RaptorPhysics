#include "PrimitiveAccelerationDataStruct.h"

PrimitiveAccelerationDataStruct::PrimitiveAccelerationDataStruct()
{
}

PrimitiveAccelerationDataStruct::~PrimitiveAccelerationDataStruct()
{
}

void PrimitiveAccelerationDataStruct::bindEntity(const RayTracingEntity* primitiveEntity)
{
  this->primitiveEntity = primitiveEntity;
  this->vertexArray     = (*primitiveEntity)[EntityPrimitiveAttributePosition];
  this->attributeArray  = (*primitiveEntity)[EntityPrimitiveAttributeIndex];

  this->primitiveCount  = primitiveEntity->getPrimitiveCount();
  this->vertexCount     = primitiveEntity->getVertexCount();

  boundingBoxes.resize(primitiveCount, false);
}
