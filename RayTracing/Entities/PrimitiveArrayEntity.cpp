#include "RayTracingEntity.h"

PrimitiveArrayEntity::PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount)
{
  setRayTracingEntityId(this->identity, type, 0);
  this->primInfo.primType = type;
  this->primitiveCount    = primitiveCount;
}

PrimitiveArrayEntity::~PrimitiveArrayEntity()
{
  
}

RayTracingEntity* PrimitiveArrayEntity::createCopy()const
{
  PrimitiveArrayEntity *newEntity = new PrimitiveArrayEntity((RayTracingEntityType)this->primInfo.primType, primitiveCount);
  *newEntity = *this;
  return newEntity;
}

void PrimitiveArrayEntity::setMaterialId(MaterialId materialId)
{
  this->material = materialId;
}

RayTracingEntityId PrimitiveArrayEntity::getIdentity()const
{
  return identity;
}

uint PrimitiveArrayEntity::getPrimCount()const
{
  return primitiveCount;
}

void PrimitiveArrayEntity::update()
{

}
