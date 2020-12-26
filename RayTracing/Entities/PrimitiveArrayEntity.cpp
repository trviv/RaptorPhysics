#include "RayTracingEntity.h"

PrimitiveArrayEntity::PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount)
{
  this->identity.identity = 0;
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

void PrimitiveArrayEntity::createBox(const real dim[])
{
  const real boxVertices[] = {
    -dim[0], -dim[1], -dim[2], 0.f, +dim[0], -dim[1], -dim[2], 0.f, -dim[0], +dim[1], -dim[2], 0.f, +dim[0], +dim[1], -dim[2], 0.f,
    -dim[0], -dim[1], +dim[2], 0.f, +dim[0], -dim[1], +dim[2], 0.f, -dim[0], +dim[1], +dim[2], 0.f, +dim[0], +dim[1], +dim[2], 0.f,
  };
  const uint indices[] = {0, 1, 2, 2, 1, 3, 4, 1, 0, 5, 1, 4, 0, 2, 4, 4, 2, 6, 6, 5, 4, 7, 5, 6, 2, 3, 6, 6, 3, 7, 5, 3, 1, 7, 3, 5};

  deviceData->host()->clear();
  for (auto i : boxVertices)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  for (auto i : indices)
  {
    deviceData->host()->push_back(i);
  }
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(sizeof(boxVertices)/sizeof(real), 1));
  primitiveCount = 12;
}

void PrimitiveArrayEntity::createSphere(const real radius)
{
  const real sphereCenter[] = {0.f, 0.f, 0.f, 0.f};
  const real sphereRadius[] = {radius};

  deviceData->host()->clear();
  for (auto i : sphereCenter)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  for (auto i : sphereRadius)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeRadius, deviceData->device(), PackingInfo(sizeof(sphereCenter)/sizeof(real), 1));
  primitiveCount = 1;
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
