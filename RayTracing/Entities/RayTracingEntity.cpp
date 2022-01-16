#include "RayTracingEntity.h"

RayTracingAllocator::RayTracingAllocator(ComputeInterface* compute)
  :rayHeap(compute)
{}

void RayTracingAllocator::create(uint initialRays)
{
  rayHeap.create(initialRays * sizeof(Ray));
}

ComputeHeap* RayTracingAllocator::getHeap(RayTracingHeapEnum type)
{
  switch (type)
  {
  case COMPUTE_HEAP_RAYS:
    return &rayHeap;
  }
  return NULL;
}

EntityPrimAttributes::EntityPrimAttributes()
{
  for (uint i=0; i<EntityPrimitiveAttributeMax; i++)
  {
    attributeBuffer[i]  = 0;
  }
}

uint EntityPrimAttributes::bindToShader(ComputeKernel& kernel, uint startIndex)
{
  kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex);
  kernel.setArg(&attributeInfo[EntityPrimitiveAttributePosition], startIndex+1);
  if (primInfo.primitiveType == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributeRadius], startIndex+2);
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeRadius], startIndex+3);
  }
  else
  if (primInfo.primitiveType == PrimitiveTriangle || primInfo.primitiveType == PrimitiveIndexedTriangle || primInfo.primitiveType == PrimitiveIndexedQuad)
  {
    if (attributeInfo[EntityPrimitiveAttributeIndex].strideIn4Bytes)
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributeIndex], startIndex+2);
    }
    else
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex+2);
    }
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeIndex], startIndex+3);
  }
  kernel.setArg(attributeBuffer[EntityPrimitiveAttributeNormal], startIndex+4);
  kernel.setArg(&attributeInfo[EntityPrimitiveAttributeNormal], startIndex+5);

  return startIndex+6;
}

void EntityPrimAttributes::setAttribute(EntityPrimitiveAttributeType type, const ComputeMemory* attributeBuffer, PackingInfo attributePacking)
{
  this->attributeBuffer[type] = attributeBuffer;
  this->attributeInfo[type]   = attributePacking;
}

const ComputeMemory* EntityPrimAttributes::operator[](EntityPrimitiveAttributeType attributeType)const
{
  return attributeBuffer[attributeType];
}

RTPrimitiveType EntityPrimAttributes::getPrimitiveType()const
{
  return (RTPrimitiveType)primInfo.primitiveType;
}

uint EntityPrimAttributes::getPrimitiveCount()const
{
  return primInfo.primitiveCount;
}

uint EntityPrimAttributes::getVertexCount()const
{
  return primInfo.vertexCount;
}

RayTracingEntity::RayTracingEntity(ComputeInterface* compute)
  :compute(compute)
{
  transform.setIdentity();
  deviceData = NULL;
  materialId.identity = -1;
}

void RayTracingEntity::setMaterialId(MaterialId materialId)
{
  this->materialId = materialId;
}

const XAB& RayTracingEntity::getPrimBound()const
{
  return primBound;
}

Matrix4& RayTracingEntity::getTransform()
{
  return transform;
}

const Matrix4& RayTracingEntity::getTransform()const
{
  return transform;
}

RayTracingEntityType RayTracingEntity::getEntityCategory()const
{
  return getRayTracingEntityCategory((RayTracingEntityType)getRayTracingEntityType(getIdentity()));
}

MaterialId& RayTracingEntity::getMaterialId()
{
  return materialId;
}
