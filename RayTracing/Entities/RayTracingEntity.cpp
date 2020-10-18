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
  if (primInfo.primType == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex);
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributeRadius], startIndex+1);
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeRadius], startIndex+2);
    return startIndex+3;
  }
  else
  if (primInfo.primType == PrimitiveTriangle)
  {
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex);
    if (attributeInfo[EntityPrimitiveAttributeIndex].strideIn4Bytes)
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributeIndex], startIndex+1);
    }
    else
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex+1);
    }
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeIndex], startIndex+2);
    return startIndex+3;
  }

  return startIndex;
}

void EntityPrimAttributes::setAttribute(EntityPrimitiveAttributeType type, const ComputeMemory* attributeBuffer, PackingInfo attributePacking)
{
  this->attributeBuffer[type] = attributeBuffer;
  this->attributeInfo[type]   = attributePacking;
}

RayTracingEntity::RayTracingEntity(ComputeInterface* compute)
  :compute(compute)
{
  transform.setIdentity();
}

Matrix4& RayTracingEntity::getTransform()
{
  return transform;
}
