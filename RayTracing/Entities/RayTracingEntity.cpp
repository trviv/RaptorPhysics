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
  materialId.identity = -1;
}

uint EntityPrimAttributes::bindToShader(ComputeKernel& kernel, uint startIndex)
{
  if (primInfo.primType == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex);
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributePosition], startIndex+1);
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributeRadius], startIndex+2);
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeRadius], startIndex+3);
    return startIndex+4;
  }
  else
  if (primInfo.primType == PrimitiveTriangle)
  {
    kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex);
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributePosition], startIndex+1);
    if (attributeInfo[EntityPrimitiveAttributeIndex].strideIn4Bytes)
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributeIndex], startIndex+2);
    }
    else
    {
      kernel.setArg(attributeBuffer[EntityPrimitiveAttributePosition], startIndex+2);
    }
    kernel.setArg(&attributeInfo[EntityPrimitiveAttributeIndex], startIndex+3);
    return startIndex+4;
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
  deviceData = NULL;
}

void RayTracingEntity::setMaterialId(MaterialId materialId)
{
  this->materialId = materialId;
}

Matrix4& RayTracingEntity::getTransform()
{
  return transform;
}
