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

RayTracingEntity::RayTracingEntity(ComputeInterface* compute)
  :compute(compute)
{
  transform.setIdentity();
}

Matrix4& RayTracingEntity::getTransform()
{
  return transform;
}
