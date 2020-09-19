#include "RayTracingSystem.h"

RayTracingSystem::RayTracingSystem()
  :allocator(NULL), camera(NULL)
{

}

RayTracingSystem::~RayTracingSystem()
{
  if (camera)
  {
    delete camera;
  }
}

void RayTracingSystem::init(ComputeInterface* compute, const uint maxRays)
{
  this->compute = compute;
  if (!allocator)
  {
    allocator = new RayTracingAllocator(compute);
    allocator->create(maxRays);
  }

  rays.create(compute, allocator->getHeap(COMPUTE_HEAP_RAYS));
  hits.create(compute);

  accelerationStruct = new AccelerationDataStruct();
  accelerationStruct->create(compute);
}

uint RayTracingSystem::getPrimCount()const
{
  return accelerationStruct->getPrimCount();
}

void RayTracingSystem::registerSphereBuffer(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count)
{
  accelerationStruct->registerSpheres(primitiveBuffer, radiusBuffer, radiusInfo, count);
}

void RayTracingSystem::updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  camera->update(projectionMatrix, modelviewMatrix);
}

void RayTracingSystem::render()
{
  camera->emitPrimaryRays(rays, RayStructPositionDirection);

  accelerationStruct->fullUpdate();

  hits.resize(rays.size(), false);

  accelerationStruct->intersectRays(hits.device(), HitStructDistanceIndex, rays.device(), RayStructPositionDirection);
}
