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

void RayTracingSystem::init(ComputeInterface* compute)
{
  this->compute = compute;
  if (!allocator)
  {
    allocator = new RayTracingAllocator(compute);
  }

  rays.create(compute);
}

void RayTracingSystem::update()
{
}


void RayTracingSystem::renderParticles(Window* window, ComputeMemory* particles, uint count)
{
  // update the camera using window
  camera->update(window->getCameraPosition(), window->getCameraUp(), window->getCameraFront());

  update();

  
}
