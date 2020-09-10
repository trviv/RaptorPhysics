#include "Scene.h"

Scene::Scene()
  :allocator(NULL)
{

}

void Scene::init(ComputeInterface* compute)
{
  this->compute = compute;
  if (!allocator)
  {
    allocator = new RayTracingAllocator(compute);
  }
}

void Scene::update()
{

}
