#include "RayTracingSystem.h"

//#define DEBUG_RAY_TRACING_SYSTEM

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

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("RayTracingStruct.h");

  for (int r=0; r<RayStructTypeMax; r++)
  {
    for (int h=0; h<HitStructTypeMax; h++)
    {
      const vector<string> oldType = {"RayStruct", "HitStruct"};
      const vector<string> newType = {getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
      registerShader(compute, "RayTracingSystem.shader", &oldType, &newType);
      shadeIntersectionKernels[r][h] = programs.back().createKernel("shadeIntersection");
    }
  }

  colorOutputBuffer.create(compute);
}

void RayTracingSystem::commit()
{
  accelerationStruct->commit();
}

uint RayTracingSystem::getPrimCount()const
{
  return accelerationStruct->getPrimCount();
}

const Camera& RayTracingSystem::getCameraStruct()const
{
  return *camera;
}

const DeviceArray<uint>& RayTracingSystem::getColorOutputBuffer()const
{
  return colorOutputBuffer;
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
  RayStructType rayType   = RayStructPositionDirection;
  HitStructType hitStruct = HitStructDistanceIndex;

  camera->emitPrimaryRays(rays, rayType);
  colorOutputBuffer.resize(camera->width * camera->height, false);

  uint rayCount = (rays.size() * 4) / getRayStructSize(rayType);

  accelerationStruct->fullBuild();

  hits.resize(rayCount * getHitStructSize(hitStruct) / 4, false);

  accelerationStruct->intersectRays(hits.device(), hitStruct, rays.device(), rayType, rayCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
  hits.syncHost();
  compute->sync();
#endif

  {
    ComputeKernel& shadeIntersectionKernel = shadeIntersectionKernels[rayType][hitStruct];

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    shadeIntersectionKernel.setArg(rays.device(), 0);
    shadeIntersectionKernel.setArg(colorOutputBuffer.device(), 1);
    shadeIntersectionKernel.setArg(hits.device(), 2);
    shadeIntersectionKernel.setArg(&rayCount, 3);

    compute->execute(shadeIntersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
    rays.syncHost();
    compute->sync();
#endif
  }
}
