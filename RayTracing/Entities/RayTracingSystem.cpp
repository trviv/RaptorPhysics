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
  lights.create(compute);
  shadowRays.create(compute);

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
      vector<string> oldType = {"RayStruct", "HitStruct"};
      vector<string> newType = {getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
      getRayStructDefines(oldType, newType, (RayStructType)r);
      getHitStructDefines(oldType, newType, (HitStructType)h);
      registerShader(compute, "RayTracingSystem.shader", &oldType, &newType);
      shadeIntersectionKernels[r][h] = programs.back().createKernel("shadeIntersection");
    }
  }

  colorOutputBuffer.create(compute);
}

uint RayTracingSystem::newEntityId()
{
  return (uint)entities.size();
}

uint RayTracingSystem::newEntityInstanceId(uint entityIndex)
{
  return (uint)entities[entityIndex].size() - 1;
}

void RayTracingSystem::commit()
{
  accelerationStruct->commit();

  lights.host()->clear();

  for (auto& instances : entities)
  {
    for (uint i=1; i<instances.size(); i++)
    {
      RayTracingEntity* entity = instances[i];
      const RayTracingEntityType entityType = (RayTracingEntityType)getRayTracingEntityType(entity->getIdentity());
      if (entityType & RayTracingEntityLight)
      {
        Light* light = (Light*)entity;
        light->update();
        lights.host()->push_back(*light);
      }
    }
  }

  lights.syncDevice();
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

void RayTracingSystem::registerTriangleBuffer(const ComputeMemory* primitiveBuffer, const ComputeMemory* indexBuffer, PackingInfo indexInfo, uint count)
{
  accelerationStruct->registerTriangles(primitiveBuffer, indexBuffer, indexInfo, count);
}

RayTracingEntityId RayTracingSystem::registerEntity(RayTracingEntity* entity)
{
  RayTracingEntityId entityId;
  resetIdentity(entityId);
  setRayTracingEntityId(entityId, getRayTracingEntityType(entity->getIdentity()), newEntityId());
  entities.push_back(vector<RayTracingEntity*>());
  entities.back().push_back(entity);

  return entityId;
}

void RayTracingSystem::addEntityInstance(const RayTracingEntityId registeredEntityId, const ushort instanceCount, const Matrix4* instanceTransforms)
{
  // get entity
  const uint entityId = getRayTracingEntityId(registeredEntityId);
  const RayTracingEntityType entityType = (RayTracingEntityType)getRayTracingEntityType(registeredEntityId);

  for (uint instance = 0; instance < instanceCount; instance++)
  {
    RayTracingEntityId entityInstanceId = registeredEntityId;
    setRayTracingInstanceId(entityInstanceId, newEntityInstanceId(entityId));

    RayTracingEntity* newEntity = entities[entityId][0]->createCopy();
    newEntity->getTransform() = instanceTransforms[instance];
    entities[entityId].push_back(newEntity);
  }
}

void RayTracingSystem::updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  camera->update(projectionMatrix, modelviewMatrix);
}

void RayTracingSystem::render()
{
  RayStructType rayType   = RayStructPositionDirectionColor;
  HitStructType hitStruct = HitStructDistanceIndex;
  RayStructType shadowRayType = RayStructPositionDirectionColor;

  camera->emitPrimaryRays(rays, rayType);
  colorOutputBuffer.resize(camera->width * camera->height, false);

  uint rayCount = (rays.size() * 4) / getRayStructSize(rayType);
  uint lightCount = lights.size();

  accelerationStruct->fullBuild();

  hits.resize(rayCount * getHitStructSize(hitStruct) / 4, false);
  shadowRays.resize(lights.size() * rayCount * getRayStructSize(shadowRayType) / 4, false);

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

    ComputeMemory* buffers[] = {
      shadowRays.device(),
      rays.device(),
      colorOutputBuffer.device(),
      hits.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    shadeIntersectionKernel.setArgs(buffers, bufferCount);
    shadeIntersectionKernel.setArg(&rayCount, bufferCount);
    shadeIntersectionKernel.setArg(lights.device(), bufferCount+1);
    shadeIntersectionKernel.setArg(&lightCount, bufferCount+2);

    compute->execute(shadeIntersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
    rays.syncHost();
    compute->sync();
#endif
  }
}
