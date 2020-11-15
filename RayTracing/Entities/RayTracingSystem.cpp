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

void RayTracingSystem::registerPrimitive(RayTracingEntityType type, RayTracingEntity* entity)
{
  EntityPrimAttributes primitiveInfo = *entity;
  const uint count = ((PrimitiveArrayEntity*)entity)->getPrimCount();

  primitiveInfo.primInfo.indexCount  = count;
  primitiveInfo.primInfo.vertexCount = count;

  ushort primType = 0;
  switch (type)
  {
    case RayTracingEntitySpheres:
      primType = PrimitiveSphere;
      break;
    case RayTracingEntityTriangles:
      primType = PrimitiveTriangle;
      primitiveInfo.primInfo.vertexCount *= 3;
      break;
    default:
      logComputeError("Invalid entity type sent for registration!");
  }
  primitiveInfo.primInfo.primType = primType;
  registeredPrimitives[primType].push_back(primitiveInfo);
}

void RayTracingSystem::composePrimitiveArray()
{
  uint indexOffset = 0;
  uint vertexOffset = 0;
  uint primBatchSize = 8;

  for (auto& rp : registeredPrimitives)
  {
    for (uint i=0; i<rp.size(); i++)
    {
      auto& prim = rp[i];
      uint primBatchCount = mAlignBy(prim.primInfo.indexCount, primBatchSize);
      uint primType = prim.primInfo.primType;

      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

      collectPrimitives.setArg(vertexArray.device(), 0);
      collectPrimitives.setArg(attributeArray.device(), 1);
      uint nextBindIndex = prim.bindToShader(collectPrimitives, 2);
      collectPrimitives.setArg(&prim.materialId, nextBindIndex);
      collectPrimitives.setArg(&primBatchSize, nextBindIndex+1);
      collectPrimitives.setArg(&prim.primInfo.indexCount, nextBindIndex+2);
      collectPrimitives.setArg(&primType, nextBindIndex+3);
      collectPrimitives.setArg(&indexOffset, nextBindIndex+4);
      collectPrimitives.setArg(&vertexOffset, nextBindIndex+5);

      compute->execute(collectPrimitives, workgroupSize, workgroupCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
      vertexArray.syncHost();
      attributeArray.syncHost();
      compute->sync();
#endif

      indexOffset += prim.primInfo.indexCount;
      vertexOffset += prim.primInfo.vertexCount;
    }
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
  materials.create(compute);
  shadowRays.create(compute);

  accelerationStruct = new AccelerationDataStruct();
  accelerationStruct->create(compute);

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("MaterialStruct.h");
  includeFiles.push_back("RayTracingStruct.h");
  includeFiles.push_back("Light.shader");

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
      processShadowRaysKernels[r][h] = programs.back().createKernel("processShadowRays");
    }
  }

  collectPrimitives = programs[0].createKernel("collectPrimitives");

  colorOutputBuffer.create(compute);
  systemSettings.create(compute);
  vertexArray.create(compute);
  attributeArray.create(compute);

  for (auto& i : registeredPrimitives)
  {
    i.clear();
  }
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
  lights.host()->clear();

  for (auto& instances : entities)
  {
    for (uint i=1; i<instances.size(); i++)
    {
      RayTracingEntity* entity = instances[i];
      const RayTracingEntityType entityType = (RayTracingEntityType)getRayTracingEntityType(entity->getIdentity());
      const RayTracingEntityType entityCategory = getRayTracingEntityCategory(entityType);

      switch (entityCategory)
      {
        case RayTracingEntityPrimArray:
          registerPrimitive(entityType, entity);
          break;
        case RayTracingEntityLight:
        {
          Light* light = (Light*)entity;
          light->update();
          lights.host()->push_back(*light);
        }
          break;
        default:
          break;
      }
    }
  }

  systemSettings.host()->resize(1);
  
  uint indexOffset  = 0;
  uint vertexOffset = 0;

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    for (const auto& p : registeredPrimitives[i])
    {
      indexOffset  += p.primInfo.indexOffset;
      vertexOffset += p.primInfo.vertexOffset;
    }

    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  indexOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);
    systemSettings.host()->at(0).globalOffsets[i] = primInfo;
  }

  vertexArray.resize(vertexOffset, false);
  attributeArray.resize(indexOffset, false);

  systemSettings.syncDevice();

  accelerationStruct->commit(&vertexArray, &attributeArray, &systemSettings);
  lights.syncDevice();
  materials.syncDevice();
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

MaterialId RayTracingSystem::registerMaterial(Material* material)
{
  MaterialId ret;
  ret.identity = (uint)materials.host()->size();
  materials.host()->push_back(*material);
  return ret;
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

void RayTracingSystem::addEntityInstance(const RayTracingEntityId& registeredEntityId, ushort instanceCount, const Matrix4* instanceTransforms)
{
  // get entity
  const uint entityId = getRayTracingEntityId(registeredEntityId);
  const RayTracingEntityType entityType = (RayTracingEntityType)getRayTracingEntityType(registeredEntityId);

  for (uint instance = 0; instance < instanceCount; instance++)
  {
    RayTracingEntityId entityInstanceId = registeredEntityId;
    setRayTracingInstanceId(entityInstanceId, newEntityInstanceId(entityId));

    RayTracingEntity* newEntity = entities[entityId][0]->createCopy();
    if (instanceTransforms)
    {
      newEntity->getTransform() = instanceTransforms[instance];
    }
    else
    {
      newEntity->getTransform().setIdentity();
    }
    entities[entityId].push_back(newEntity);
  }
}

RayTracingEntityId RayTracingSystem::registerAndInstantiateEntity(RayTracingEntity* entity, ushort instanceCount, const Matrix4* instanceTransforms)
{
  RayTracingEntityId entityId = registerEntity(entity);
  addEntityInstance(entityId, instanceCount, instanceTransforms);

  return entityId;
}

void RayTracingSystem::updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  camera->update(projectionMatrix, modelviewMatrix);
}

void RayTracingSystem::render()
{
  composePrimitiveArray();

  RayStructType rayType   = RayStructPositionDirectionColor;
  HitStructType hitStruct = HitStructDistanceIndexNormal;
  RayStructType shadowRayType   = RayStructPositionDirectionColor;
  HitStructType shadowHitStruct = HitStructDistanceIndexNormal;

  camera->emitPrimaryRays(rays, rayType);
  colorOutputBuffer.resize(camera->width * camera->height, false);

  uint rayCount = (rays.size() * 4) / getRayStructSize(rayType);

  accelerationStruct->fullBuild();

  hits.resize(rayCount * getHitStructSize(hitStruct) / 4, false);
  shadowRays.resize(lights.size() * rayCount * getRayStructSize(shadowRayType) / 4, false);

  accelerationStruct->intersectRays(hits.device(), hitStruct, rays.device(), rayType, rayCount, IntersectionTypeClosest);

#ifdef DEBUG_RAY_TRACING_SYSTEM
  hits.syncHost();
  compute->sync();
#endif
  
  {
    ushort lightOffset = 0;
    ushort lightCount = lights.size();

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
    shadeIntersectionKernel.setArg(&lightOffset, bufferCount+2);
    shadeIntersectionKernel.setArg(&lightCount, bufferCount+3);
    shadeIntersectionKernel.setArg(materials.device(), bufferCount+4);

    compute->execute(shadeIntersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
    rays.syncHost();
    shadowRays.syncHost();
    compute->sync();
#endif
  }

  accelerationStruct->intersectRays(hits.device(), shadowHitStruct, shadowRays.device(), shadowRayType, rayCount, IntersectionTypeAny);

  {
    ComputeKernel& processShadowRaysKernel = processShadowRaysKernels[shadowRayType][hitStruct];

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    ComputeMemory* buffers[] = {
      colorOutputBuffer.device(),
      shadowRays.device(),
      hits.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    processShadowRaysKernel.setArgs(buffers, bufferCount);
    processShadowRaysKernel.setArg(&rayCount, bufferCount);

    compute->execute(processShadowRaysKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_RAY_TRACING_SYSTEM
    compute->sync();
#endif
  }
}
