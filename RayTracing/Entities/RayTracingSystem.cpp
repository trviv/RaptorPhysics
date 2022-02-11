#include "RayTracingSystem.h"

//#define DEBUG_RAY_TRACING_SYSTEM
static uint rearrangeMultiplier = 1;
static bool usePrimitiveInstancing = false;

uint RayTracingSystem::rayComputeUtilId[RayStructTypeMax] = {0, 0};
uint RayTracingSystem::maxPrimIndex = 0;

RayTracingSystem::RayTracingSystem()
  :allocator(NULL), camera(NULL), currentCamera(NULL)
{

}

RayTracingSystem::~RayTracingSystem()
{
  if (camera)
  {
    delete camera;
  }
}

bool RayTracingSystem::isAvailable()
{
  return camera;
}

void RayTracingSystem::init(ComputeInterface* compute, const uint maxRays)
{
  this->compute = compute;
  this->currentCamera = new DeviceArray<CameraStruct>(compute);
  this->currentCamera->resize(1, false);
  if (!allocator)
  {
//    allocator = new RayTracingAllocator(compute);
//    allocator->create(maxRays);
  }

  for (uint i=0; i<RAY_TRACING_SYSTEM_ARRAY_COUNT+1; i++)
  {
    rays[i].create(compute);
  }

  for (uint i=0; i<2; i++)
  {
    shadowRays[i].create(compute);
  }

  hits.create(compute);
  lights.create(compute);
  materials.create(compute);

  accelerationStruct = new PrimitiveInstanceAccelerationDataStruct(usePrimitiveInstancing);
  accelerationStruct->create(compute);

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("MaterialStruct.h");
  includeFiles.push_back("RayTracingStruct.h");
  includeFiles.push_back("Light.shader");

  const vector<string> rayUtilIncludeFiles = {"RayStructs.h"};
  rearrangeMultiplier = 8192 / compute->maxThreadsPerGroup();

  for (int r=0; r<RayStructTypeMax; r++)
  {
    for (int h=0; h<HitStructTypeMax; h++)
    {
      vector<string> oldType = {"RayStruct", "HitStruct", "RAYS_REARRANGE_MULTIPLIER"};
      vector<string> newType = {getRayStructName((RayStructType)r), getHitStructName((HitStructType)h), to_string(rearrangeMultiplier)};
      getRayStructDefines(oldType, newType, (RayStructType)r);
      getHitStructDefines(oldType, newType, (HitStructType)h);
      registerShader(compute, "RayTracingSystem.shader", &oldType, &newType);
      shadeIntersectionKernels[r][h] = programs.back().createKernel("shadeIntersection");
      processShadowRaysKernels[r][h] = programs.back().createKernel("processShadowRays");
    }

    map<ComputeUtilKey, string> rayUtilSetting;
    rayUtilSetting[ComputeUtilStructType]         = getRayStructName((RayStructType)r);
    rayUtilSetting[ComputeUtilStructMember]       = "maxDistance";
    rayUtilSetting[ComputeUtilStructMemberType]   = "uint";
    rayUtilSetting[ComputeUtilStructMemberSize]   = "4";
    rayUtilSetting[ComputeUtilBatchSize]          = "1";
    rayUtilSetting[ComputeUtilOnlyCompaction]     = "1";
    rayUtilSetting[ComputeUtilStructTypeIntegral] = "1";
    rayUtilSetting[ComputeUtilSkipParallelPrimitives] = "1";

    rayComputeUtilId[r] = ComputeUtil::create(compute, rayUtilSetting, &rayUtilIncludeFiles);
    reorderRaysKernels[r] = programs.back().createKernel("reorderRays");
  }

  map<ComputeUtilKey, string> maxPrimIndexSetting;
  maxPrimIndexSetting[ComputeUtilStructType]              = "uint";
  maxPrimIndexSetting[ComputeUtilOnlyReduce]              = "1";
  maxPrimIndexSetting[ComputeUtilCustomAddFunction]       = "maxReduce";
  maxPrimIndexSetting[ComputeUtilCustomReduceFunction]    = "maxReduceSimd";
  maxPrimIndexSetting[ComputeUtilStructTypeIntegral]      = "1";
  maxPrimIndexSetting[ComputeUtilSkipParallelPrimitives]  = "1";

  maxPrimIndex = ComputeUtil::create(compute, maxPrimIndexSetting);

  accumulateColor = programs[0].createKernel("accumulateColor");
  updateCameraKernel = programs[0].createKernel("updateCameraKernel");
  transformPrimitives = programs[0].createKernel("transformPrimitives");

  randomUints.create(compute);
  colorOutputBuffer.create(compute);
  accumulatedColorBuffer.create(compute);

  for (auto& i : registeredPrimitives)
  {
    i.clear();
  }

  indirectCount.create(compute);
  indirectCount.resize(256/4*3*RAY_TRACING_SYSTEM_ARRAY_COUNT, false);

  // initialize temporary arrays from single indirect array
  for (uint i=0; i<RAY_TRACING_SYSTEM_ARRAY_COUNT; i++)
  {
    validRayCount[i]   = ComputeMemory(*indirectCount.device(), 256*i*3,     4*4);
    currentWGCount[i]  = ComputeMemory(*indirectCount.device(), 256*(i*3+1), 4*4);
    currentRayCount[i] = ComputeMemory(*indirectCount.device(), 256*(i*3+2), 4*4);
  }

  maxIterations = 3;
}

uint RayTracingSystem::newEntityId()
{
  return (uint)registeredEntities.size();
}

uint RayTracingSystem::newEntityInstanceId(uint entityIndex)
{
  return (uint)entitiyInstances[entityIndex].size();
}

void RayTracingSystem::commit()
{
  lights.host()->clear();

  for (uint i=0; i<registeredEntities.size(); i++)
  {
    for (auto& entity : entitiyInstances[i])
    {
      switch (entity->getEntityCategory())
      {
        case RayTracingEntityPrimArray:
        {
          registeredPrimitives[entity->getPrimitiveType()].push_back(entity);

          if (entity == entitiyInstances[i].front())
          {
            PrimitiveAccelerationDataStruct *primitiveEntity = new PrimitiveAccelerationDataStruct();
            primitiveEntity->create(compute);
            primitiveEntity->bindEntity(registeredEntities[i]);
            primitiveEntity->fullBuild();
            accelerationStruct->registerPrimitiveADS(primitiveEntity);
          }

          accelerationStruct->registerPrimitiveInstance(entity);
          break;
        }
        case RayTracingEntityLight:
        {
          Light* light = (Light*)entity;
          light->update();
          lights.host()->push_back(*light);

          // add a new primitive and material representing the area light
          if (light->getEntityType() == RayTracingEntityLightArea)
          {
            Material *newMaterial = new Material(MaterialTypePlastic);
            newMaterial->emissive = Half4(light->color.x, light->color.y, light->color.z, 1.f);
            MaterialId materialIdentity = registerMaterial(newMaterial);
            setIdentityTwoSided(materialIdentity, true);
            setIdentityEntityNoShadow(materialIdentity, true);

            PrimitiveArrayEntity *prim = new PrimitiveArrayEntity(RayTracingEntityTriangles, 0, compute);
            real dim[3] = {1.f, 1.f, 0.f};
            prim->createBox(dim);
            registerAndInstantiateEntity(prim, materialIdentity, 1, &light->getTransform());
          }
          break;
        }
        default:
          break;
      }
    }
  }

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

const DeviceArray<colorType4>& RayTracingSystem::getColorOutputBuffer()const
{
  return accumulatedColorBuffer;
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
  const RayTracingEntityType entityType = entity->getEntityType();

  RayTracingEntityId entityId;
  resetIdentity(entityId);
  setRayTracingEntityId(entityId, entityType, newEntityId());
  entitiyInstances.push_back(vector<RayTracingEntity*>());
  registeredEntities.push_back(entity);

  // transform primitive vertex data
  if (entity->getEntityCategory() == RayTracingEntityPrimArray)
  {
    XAB primBound = ((PrimitiveArrayEntity*)entity)->getPrimBound();

    const Real3 bound = Real3(primBound.max) - Real3(primBound.min);
    const float maxRatio = max(bound.x, max(bound.y, bound.z));

    Matrix primMatrix;
    primMatrix.setIdentity();
    primMatrix.scale(Real3(2.f) / maxRatio);
    primMatrix.translate(Real3(0.f) - (Real3(primBound.min) + bound*0.5f));

    Matrix4 transform = primMatrix[TRANS];
    size_t workgroupSize[3], workgroupCount[3];

    uint maxIndex = entity->primInfo.vertexCount-1;

    transformPrimitives.setArg(entity->attributeBuffer[EntityPrimitiveAttributePosition], 0);
    transformPrimitives.setArg(&entity->attributeInfo[EntityPrimitiveAttributePosition], 1);
    transformPrimitives.setArg(&maxIndex, 2);

    compute->configureSize(workgroupSize, workgroupCount, maxIndex+1);

    transformPrimitives.setArg<Matrix4>(&transform, 3);
    compute->execute(transformPrimitives, workgroupSize, workgroupCount);
  }

  return entityId;
}

void RayTracingSystem::addEntityInstance(const RayTracingEntityId& registeredEntityId, const MaterialId& material, ushort instanceCount, const Matrix4* instanceTransforms)
{
  // get entity
  const uint entityId = getRayTracingEntityId(registeredEntityId);

  for (uint instance = 0; instance < instanceCount; instance++)
  {
    RayTracingEntityId entityInstanceId = registeredEntityId;
    setRayTracingInstanceId(entityInstanceId, newEntityInstanceId(entityId));

    RayTracingEntity* newEntity = registeredEntities[entityId]->createCopy();
    if (instanceTransforms)
    {
      newEntity->getTransform() = instanceTransforms[instance];
    }
    else
    {
      newEntity->getTransform().setIdentity();
    }
    if (material.identity != -1)
    {
      newEntity->setMaterialId(material);
    }
    entitiyInstances[entityId].push_back(newEntity);
  }
}

RayTracingEntityId RayTracingSystem::registerAndInstantiateEntity(RayTracingEntity* entity, const MaterialId& material, ushort instanceCount, const Matrix4* instanceTransforms)
{
  RayTracingEntityId entityId = registerEntity(entity);
  addEntityInstance(entityId, material, instanceCount, instanceTransforms);

  return entityId;
}

void RayTracingSystem::updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  camera->update(projectionMatrix, modelviewMatrix);
}

void RayTracingSystem::render(bool updatePrimitives)
{
  if (updatePrimitives)
  {
    accelerationStruct->fullBuild();
  }

  ComputeUtil* uintUtil = ComputeUtil::get(ComputeUtil::getUIntUtil(compute));

  RayStructType rayType   = RayStructPositionDirectionColor;
  HitStructType hitStruct = HitStructDistanceIndexIdentityNormal;
  RayStructType shadowRayType   = RayStructPositionDirectionColor;
  HitStructType shadowHitStruct = HitStructDistanceIdentity;

  camera->emitPrimaryRays(rays[0], rayType);
  uintUtil->copyBuffer(compute, camera->getRayCount()->device(), &currentRayCount[0], 0, 0, sizeof(uint)*4);

  size_t workgroupSize[3] = {compute->maxThreadsPerGroup(), 1, 1};
  colorOutputBuffer.resize(camera->width * camera->height, false);
  accumulatedColorBuffer.resize(camera->width * camera->height, false);

  uintUtil->clearBuffer(compute, colorOutputBuffer.device(), camera->width * camera->height * sizeof(colorType4) / sizeof(uint));
  uint rayCount = (rays[0].size() * 4) / getRayStructSize(rayType);
  hits.resize(rayCount * getHitStructSize(hitStruct) / 4, false);

  if (randomUints.size() != rayCount)
  {
    randomUints.resize(rayCount, false);
    randomUints.host()->resize(rayCount);
    for (uint i=0; i<rayCount; i++)
    {
      (*randomUints.host())[i] = rand() & RAND_MAX;
    }
    randomUints.syncDevice();
  }

  for (uint i=1; i<RAY_TRACING_SYSTEM_ARRAY_COUNT+1; i++)
  {
    rays[i].resize(rays[0].size(), false);
  }

  for (uint i=0; i<2; i++)
  {
    shadowRays[i].resize(lights.size() * rayCount * getRayStructSize(shadowRayType) / 4, false);
  }

  uint bufferIndex = 0;

  // update camera or increase the frame count 
  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, 1);
    updateCameraKernel.setArg(currentCamera->device(), 0);
    updateCameraKernel.setArg(camera->getDeviceCamera(), 1);

    compute->execute(updateCameraKernel, workgroupSize, workgroupCount);
  }

  for (uint iteration=0; iteration<maxIterations; iteration++, bufferIndex = (bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT)
  {
    uintUtil->configureWorkgroupCount(compute, &currentWGCount[bufferIndex], &currentRayCount[bufferIndex], workgroupSize);
    accelerationStruct->intersectRays(hits.device(), hitStruct, rays[bufferIndex].device(), rayType, &currentRayCount[bufferIndex], IntersectionTypeClosest);

#ifdef DEBUG_RAY_TRACING_SYSTEM
    indirectCount.syncHost();
    hits.syncHost();
    compute->sync();
#endif

    {
      ushort lightOffset = 0;
      ushort lightCount = lights.size();

      ComputeKernel& shadeIntersectionKernel = shadeIntersectionKernels[rayType][hitStruct];

      ComputeMemory* buffers[] = {
        colorOutputBuffer.device(),
        shadowRays[0].device(),
        rays[bufferIndex].device(),
        hits.device(),
        randomUints.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      shadeIntersectionKernel.setArgs(buffers, bufferCount);
      shadeIntersectionKernel.setArg(&currentRayCount[bufferIndex], bufferCount);
      shadeIntersectionKernel.setArg(lights.device(), bufferCount+1);
      shadeIntersectionKernel.setArg(&lightOffset, bufferCount+2);
      shadeIntersectionKernel.setArg(&lightCount, bufferCount+3);
      shadeIntersectionKernel.setArg(materials.device(), bufferCount+4);
      shadeIntersectionKernel.setArg(currentCamera->device(), bufferCount+5);
      shadeIntersectionKernel.setArg(&iteration, bufferCount+6);

      compute->execute(shadeIntersectionKernel, workgroupSize, &currentWGCount[bufferIndex], 0);

#ifdef DEBUG_RAY_TRACING_SYSTEM
      rays[bufferIndex].syncHost();
      shadowRays[0].syncHost();
      compute->sync();
#endif
    }

    // use different buffers for source and destination
    ComputeUtil::get(rayComputeUtilId[shadowRayType])->compactSparseArrayAndCopy(compute, &validRayCount[bufferIndex], shadowRays[1].device(), shadowRays[0].device(), &currentRayCount[bufferIndex], rayCount);

    uintUtil->configureWorkgroupCount(compute, &currentWGCount[bufferIndex], &validRayCount[bufferIndex], workgroupSize);

#ifdef DEBUG_RAY_TRACING_SYSTEM
      indirectCount.syncHost();
      compute->sync();
#endif

    accelerationStruct->intersectRays(hits.device(), shadowHitStruct, shadowRays[1].device(), shadowRayType, &validRayCount[bufferIndex], IntersectionTypeAny);

    {
      ComputeKernel& processShadowRaysKernel = processShadowRaysKernels[shadowRayType][shadowHitStruct];

      ComputeMemory* buffers[] = {
        colorOutputBuffer.device(),
        shadowRays[1].device(),
        hits.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      processShadowRaysKernel.setArgs(buffers, bufferCount);
      processShadowRaysKernel.setArg(&validRayCount[bufferIndex], bufferCount);

      compute->execute(processShadowRaysKernel, workgroupSize, &currentWGCount[bufferIndex], 0);
    }

    if (iteration < (maxIterations-1))
    {
      const bool reorderRays = true;

      ComputeUtil::get(rayComputeUtilId[rayType])->compactSparseArrayAndCopy(compute, &currentRayCount[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT],
        reorderRays ? rays[RAY_TRACING_SYSTEM_ARRAY_COUNT].device() : rays[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT].device(),
        rays[bufferIndex].device(), &currentRayCount[bufferIndex], rayCount);

      if (reorderRays)
      {
        size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * rearrangeMultiplier, 1, 1};

        uintUtil->configureWorkgroupCount(compute, &currentWGCount[bufferIndex], &currentRayCount[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT], workgroupSize);

        ComputeKernel& reorderRaysKernel = reorderRaysKernels[rayType];

        reorderRaysKernel.setArg(rays[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT].device(), 0);
        reorderRaysKernel.setArg(rays[RAY_TRACING_SYSTEM_ARRAY_COUNT].device(), 1);
        reorderRaysKernel.setArg(&currentRayCount[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT], 2);
        reorderRaysKernel.setSharedMemArg(sizeof(ushort)*2*workgroupSize[0]*workgroupSize[1]*workgroupSize[2], 3);

        workgroupSize[0] /= rearrangeMultiplier;
        compute->execute(reorderRaysKernel, workgroupSize, &currentWGCount[bufferIndex], 0);

#ifdef DEBUG_RAY_TRACING_SYSTEM
        rays[(bufferIndex+1)%RAY_TRACING_SYSTEM_ARRAY_COUNT].syncHost();
        compute->sync();
#endif
      }
    }
  }

  // accumulate color
  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount, 1024);

    ComputeMemory* buffers[] = {
      accumulatedColorBuffer.device(),
      colorOutputBuffer.device(),
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    accumulateColor.setArgs(buffers, bufferCount);
    accumulateColor.setArg(currentCamera->device(), bufferCount);
    accumulateColor.setArg(&rayCount, bufferCount+1);

    compute->execute(accumulateColor, workgroupSize, workgroupCount);
  }
}
