#include "PrimitiveInstanceAccelerationDataStruct.h"

//#define DEBUG_PI_ADS
#define RAY_TRAVERSAL_BVH_MAX_LEAFS 5
#define BVH_ADS_PERSISTENT_MULTIPLIER 1
#define RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE RAY_TRAVERSAL_BVH_MAX_LEAFS

PrimitiveInstanceAccelerationDataStruct::PrimitiveInstanceAccelerationDataStruct(bool usePrimitiveInstancing)
  :primitiveChanged(true), primitiveInstanceChanged(true), primitiveInstanceTransformsChanged(true), usePrimitiveInstancing(usePrimitiveInstancing),
  primitiveInstanceNodes((DeviceArray<PrimitiveInstanceADSLeaf>&)leafNodeBoundingBoxes)
{
}

PrimitiveInstanceAccelerationDataStruct::~PrimitiveInstanceAccelerationDataStruct()
{
}

void PrimitiveInstanceAccelerationDataStruct::validateBuild()const
{
  AccelerationDataStruct::validateBuild();

  if (primitiveChanged)
  {
    logComputeError("Acceleration Data Structure need to be build, before using!");
  }
  if (primitiveInstanceChanged)
  {
    logComputeError("Acceleration Data Structure need to be build, before using!");
  }
  if (primitiveInstanceTransformsChanged)
  {
    logComputeError("Acceleration Data Structure need to be build, before using!");
  }
}

void PrimitiveInstanceAccelerationDataStruct::initializeData()
{
  BoundingVolumeHierarchyADS::initializeData();

  primitiveInstanceTransforms.create(compute);
  primitiveADSResources.create(compute, NULL, true);

  if (!usePrimitiveInstancing)
  {
    vertexArray.create(compute);
    attributeArray.create(compute);
    vertexAttributeArray.create(compute);
  }

  systemSettings.create(compute);

  for (auto i : primitiveInstancesPerType)
  {
    i.clear();
  }
}

void PrimitiveInstanceAccelerationDataStruct::updatePointers()
{
  primitiveADSResources.syncDevicePointerBuffer();
}

void PrimitiveInstanceAccelerationDataStruct::createLeafBoundingBoxes()
{
  if (!usePrimitiveInstancing)
  {
    BoundingVolumeHierarchyADS::createLeafBoundingBoxes();
    return;
  }

  primitiveADSResources.syncDevicePointerBuffer();

  registerResources(updatePrimitiveInstanceData);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

  updatePrimitiveInstanceData.setArg(primitiveInstanceNodes.device(), 0);
  updatePrimitiveInstanceData.setArg(primitiveInstanceTransforms.device(), 1);
  updatePrimitiveInstanceData.setArg(primitiveADSResources.device(), 2);
  updatePrimitiveInstanceData.setArg(&primitiveCount, 3);

  compute->execute(updatePrimitiveInstanceData, workgroupSize, workgroupCount);

#ifdef DEBUG_PI_ADS
  primitiveInstanceTransforms.syncHost();
  primitiveInstanceNodes.syncHost();
  compute->sync();
#endif
}

void PrimitiveInstanceAccelerationDataStruct::assignLeafMortonCode()
{
  if (!usePrimitiveInstancing)
  {
    BoundingVolumeHierarchyADS::assignLeafMortonCode();
    return;
  }

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

  // assign morton code to primitive ADS bounding boxes
  assignMortonCode.setArg(primitiveLeafData.device(), 0);
  assignMortonCode.setArg(primitiveInstanceNodes.device(), 1);
  assignMortonCode.setArg(pointerSystemSettings, 2);
  assignMortonCode.setArg(&primitiveCount, 3);

  compute->execute(assignMortonCode, workgroupSize, workgroupCount);

#ifdef DEBUG_PI_ADS
  primitiveLeafData.syncHost();
  compute->sync();
#endif
}

void PrimitiveInstanceAccelerationDataStruct::registerCreateShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  BoundingVolumeHierarchyADS::registerCreateShaders();

  includeFiles.clear();

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "PrimitiveInstanceADSCreate.shader", NULL, NULL);

  updatePrimitiveInstanceData = programs.back().createKernel("updatePrimitiveInstanceData");
  collectPrimitives = programs.back().createKernel("collectPrimitives");
  if (usePrimitiveInstancing)
  {
    assignMortonCode = programs.back().createKernel("primitiveADSAssignMortonCode");
  }
}

void PrimitiveInstanceAccelerationDataStruct::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("BoundingVolumeHierarchyADSCreate.shader");
  includeFiles.push_back("AccelerationDataStructTraverse.shader");
  includeFiles.push_back("BoundingVolumeHierarchyADSTraverse.shader");

  vector<string> oldType = {"BVH_ADS_INTERSECT_RAY_BVH_FUNCTION", "ADS_TRAVERSAL_SHADER_PROGRAM"};
  vector<string> newType = {usePrimitiveInstancing ? "intersectRaysBVHPrimitiveInstances" : "intersectRaysBVHPrimitiveInstancesFlattened", "PrimitiveInstanceADSTraverse.shader"};

  if (usePrimitiveInstancing)
  {
    oldType.push_back("PRIMITIVE_INSTANCE_TRAVERSAL");
    newType.push_back("");
  }

  if (oldTypeArg) oldType.insert(oldType.end(), oldTypeArg->begin(), oldTypeArg->end());
  if (newTypeArg) newType.insert(newType.end(), newTypeArg->begin(), newTypeArg->end());

  BoundingVolumeHierarchyADS::registerTraverseShaders(&oldType, &newType);
}

void PrimitiveInstanceAccelerationDataStruct::registerResources(ComputeKernel& kernel)
{
  if (usePrimitiveInstancing)
  {
    for (auto& i : primitiveInstances)
    {
      i.first->registerResources(kernel);
    }
  }
  else
  {
    kernel.registerResource(treeInternalNodes.device());
    kernel.registerResource(leafParentNodeIndices.device());
    kernel.registerResource(nodeParentNodeIndices.device());
    kernel.registerResource(leafNodeBoundingBoxes.device());
    kernel.registerResource(treeNodeBoundingBoxes.device());
    kernel.registerResource(pointerVertexArray);
    kernel.registerResource(pointerAttributeArray);
    kernel.registerResource(pointerVertexAttributeArray);
    kernel.registerResource(pointerSystemSettings);
  }
}

void PrimitiveInstanceAccelerationDataStruct::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                                          const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings)
{
  logComputeError("Bind buffers method is not available with PrimitiveAccelerationDataStruct!");
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS)
{
  needsRebuild = true;
  primitiveChanged = true;
  primitiveInstanceTransformsChanged = true;

  primitiveEntityADS->validateBuild();

  if (!primitiveInstances.count(primitiveEntityADS))
  {
    primitiveInstances[primitiveEntityADS] = vector<const RayTracingEntity*>();
  }
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveInstance(const RayTracingEntity* entityInstance)
{
  needsRebuild = true;
  primitiveInstanceChanged = true;
  primitiveInstanceTransformsChanged = true;

  const auto instanceIdentity = getRayTracingEntityId(entityInstance->getIdentity());
  for (auto& prim : primitiveInstances)
  {
    if (getRayTracingEntityId(prim.first->primitiveEntity->getIdentity()) == instanceIdentity)
    {
      prim.second.push_back(entityInstance);
      primitiveCount++;
      break;
    }
  }

  primitiveInstancesPerType[entityInstance->getPrimitiveType()].push_back(entityInstance);
}

void PrimitiveInstanceAccelerationDataStruct::resizePrimitiveArray()
{
  systemSettings.host()->resize(1);

  uint primOffset   = 0;
  uint vertexOffset = 0;

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    for (const auto& primInstance : primitiveInstancesPerType[i])
    {
      primOffset    += primInstance->getPrimitiveCount();
      vertexOffset  += primInstance->getPrimitiveVertexCount();
    }

    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  primOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);
    systemSettings.host()->at(0).globalOffsets[i] = primInfo;
  }

  vertexArray.resize(vertexOffset, false);
  attributeArray.resize(primOffset, false);
  vertexAttributeArray.resize(vertexOffset, false);

  systemSettings.syncDevice();

  BoundingVolumeHierarchyADS::bindBuffers(vertexArray.device(), attributeArray.device(), vertexAttributeArray.device(), &systemSettings);
}

void PrimitiveInstanceAccelerationDataStruct::composePrimitiveArray()
{
  uint vertexOffset   = 0;
  uint primOffset     = 0;
  uint primBatchSize  = 8;

  for (const auto& primInstances : primitiveInstancesPerType)
  {
    for (const auto& primInstance : primInstances)
    {
      const auto& prim = *primInstance;
      uint primBatchCount = mAlignBy(prim.getPrimitiveCount(), primBatchSize);
      uint primType = prim.getPrimitiveType();

      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

      collectPrimitives.setArg(vertexArray.device(), 0);
      collectPrimitives.setArg(attributeArray.device(), 1);
      collectPrimitives.setArg(vertexAttributeArray.device(), 2);
      uint nextBindIndex = prim.bindToShader(collectPrimitives, 3);
      collectPrimitives.setArg(&prim.getMaterialId(), nextBindIndex);
      collectPrimitives.setArg(&primBatchSize, nextBindIndex+1);
      collectPrimitives.setArg(&prim.getPrimitiveCount(), nextBindIndex+2);
      collectPrimitives.setArg(&primType, nextBindIndex+3);
      collectPrimitives.setArg(&primOffset, nextBindIndex+4);
      collectPrimitives.setArg(&vertexOffset, nextBindIndex+5);
      collectPrimitives.setArg<const Matrix4>(&prim.getTransform(), nextBindIndex+6);

      compute->execute(collectPrimitives, workgroupSize, workgroupCount);

#ifdef DEBUG_PI_ADS
      vertexArray.syncHost();
      attributeArray.syncHost();
      vertexAttributeArray.syncHost();
      compute->sync();
#endif

      vertexOffset  += prim.getPrimitiveVertexCount();
      primOffset    += prim.getPrimitiveCount();
    }
  }
}

void PrimitiveInstanceAccelerationDataStruct::fullBuild()
{
  if (!needsRebuild) return;

  if (!usePrimitiveInstancing)
  {
    resizePrimitiveArray();
    composePrimitiveArray();

    BoundingVolumeHierarchyADS::fullBuild();

    primitiveADSResources.host()->resize(9);
    primitiveADSResources.host()->at(0) = treeInternalNodes.device();
    primitiveADSResources.host()->at(1) = leafParentNodeIndices.device();
    primitiveADSResources.host()->at(2) = nodeParentNodeIndices.device();
    primitiveADSResources.host()->at(3) = leafNodeBoundingBoxes.device();
    primitiveADSResources.host()->at(4) = treeNodeBoundingBoxes.device();
    primitiveADSResources.host()->at(5) = pointerVertexArray;
    primitiveADSResources.host()->at(6) = pointerAttributeArray;
    primitiveADSResources.host()->at(7) = pointerVertexAttributeArray;
    primitiveADSResources.host()->at(8) = pointerSystemSettings;

    updatePointers();

    primitiveChanged = false;
    primitiveInstanceChanged = false;
    primitiveInstanceTransformsChanged = false;


    // initialize just to have valid kernel input
    primitiveInstanceTransforms.resize(1, false);
    return;
  }

  primitiveInstanceNodes.resize(primitiveCount, false);
  primitiveInstanceTransforms.resize(primitiveCount * 3 * 2, false);

  if (primitiveInstanceChanged)
  {
    primitiveADSResources.host()->clear();
    uint index = 0;

    for (const auto& instances : primitiveInstances)
    {
      if (instances.second.size())
      {
        primitiveADSResources.host()->push_back(instances.first->treeInternalNodes.device());
        primitiveADSResources.host()->push_back(instances.first->leafParentNodeIndices.device());
        primitiveADSResources.host()->push_back(instances.first->nodeParentNodeIndices.device());
        primitiveADSResources.host()->push_back(instances.first->leafNodeBoundingBoxes.device());
        primitiveADSResources.host()->push_back(instances.first->treeNodeBoundingBoxes.device());
        primitiveADSResources.host()->push_back(instances.first->pointerVertexArray);
        primitiveADSResources.host()->push_back(instances.first->pointerAttributeArray);
        primitiveADSResources.host()->push_back(instances.first->pointerVertexAttributeArray);
        primitiveADSResources.host()->push_back(instances.first->pointerSystemSettings);
      }

      for (const auto& instance : instances.second)
      {
        real invMatrix[16];
        Matrix4::invert(invMatrix, (const real*)&instance->getTransform());

        compute->copyFromHost(primitiveInstanceTransforms.device(), index * 3 * sizeof(float) * 4, sizeof(float) * 4 * 3, &instance->getTransform(), false);
        compute->copyFromHost(primitiveInstanceTransforms.device(), (primitiveCount + index) * 3 * sizeof(float) * 4, sizeof(float) * 4 * 3, invMatrix, false);

        PrimitiveInstanceADSLeaf nodeData;
        nodeData.bounds             = instances.first->primitiveEntity->getPrimBound();
        nodeData.primitiveADSIndex  = index;
        nodeData.primitiveInstance  = instance->getIdentity();
        compute->copyFromHost(primitiveInstanceNodes.device(), (primitiveCount + index) * 3 * sizeof(float) * 4, sizeof(float) * 4 * 3, invMatrix, false);

        index++;
      }
    }

#ifdef DEBUG_PI_ADS
    primitiveInstanceTransforms.syncHost();
    primitiveInstanceNodes.syncHost();
    compute->sync();
#endif
  }

  if (primitiveInstanceTransformsChanged)
  {
    // don't pass systemSettings so primitiveCount does not change
    BoundingVolumeHierarchyADS::bindBuffers(NULL, NULL, NULL, NULL);

    systemSettings.resize(1, false);
    pointerSystemSettings = systemSettings.device();

    BoundingVolumeHierarchyADS::fullBuild();
  }

  needsRebuild = false;
  primitiveChanged = false;
  primitiveInstanceChanged = false;
  primitiveInstanceTransformsChanged = false;
}

void PrimitiveInstanceAccelerationDataStruct::setInstanceTransformsChanged()
{
  primitiveInstanceTransformsChanged = true;
}

void PrimitiveInstanceAccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                                            uint rayCount, IntersectionType intersectionType)
{
  BoundingVolumeHierarchyADS::intersectRays(hits, hitType, rays, rayType, rayCount, intersectionType);
}

void PrimitiveInstanceAccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                                            const ComputeMemory* rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * BVH_ADS_PERSISTENT_MULTIPLIER, 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);
    workgroupSize[0] /= BVH_ADS_PERSISTENT_MULTIPLIER;

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(rayCount, 2);
    intersectionKernel.setArg(pointerTreeInternalNodes, 3);
    intersectionKernel.setArg(pointerLeafParentNodeIndices, 4);
    intersectionKernel.setArg(pointerNodeParentNodeIndices, 5);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 6);
    intersectionKernel.setArg(pointerTreeNodeBoundingBoxes, 7);
    intersectionKernel.setArg(primitiveADSResources.device(), 8);
    intersectionKernel.setArg(primitiveInstanceTransforms.device(), 9);
    intersectionKernel.setArg(&primitiveCount, 10);
    intersectionKernel.setArg(workgroupCount.device(), 11);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * ((usePrimitiveInstancing ? 4 : 0) + RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE), (size_t)4), 12);
    registerResources(intersectionKernel);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);
  }
#ifdef DEBUG_PI_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
}
