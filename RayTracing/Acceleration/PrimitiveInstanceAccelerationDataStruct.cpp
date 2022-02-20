#include "PrimitiveInstanceAccelerationDataStruct.h"

//#define DEBUG_PI_ADS

PrimitiveInstanceAccelerationDataStruct::PrimitiveInstanceAccelerationDataStruct(bool usePrimitiveInstancing)
  :primitiveChanged(true), primitiveInstanceChanged(true), primitiveInstanceTransformsChanged(true), usePrimitiveInstancing(usePrimitiveInstancing),
  primitiveInstanceNodes((DeviceArray<PrimitiveInstanceADSLeaf>&)leafNodeBoundingBoxes)
{
  sharedMemoryStride = usePrimitiveInstancing ? sharedMemoryStride + 4 : sharedMemoryStride;
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

void PrimitiveInstanceAccelerationDataStruct::fullBuild()
{
  if (!needsRebuild) return;

  if (!usePrimitiveInstancing)
  {
    resizePrimitiveArray();
    BoundingVolumeHierarchyADS::bindBuffers(vertexArray.device(), attributeArray.device(), vertexAttributeArray.device(), &systemSettings);
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
  primitiveInstanceNodes.host()->resize(primitiveCount);
  primitiveInstanceTransforms.resize(primitiveCount * 2, false);
  primitiveInstanceTransforms.host()->resize(primitiveCount * 2);

  if (primitiveInstanceChanged)
  {
    primitiveADSResources.host()->clear();
    uint instanceIndex = 0;
    uint adsIndex = -1;
    const uint matrixSize = 4 * sizeof(float) * 4;

    for (const auto& instances : primitiveInstances)
    {
      if (!instances.second.size()) continue;

      const auto& primitiveADS = instances.first;
      adsIndex++;
      primitiveADSResources.host()->push_back(primitiveADS->pointerTreeInternalNodes);
      primitiveADSResources.host()->push_back(primitiveADS->pointerLeafParentNodeIndices);
      primitiveADSResources.host()->push_back(primitiveADS->pointerNodeParentNodeIndices);
      primitiveADSResources.host()->push_back(primitiveADS->pointerLeafNodeBoundingBoxes);
      primitiveADSResources.host()->push_back(primitiveADS->pointerTreeNodeBoundingBoxes);
      primitiveADSResources.host()->push_back(primitiveADS->pointerVertexArray);
      primitiveADSResources.host()->push_back(primitiveADS->pointerAttributeArray);
      primitiveADSResources.host()->push_back(primitiveADS->pointerVertexAttributeArray);
      primitiveADSResources.host()->push_back(primitiveADS->pointerSystemSettings);

      for (const auto& instance : instances.second)
      {
        (*primitiveInstanceTransforms.host())[instanceIndex] = instance->getTransform();

        PrimitiveInstanceADSLeaf nodeData;
        nodeData.bounds             = primitiveADS->primitiveEntity->getPrimBound();
        nodeData.primitiveADSIndex  = adsIndex;
        nodeData.primitiveInstance  = instance->getMaterialId();
        (*primitiveInstanceNodes.host())[instanceIndex] = nodeData;

        instanceIndex++;
      }
    }

    primitiveInstanceNodes.syncDevice();
    primitiveInstanceTransforms.syncDevice();

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
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * bvhPersistentMultiplier, 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);
    workgroupSize[0] /= bvhPersistentMultiplier;

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
    intersectionKernel.setArg(primitiveInstanceNodes.device(), 10);
    intersectionKernel.setArg(&primitiveCount, 11);
    intersectionKernel.setArg(workgroupCount.device(), 12);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * sharedMemoryStride, (size_t)4), 13);
    registerResources(intersectionKernel);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);
  }
#ifdef DEBUG_PI_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
}
