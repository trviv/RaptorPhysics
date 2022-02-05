#include "PrimitiveInstanceAccelerationDataStruct.h"

//#define DEBUG_PI_ADS
#define RAY_TRAVERSAL_BVH_MAX_LEAFS 5
#define BVH_ADS_PERSISTENT_MULTIPLIER 1
#define RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE RAY_TRAVERSAL_BVH_MAX_LEAFS

PrimitiveInstanceAccelerationDataStruct::PrimitiveInstanceAccelerationDataStruct(bool usePrimitiveInstancing)
  :primitiveChanged(true), primitiveInstanceChanged(true), primitiveInstanceTransformsChanged(true), usePrimitiveInstancing(usePrimitiveInstancing)
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
  primitiveInstanceNodes.create(compute);

  primitiveADSResources.create(compute, NULL, true);
}

void PrimitiveInstanceAccelerationDataStruct::updatePointers()
{
  primitiveADSResources.syncDevicePointerBuffer();
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
}

void PrimitiveInstanceAccelerationDataStruct::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");
  includeFiles.push_back("BoundingVolumeHierarchyADSCreate.shader");
  includeFiles.push_back("AccelerationDataStructTraverse.shader");
  includeFiles.push_back("BoundingVolumeHierarchyADSTraverse.shader");

  vector<string> oldType = {"BVH_ADS_INTERSECT_RAY_BVH_FUNCTION", "ADS_TRAVERSAL_SHADER_PROGRAM"};
  vector<string> newType = {"intersectRaysBVHPrimitiveInstances", "PrimitiveInstanceADSTraverse.shader"};

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
    kernel.registerResource(pointerSystemSettings);
  }
}

void PrimitiveInstanceAccelerationDataStruct::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                                          DeviceArray<RTSystemSettings>* systemSettings)
{
  if (usePrimitiveInstancing)
  {
    logComputeError("Bind buffers method is not available with PrimitiveAccelerationDataStruct!");
  }
  else
  {
    BoundingVolumeHierarchyADS::bindBuffers(vertexArray, attributeArray, systemSettings);
  }
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS)
{
  needsRebuild = true;
  primitiveChanged = true;
  primitiveInstanceTransformsChanged = true;

  if (!usePrimitiveInstancing && primitiveInstances.count(primitiveEntityADS) != 0) return;

  primitiveEntityADS->validateBuild();
  primitiveInstances[primitiveEntityADS] = vector<const RayTracingEntity*>();
  primitiveADSResources.host()->push_back(treeInternalNodes.device());
  primitiveADSResources.host()->push_back(leafParentNodeIndices.device());
  primitiveADSResources.host()->push_back(nodeParentNodeIndices.device());
  primitiveADSResources.host()->push_back(leafNodeBoundingBoxes.device());
  primitiveADSResources.host()->push_back(treeNodeBoundingBoxes.device());
  primitiveADSResources.host()->push_back(pointerVertexArray);
  primitiveADSResources.host()->push_back(pointerAttributeArray);
  primitiveADSResources.host()->push_back(pointerSystemSettings);
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveInstance(const RayTracingEntity* entityInstance)
{
  needsRebuild = true;
  primitiveInstanceChanged = true;
  primitiveInstanceTransformsChanged = true;

  if (!usePrimitiveInstancing) return;

  const auto instanceIdentity = getRayTracingEntityId(entityInstance->getIdentity());
  for (auto& prim : primitiveInstances)
  {
    if (getRayTracingEntityId(prim.first->primitiveEntity->getIdentity()) == instanceIdentity)
    {
      prim.second.push_back(entityInstance);
      primitiveCount++;
    }
  }
}

void PrimitiveInstanceAccelerationDataStruct::fullBuild()
{
  if (!needsRebuild) return;

  if (!usePrimitiveInstancing)
  {
    BoundingVolumeHierarchyADS::fullBuild();

    primitiveADSResources.host()->resize(8);

    primitiveADSResources.host()->at(0) = treeInternalNodes.device();
    primitiveADSResources.host()->at(1) = leafParentNodeIndices.device();
    primitiveADSResources.host()->at(2) = nodeParentNodeIndices.device();
    primitiveADSResources.host()->at(3) = leafNodeBoundingBoxes.device();
    primitiveADSResources.host()->at(4) = treeNodeBoundingBoxes.device();
    primitiveADSResources.host()->at(5) = pointerVertexArray;
    primitiveADSResources.host()->at(6) = pointerAttributeArray;
    primitiveADSResources.host()->at(7) = pointerSystemSettings;

    updatePointers();

    primitiveChanged = false;
    primitiveInstanceChanged = false;
    primitiveInstanceTransformsChanged = false;

    return;
  }

  if (needsRebuild)
  {
    primitiveInstanceNodes.resize(primitiveCount, false);
    primitiveInstanceTransforms.resize(primitiveCount * 3 * 2, false);
  }

  if (primitiveInstanceChanged)
  {
    uint index = 0;
    for (const auto& instances : primitiveInstances)
    {
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
    primitiveInstanceNodes.syncHost();
    compute->sync();
#endif
  }

  if (primitiveInstanceTransformsChanged)
  {
    updatePointers();
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
    intersectionKernel.setArg(primitiveADSResources.device(), 3);
    intersectionKernel.setArg(&primitiveCount, 4);
    intersectionKernel.setArg(workgroupCount.device(), 5);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE, (size_t)4), 6);
    registerResources(intersectionKernel);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);
  }
#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
}
