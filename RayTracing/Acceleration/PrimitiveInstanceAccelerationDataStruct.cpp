#include "PrimitiveInstanceAccelerationDataStruct.h"

//#define DEBUG_PI_ADS

PrimitiveInstanceAccelerationDataStruct::PrimitiveInstanceAccelerationDataStruct()
  :primitiveChanged(true), primitiveInstanceChanged(true), primitiveInstanceTransformsChanged(true)
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

  primitiveADSLeafParentNodeIndices.create(compute, NULL, true);
  primitiveADSNodeParentNodeIndices.create(compute, NULL, true);
  primitiveADSLeafNodeBoundingBoxes.create(compute, NULL, true);
  primitiveADSTreeNodeBoundingBoxes.create(compute, NULL, true);
  primitiveADSTreeInternalNodes.create(compute, NULL, true);
  primitiveADSSystemSettings.create(compute, NULL, true);
  primitiveADSVertexArray.create(compute, NULL, true);
  primitiveADSAttributeArray.create(compute, NULL, true);
}

void PrimitiveInstanceAccelerationDataStruct::registerCreateShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  BoundingVolumeHierarchyADS::registerCreateShaders();

  includeFiles.clear();

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "PrimitiveInstanceAccelerationDataStruct.shader", NULL, NULL);

  updatePrimitiveInstanceData = programs.back().createKernel("updatePrimitiveInstanceData");
}

void PrimitiveInstanceAccelerationDataStruct::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  BoundingVolumeHierarchyADS::registerTraverseShaders(oldTypeArg, newTypeArg);
}

void PrimitiveInstanceAccelerationDataStruct::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                                          DeviceArray<RTSystemSettings>* systemSettings)
{
  logComputeError("Bind buffers method is not available with PrimitiveAccelerationDataStruct!");
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS)
{
  if (primitiveInstances.count(primitiveEntityADS) != 0) return;

  primitiveEntityADS->validateBuild();
  primitiveInstances[primitiveEntityADS] = vector<const RayTracingEntity*>();
  primitiveADSLeafParentNodeIndices.host()->push_back(primitiveEntityADS->pointerLeafParentNodeIndices);
  primitiveADSNodeParentNodeIndices.host()->push_back(primitiveEntityADS->pointerNodeParentNodeIndices);
  primitiveADSLeafNodeBoundingBoxes.host()->push_back(primitiveEntityADS->pointerLeafNodeBoundingBoxes);
  primitiveADSTreeNodeBoundingBoxes.host()->push_back(primitiveEntityADS->pointerTreeNodeBoundingBoxes);
  primitiveADSTreeInternalNodes.host()->push_back(primitiveEntityADS->pointerTreeInternalNodes);
  primitiveADSSystemSettings.host()->push_back(primitiveEntityADS->pointerSystemSettings);
  primitiveADSVertexArray.host()->push_back(primitiveEntityADS->pointerVertexArray);
  primitiveADSAttributeArray.host()->push_back(primitiveEntityADS->pointerAttributeArray);

  needsRebuild = true;
  primitiveChanged = true;
  primitiveInstanceTransformsChanged = true;
}

void PrimitiveInstanceAccelerationDataStruct::registerPrimitiveInstance(const RayTracingEntity* entityInstance)
{
  const auto instanceIdentity = getRayTracingEntityId(entityInstance->getIdentity());
  for (auto& prim : primitiveInstances)
  {
    if (getRayTracingEntityId(prim.first->primitiveEntity->getIdentity()) == instanceIdentity)
    {
      prim.second.push_back(entityInstance);
      primitiveCount++;
    }
  }
  needsRebuild = true;
  primitiveInstanceChanged = true;
  primitiveInstanceTransformsChanged = true;
}

void PrimitiveInstanceAccelerationDataStruct::fullBuild()
{
  updatePointers();

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
    primitiveADSLeafParentNodeIndices.syncDevicePointerBuffer();
    primitiveADSNodeParentNodeIndices.syncDevicePointerBuffer();
    primitiveADSLeafNodeBoundingBoxes.syncDevicePointerBuffer();
    primitiveADSTreeNodeBoundingBoxes.syncDevicePointerBuffer();
    primitiveADSTreeInternalNodes.syncDevicePointerBuffer();
    primitiveADSSystemSettings.syncDevicePointerBuffer();
    primitiveADSVertexArray.syncDevicePointerBuffer();
    primitiveADSAttributeArray.syncDevicePointerBuffer();

    pointerLeafParentNodeIndices = primitiveADSLeafParentNodeIndices.device();
    pointerNodeParentNodeIndices = primitiveADSNodeParentNodeIndices.device();
    pointerLeafNodeBoundingBoxes = primitiveADSLeafNodeBoundingBoxes.device();
    pointerTreeNodeBoundingBoxes = primitiveADSTreeNodeBoundingBoxes.device();
    pointerTreeInternalNodes     = primitiveADSTreeInternalNodes.device();
    pointerSystemSettings        = primitiveADSSystemSettings.device();

    pointerVertexArray    = primitiveADSVertexArray.device();
    pointerAttributeArray = primitiveADSAttributeArray.device();
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
  BoundingVolumeHierarchyADS::intersectRays(hits, hitType, rays, rayType, rayCount, intersectionType);
}
