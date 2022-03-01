#include "PrimitiveAccelerationDataStruct.h"

//#define DEBUG_BVH_ADS

PrimitiveAccelerationDataStruct::PrimitiveAccelerationDataStruct(CreationMethod treeCreationMethod)
  :BoundingVolumeHierarchyADS(treeCreationMethod)
{
  primitiveEntity = NULL;
}

PrimitiveAccelerationDataStruct::~PrimitiveAccelerationDataStruct()
{
}

void PrimitiveAccelerationDataStruct::initializeData()
{
  BoundingVolumeHierarchyADS::initializeData();
  primitiveInformation.create(compute);
}

void PrimitiveAccelerationDataStruct::registerCreateShaders(const vector<string>* oldType, const vector<string>* newType)
{
  BoundingVolumeHierarchyADS::registerCreateShaders(oldType, newType);
}

void PrimitiveAccelerationDataStruct::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  BoundingVolumeHierarchyADS::registerTraverseShaders(oldTypeArg, newTypeArg);
}

void PrimitiveAccelerationDataStruct::create(ComputeInterface* compute)
{
  this->compute = compute;
  initializeData();
}

void PrimitiveAccelerationDataStruct::bindEntity(const RayTracingEntity* primitiveEntity)
{
  this->primitiveEntity = primitiveEntity;
  vector<string> oldType = {};//{"RAY_TRACING_SINGLE_PRIMITIVE_ADS"};
  vector<string> newType = {};//{to_string(primitiveEntity->getPrimitiveType())};

  registerCreateShaders(&oldType, &newType);
  registerTraverseShaders();

  primitiveInstancesPerType[primitiveEntity->getPrimitiveType()].push_back(primitiveEntity);

  resizePrimitiveArray();
  BoundingVolumeHierarchyADS::bindBuffers(vertexArray.device(), attributeArray.device(), vertexAttributeArray.device(), &systemSettings);
  composePrimitiveArray();
}

void PrimitiveAccelerationDataStruct::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                                  const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings)
{
  logComputeError("Bind buffers method is not available with PrimitiveAccelerationDataStruct!");
}

void PrimitiveAccelerationDataStruct::fullBuild()
{
  BoundingVolumeHierarchyADS::fullBuild();

#ifdef DEBUG_BVH_ADS
  primitiveInformation.syncHost();
  compute->sync();
#endif
}
