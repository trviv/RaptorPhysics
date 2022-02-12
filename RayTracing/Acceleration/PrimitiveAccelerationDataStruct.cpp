#include "PrimitiveAccelerationDataStruct.h"

//#define DEBUG_BVH_ADS

PrimitiveAccelerationDataStruct::PrimitiveAccelerationDataStruct()
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
  this->primitiveCount  = primitiveEntity->getPrimitiveCount();
  this->vertexCount     = primitiveEntity->getPrimitiveVertexCount();

  primitiveInformation.host()->resize(1);

  uint primOffset   = 0;
  uint vertexOffset = 0;

  RTPrimitiveType primType = primitiveEntity->getPrimitiveType();

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    if (primType == i)
    {
      primOffset    += primitiveEntity->getPrimitiveCount();
      vertexOffset  += primitiveEntity->getPrimitiveVertexCount();
    }

    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  primOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);

    primitiveInformation.host()->at(0).globalOffsets[i] = primInfo;
  }

  primitiveInformation.syncDevice();

  BoundingVolumeHierarchyADS::bindBuffers((*primitiveEntity)[EntityPrimitiveAttributePosition], (*primitiveEntity)[EntityPrimitiveAttributeIndex],
                                          (*primitiveEntity)[EntityPrimitiveAttributeNormal], &primitiveInformation);

  vector<string> oldType = {"RAY_TRACING_SINGLE_PRIMITIVE_ADS"};
  vector<string> newType = {to_string(primitiveEntity->getEntityType())};

  registerCreateShaders(&oldType, &newType);
  registerTraverseShaders();
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
