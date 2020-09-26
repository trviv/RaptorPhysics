#ifndef ACCELERATION_DATA_STRUCT_H
#define ACCELERATION_DATA_STRUCT_H

#include <Common/RayTracingStruct.h>

/*!
@class Base class for all acceleration structures.
*/
class AccelerationDataStruct : public ShaderEntity
{
protected:
  static uint accXABComputeUtilId;

  ComputeInterface* compute;

  ComputeKernel createPrimitiveBoundingBoxes;
  ComputeKernel intersectRayKernels[RayStructTypeMax][HitStructTypeMax];

  enum PrimitiveAttributeType
  {
    PrimitiveAttributePosition,
    PrimitiveAttributeRadius,
    PrimitiveAttributeMax
  };

  struct PrimitiveAttributeInfo
  {
    RTPrimitiveType       type;
    uint                  count;
    const ComputeMemory*  attributeBuffer[PrimitiveAttributeMax];
    PackingInfo           attributeInfo[PrimitiveAttributeMax];

    uint bindToShader(ComputeKernel& kernel, uint startIndex);
  };

  vector<PrimitiveAttributeInfo>  registeredPrimitives;

  DeviceArray<uint> primStartingOffset;

  DeviceArray<XAB>  boundingBoxes;

  uint              primitiveCount;

public:

  AccelerationDataStruct();

  ~AccelerationDataStruct();

  virtual void create(ComputeInterface* compute);

  uint getPrimCount()const;

  /*!
  @param primitiveBuffer Should be a device array similar to or of type PositionStruct_t.
  */
  virtual void registerSpheres(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count);

  virtual void commit();

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType, uint rayCount);
};

#endif
