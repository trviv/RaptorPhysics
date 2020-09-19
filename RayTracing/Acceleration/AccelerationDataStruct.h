#ifndef ACCELERATION_DATA_STRUCT_H
#define ACCELERATION_DATA_STRUCT_H

#include <Common/RayTracingStruct.h>

/*!
@class Base class for all acceleration structures.
*/
class AccelerationDataStruct : public ShaderEntity
{
protected:

  string getPrimitiveModeName()const;

  string createIntersectionKey(RayStructType rayType, HitStructType hitType)const;

  static uint accXABComputeUtilId;

  ComputeInterface* compute;

  ComputeKernel createPrimitiveBoundingBoxes;
  unordered_map<string, ComputeKernel> intersectRayKernels;

  enum PrimitiveType
  {
    PrimitiveSphere
  };

  enum PrimitiveAttributeType
  {
    PrimitiveAttributePosition,
    PrimitiveAttributeRadius,
    PrimitiveAttributeMax
  };

  struct PrimitiveAttributeInfo
  {
    PrimitiveType         type;
    uint                  count;
    const ComputeMemory*  attributeBuffer[PrimitiveAttributeMax];
    PackingInfo           attributeInfo[PrimitiveAttributeMax];

    uint bindToShader(ComputeKernel kernel, uint startIndex);
  };

  vector<PrimitiveAttributeInfo>  registeredPrimitives;

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

  virtual void fullUpdate();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType);
};

#endif
