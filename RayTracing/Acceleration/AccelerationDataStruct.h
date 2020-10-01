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
  static uint sortComputeUtilId;

  ComputeInterface* compute;

  ComputeKernel assignMortonCode;
  ComputeKernel createPrimitiveBoundingBoxes;
  ComputeKernel intersectRayKernels[RayStructTypeMax][HitStructTypeMax];

  enum PrimitiveAttributeType
  {
    PrimitiveAttributePosition,
    PrimitiveAttributeRadius,
    PrimitiveAttributeIndex = PrimitiveAttributeRadius,
    PrimitiveAttributeMax
  };

  struct PrimitiveAttributeInfo
  {
    DecodedPrimitiveInfo  primInfo;
    const ComputeMemory*  attributeBuffer[PrimitiveAttributeMax];
    PackingInfo           attributeInfo[PrimitiveAttributeMax];

    uint bindToShader(ComputeKernel& kernel, uint startIndex);
  };

  vector<PrimitiveAttributeInfo>  registeredPrimitives[RTPrimitiveCount];

  DeviceArray<RTSystemSettings> systemSettings;

  /*!@member Per primitive bounding box array.*/
  DeviceArray<XAB>              boundingBoxes;

  /*!@member Composite array containing all positions.*/
  DeviceArray<PrimitiveStruct>  vertexArray;

  DeviceArray<BVHLeafInfo>      primitiveLeafData;

  DeviceArray<BVHLeafInfo>      primitiveLeafDataSorted;

  /*!@member Total primitives in the system.*/
  uint  primitiveCount;

  /*!@member Total positions/vertex in the system.*/
  uint  vertexCount;

public:

  AccelerationDataStruct();

  ~AccelerationDataStruct();

  virtual void create(ComputeInterface* compute);

  uint getPrimCount()const;

  /*!
  @param primitiveBuffer Should be a device array similar to or of type PositionStruct_t.
  */
  virtual void registerSpheres(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count);

  /*!
  @param primitiveBuffer Should be a device array similar to or of type PositionStruct_t.
  */
  virtual void registerTriangles(const ComputeMemory* primitiveBuffer, const ComputeMemory* indexBuffer, PackingInfo indexInfo, uint count);

  virtual void commit();

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType, uint rayCount);
};

#endif
