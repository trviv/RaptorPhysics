#ifndef ACCELERATION_DATA_STRUCT_H
#define ACCELERATION_DATA_STRUCT_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>

enum IntersectionType
{
  IntersectionTypeClosest,
  IntersectionTypeAny,
  IntersectionTypeMax
};

extern string getIntersectionTypeName(IntersectionType type);

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
  ComputeKernel intersectRayKernels[IntersectionTypeMax][RayStructTypeMax][HitStructTypeMax];

  vector<EntityPrimAttributes>  registeredPrimitives[RTPrimitiveCount];

  DeviceArray<RTSystemSettings> systemSettings;

  /*!@member Per primitive bounding box array.*/
  DeviceArray<XAB>              boundingBoxes;

  /*!@member Composite array containing all positions.*/
  DeviceArray<PrimitiveStruct>  vertexArray;

  DeviceArray<PrimitiveAttrib>  attributeArray;

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

  /*!@function Register primitive from entity attributes.*/
  virtual void registerPrimitive(RayTracingEntityType type, EntityPrimAttributes primitiveInfo, uint count);

  virtual void commit();

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType,
                             uint rayCount, IntersectionType intersectionType, bool initializeHit);
};

#endif
