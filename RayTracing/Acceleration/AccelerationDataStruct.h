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

  ComputeKernel createPrimitiveBoundingBoxes;
  ComputeKernel intersectRayKernels[IntersectionTypeMax][RayStructTypeMax][HitStructTypeMax];

  /*!@member Composite array containing all positions.*/
  const DeviceArray<PrimitiveStruct>* vertexArray;

  /*!@member Composite array containing all attributes.*/
  const DeviceArray<PrimitiveAttrib>* attributeArray;

  DeviceArray<RTSystemSettings>*      systemSettings;

  /*!@member Per primitive bounding box array.*/
  DeviceArray<XAB>          boundingBoxes;

  /*!@member Total primitives in the system.*/
  uint  primitiveCount;

  /*!@member Total positions/vertex in the system.*/
  uint  vertexCount;

public:

  AccelerationDataStruct();

  ~AccelerationDataStruct();

  virtual void create(ComputeInterface* compute);

  uint getPrimCount()const;

  virtual void commit(const DeviceArray<PrimitiveStruct>* vertexArray, const DeviceArray<PrimitiveAttrib>* attributeArray,
                      DeviceArray<RTSystemSettings>* systemSettings);

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             uint rayCount, IntersectionType intersectionType);

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             const ComputeMemory* rayCount, IntersectionType intersectionType);
};

#endif
