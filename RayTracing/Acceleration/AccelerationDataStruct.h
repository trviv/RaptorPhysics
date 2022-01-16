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
  const ComputeMemory*  pointerVertexArray;
  /*!@member Composite array containing all attributes.*/
  const ComputeMemory*  pointerAttributeArray;
  /*!@member Pointer to ray tracing system settings .*/
  ComputeMemory*        pointerSystemSettings;

  const ComputeMemory*  pointerLeafNodeBoundingBoxes;

  /*!@member Per primitive bounding box array.*/
  DeviceArray<XAB>  leafNodeBoundingBoxes;

  /*!@member Workgroup count buffer.*/
  DeviceArray<uint> workgroupCount;

  /*!@member Total primitives in the system.*/
  uint  primitiveCount;

  /*!@member Total positions/vertex in the system.*/
  uint  vertexCount;

  /*!@member Indicate's if ADS need to be build before being used.*/
  bool needsRebuild;

  virtual void validateBuild()const;

  virtual void updatePointers();

  virtual void initializeData();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  virtual void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  /*!@function Create Acceleration Data Structure traverse shaders.*/
  virtual void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

public:

  AccelerationDataStruct();

  ~AccelerationDataStruct();

  virtual void create(ComputeInterface* compute);

  uint getPrimCount()const;

  virtual void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                           DeviceArray<RTSystemSettings>* systemSettings);

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             uint rayCount, IntersectionType intersectionType);

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             const ComputeMemory* rayCount, IntersectionType intersectionType);
};

#endif
