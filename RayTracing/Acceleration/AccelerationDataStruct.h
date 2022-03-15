#ifndef ACCELERATION_DATA_STRUCT_H
#define ACCELERATION_DATA_STRUCT_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>
#include <Common/HitStructs.h>
#include <Common/MaterialStruct.h>

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
class AccelerationDataStruct : public ShaderEntity, protected PrimitiveADSResources
{
protected:
  static uint accXABComputeUtilId;
  static uint sortComputeUtilId;

  ComputeInterface* compute;

  ComputeKernel createPrimitiveBoundingBoxes;
  ComputeKernel intersectRayKernels[IntersectionTypeMax][RayStructTypeMax][HitStructTypeMax];

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

  void updatePointers();

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
                           const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings);

  virtual void fullBuild();

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             uint rayCount, IntersectionType intersectionType);

  virtual void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                             const ComputeMemory* rayCount, IntersectionType intersectionType);

  virtual void appendTraversalSettings(vector<string>& oldType, vector<string>& newType)const;
};

#endif
