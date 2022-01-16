#ifndef PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCTURE_H
#define PRIMITIVE_INSTANCE_ACCELERATION_DATA_STRUCTURE_H

#include "PrimitiveAccelerationDataStruct.h"

/*!
@class Class to create acceleration data structure for a primitive instances.
*/
class PrimitiveInstanceAccelerationDataStruct : public BoundingVolumeHierarchyADS
{
protected:

  ComputeKernel updatePrimitiveInstanceData;

  bool primitiveChanged;
  bool primitiveInstanceChanged;
  bool primitiveInstanceTransformsChanged;

  unordered_map<const PrimitiveAccelerationDataStruct*, vector<const RayTracingEntity*>> primitiveInstances;

  DeviceArray<const ComputeMemory*>     primitiveADSLeafParentNodeIndices;
  DeviceArray<const ComputeMemory*>     primitiveADSNodeParentNodeIndices;
  DeviceArray<const ComputeMemory*>     primitiveADSLeafNodeBoundingBoxes;
  DeviceArray<const ComputeMemory*>     primitiveADSTreeNodeBoundingBoxes;
  DeviceArray<const ComputeMemory*>     primitiveADSTreeInternalNodes;
  DeviceArray<const ComputeMemory*>     primitiveADSSystemSettings;
  DeviceArray<const ComputeMemory*>     primitiveADSVertexArray;
  DeviceArray<const ComputeMemory*>     primitiveADSAttributeArray;

  DeviceArray<float4>                   primitiveInstanceTransforms;
  DeviceArray<PrimitiveInstanceADSLeaf> primitiveInstanceNodes;

  void validateBuild()const;

  void initializeData();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

public:

  PrimitiveInstanceAccelerationDataStruct();

  ~PrimitiveInstanceAccelerationDataStruct();

  void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                   DeviceArray<RTSystemSettings>* systemSettings);

  void registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS);

  void registerPrimitiveInstance(const RayTracingEntity* entityInstance);

  void fullBuild();

  void setInstanceTransformsChanged();

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     uint rayCount, IntersectionType intersectionType);

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     const ComputeMemory* rayCount, IntersectionType intersectionType);
};

#endif
