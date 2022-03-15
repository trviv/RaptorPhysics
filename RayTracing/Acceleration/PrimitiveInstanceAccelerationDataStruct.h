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
  bool usePrimitiveInstancing;

  unordered_map<const PrimitiveAccelerationDataStruct*, vector<const RayTracingEntity*>> primitiveInstances;

  DeviceArray<const ComputeMemory*>     primitiveADSResources;
  DeviceArray<Matrix4>                  primitiveInstanceTransforms;
  DeviceArray<PrimitiveInstanceADSLeaf>&primitiveInstanceNodes;
  DeviceArray<const ComputeMemory*>     primitiveInstanceADSResources;

  void validateBuild()const;

  void initializeData();

  void updatePointers();

  void createLeafBoundingBoxes();

  void assignLeafMortonCode();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  void registerResources(ComputeKernel& kernel);

public:

  PrimitiveInstanceAccelerationDataStruct(CreationMethod treeCreationMethod, bool usePrimitiveInstancing = true);

  ~PrimitiveInstanceAccelerationDataStruct();

  void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                   const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings);

  void registerPrimitiveADS(const PrimitiveAccelerationDataStruct* primitiveEntityADS);

  void registerPrimitiveInstance(const RayTracingEntity* entityInstance);

  void fullBuild();

  void setInstanceTransformsChanged();

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     uint rayCount, IntersectionType intersectionType);

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     const ComputeMemory* rayCount, IntersectionType intersectionType);

  void encodePrimitiveADS(ComputeKernel& kernel, const uint index);

  void appendTraversalSettings(vector<string>& oldType, vector<string>& newType)const;
};

#endif
