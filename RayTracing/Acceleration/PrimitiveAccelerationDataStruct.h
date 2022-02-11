#ifndef PRIMITIVE_ACCELERATION_DATA_STRUCTURE_H
#define PRIMITIVE_ACCELERATION_DATA_STRUCTURE_H

#include <Entities/RayTracingEntity.h>
#include "BoundingVolumeHierarchyADS.h"

/*!
@class Class to create acceleration data structure for a primitive.
*/
class PrimitiveAccelerationDataStruct : protected BoundingVolumeHierarchyADS
{
  friend class PrimitiveInstanceAccelerationDataStruct;

protected:

  const RayTracingEntity* primitiveEntity;

  DeviceArray<RTSystemSettings> primitiveInformation;

  void initializeData();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  /*!@function Create Acceleration Data Structure traverse shaders.*/
  void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

public:

  PrimitiveAccelerationDataStruct();

  ~PrimitiveAccelerationDataStruct();

  void create(ComputeInterface* compute);

  void bindEntity(const RayTracingEntity* primitiveEntity);

  void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                   const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings);

  void fullBuild();
};

#endif
