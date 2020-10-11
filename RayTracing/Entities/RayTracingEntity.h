#ifndef RAY_TRACING_ENTITY_H
#define RAY_TRACING_ENTITY_H

#include <Common/RayTracingStruct.h>

enum RayTracingHeapEnum
{
  COMPUTE_HEAP_RAYS
};

class RayTracingAllocator
{
  ComputeHeap rayHeap;

public:

  RayTracingAllocator(ComputeInterface* compute);

  void create(uint initialRays);

  ComputeHeap* getHeap(RayTracingHeapEnum type);
};

enum RayTracingEntityType
{
  RayTracingEntityCamera      = 0x1,
  RayTracingEntityLight       = 0x2,
  RayTracingEntityLightPoint  = 0x2
};

/*!
@class Base class for all ray traced entities.
*/
class RayTracingEntity : protected ShaderEntity
{
  friend class ReaderScene;

protected:

  ComputeInterface*     compute;
  Matrix4               transform; // the transformations related to entity

public:

  RayTracingEntity(ComputeInterface* compute = NULL);

  virtual ~RayTracingEntity(){}

  /*!@function Create a Copy of the current object.*/
  virtual RayTracingEntity* createCopy()const = 0;

  virtual RayTracingEntityId getIdentity()const = 0;

  // Should be called while initializing, after transformations are done.
  virtual void update() = 0;

  Matrix4& getTransform();
};

#endif
