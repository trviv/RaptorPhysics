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
  RayTracingEntityLight = 0x1,
  RayTracingEntityCamera = 0x2,
};

/*!
@class Base class for all ray traced entities.
*/
class RayTracingEntity : protected ShaderEntity
{
  friend class ReaderScene;

protected:

  ComputeInterface*     compute;
  Matrix                affine; // the transformations related to entity
  RayTracingEntityType  type;

public:

  RayTracingEntity(RayTracingEntityType type, ComputeInterface* compute = NULL);

  virtual ~RayTracingEntity(){}

  // Should be called while initializing, after transformations are done.
  virtual void update() = 0;

  Matrix& getAffine()
  {
    return affine;
  }
};

#endif
