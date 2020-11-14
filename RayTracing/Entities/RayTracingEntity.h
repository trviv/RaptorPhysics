#ifndef RAY_TRACING_ENTITY_H
#define RAY_TRACING_ENTITY_H

#include <Common/RayTracingStruct.h>
#include <Common/MaterialStruct.h>

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


enum EntityPrimitiveAttributeType
{
  EntityPrimitiveAttributePosition,
  EntityPrimitiveAttributeRadius,
  EntityPrimitiveAttributeIndex = EntityPrimitiveAttributeRadius,
  EntityPrimitiveAttributeMax
};

struct EntityPrimAttributes
{
  DecodedPrimitiveInfo  primInfo;
  const ComputeMemory*  attributeBuffer[EntityPrimitiveAttributeMax];
  PackingInfo           attributeInfo[EntityPrimitiveAttributeMax];

  EntityPrimAttributes();

  uint bindToShader(ComputeKernel& kernel, uint startIndex);

  void setAttribute(EntityPrimitiveAttributeType type, const ComputeMemory* attributeBuffer, PackingInfo attributePacking);
};


enum RayTracingEntityType
{
  RayTracingEntityCamera      = 0,

  RayTracingEntityLight       = 1,
  RayTracingEntityLightPoint  = 1,

  RayTracingEntityPrimArray   = 2,
  RayTracingEntitySpheres     = 2,
  RayTracingEntityTriangles   = 3
};

#ifndef COMPUTE_SHADER_SCOPE

static RayTracingEntityType getRayTracingEntityCategory(ushort type)
{
  if (type < RayTracingEntityLight)
    return RayTracingEntityCamera;
  if (type < RayTracingEntityPrimArray)
    return RayTracingEntityLight;
  return RayTracingEntityPrimArray;
}

#endif

/*!
@class Base class for all ray traced entities.
*/
class RayTracingEntity : public EntityPrimAttributes
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

  virtual void setMaterialId(MaterialId materialId){}

  virtual RayTracingEntityId getIdentity()const = 0;

  // Should be called while initializing, after transformations are done.
  virtual void update() = 0;

  Matrix4& getTransform();
};


/*!
@class Class for objects made of an array of primitives of the same type.
*/
class PrimitiveArrayEntity : public RayTracingEntity
{
  friend class ReaderScene;

  MaterialId          material;
  RayTracingEntityId  identity;
  uint                primitiveCount;

public:

  PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount);

  ~PrimitiveArrayEntity();

  RayTracingEntity* createCopy()const;

  void setMaterialId(MaterialId materialId);

  RayTracingEntityId getIdentity()const;

  uint getPrimCount()const;

  void update();
};

#endif
