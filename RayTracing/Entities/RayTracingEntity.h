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


class EntityPrimAttributes
{
  friend class RayTracingSystem;

protected:
  DecodedPrimitiveInfo  primInfo;
  const ComputeMemory*  attributeBuffer[EntityPrimitiveAttributeMax];
  PackingInfo           attributeInfo[EntityPrimitiveAttributeMax];

public:
  EntityPrimAttributes();

  uint bindToShader(ComputeKernel& kernel, uint startIndex);

  void setAttribute(EntityPrimitiveAttributeType type, const ComputeMemory* attributeBuffer, PackingInfo attributePacking);
};


static RayTracingEntityType getRayTracingEntityCategory(ushort type)
{
  if (type < RayTracingEntityLight)
    return RayTracingEntityCamera;
  if (type < RayTracingEntityPrimArray)
    return RayTracingEntityLight;
  return RayTracingEntityPrimArray;
}


/*!
@class Base class for all ray traced entities.
*/
class RayTracingEntity : public EntityPrimAttributes
{
  friend class ReaderScene;

protected:

  ComputeInterface*   compute;
  Matrix4             transform; // the transformations related to entity
  DeviceArray<uint>*  deviceData;
  MaterialId          materialId;

public:

  RayTracingEntity(ComputeInterface* compute = NULL);

  virtual ~RayTracingEntity(){}

  /*!@function Create a Copy of the current object.*/
  virtual RayTracingEntity* createCopy()const = 0;

  void setMaterialId(MaterialId materialId);

  virtual RayTracingEntityId getIdentity()const = 0;

  // Should be called while initializing, after transformations are done.
  virtual void update() = 0;

  Matrix4& getTransform();

  RayTracingEntityType getEntityCategory()const;

  MaterialId& getMaterialId();
};


/*!
@class Class for objects made of an array of primitives of the same type.
*/
class PrimitiveArrayEntity : public RayTracingEntity
{
  friend class ReaderScene;

  RayTracingEntityId  identity;
  XAB                 primBound;

  void generateNeighbourBasedNormal(uint vertexCount, vector<Real3>* normals = NULL);

  void changeEntityType(RayTracingEntityType type);

public:

  PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount, ComputeInterface* compute = NULL);

  ~PrimitiveArrayEntity();

  RayTracingEntity* createCopy()const;

  void createBox(const real dim[]);

  void createSphere(const real radius);

  void createMesh(const string fileName);

  RayTracingEntityId getIdentity()const;

  const XAB& getPrimBound()const;

  void update();
};

#endif
