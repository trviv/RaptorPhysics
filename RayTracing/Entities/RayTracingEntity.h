/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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

  uint bindToShader(ComputeKernel& kernel, uint startIndex)const;

  void setAttribute(EntityPrimitiveAttributeType type, const ComputeMemory* attributeBuffer, PackingInfo attributePacking);

  const ComputeMemory* operator[](EntityPrimitiveAttributeType attributeType)const;

  RTPrimitiveType getPrimitiveType()const;

  const uint& getPrimitiveCount()const;

  const uint& getVertexCount()const;

  uint getPrimitiveVertexCount()const;
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
  XAB                 primBound;
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

  virtual void setIdentity(RayTracingEntityId identity) = 0;

  // Should be called while initializing, after transformations are done.
  virtual void update() = 0;

  const XAB& getPrimBound()const;

  Matrix4& getTransform();

  const Matrix4& getTransform()const;

  RayTracingEntityType getEntityCategory()const;

  RayTracingEntityType getEntityType()const;

  MaterialId& getMaterialId();

  const MaterialId& getMaterialId()const;
};


/*!
@class Class for objects made of an array of primitives of the same type.
*/
class PrimitiveArrayEntity : public RayTracingEntity
{
  friend class ReaderScene;

  RayTracingEntityId  identity;

  void generateNeighbourBasedNormal(uint vertexCount, vector<Real3>* normals = NULL);

  void changeEntityType(RayTracingEntityType type);

  RTPrimitiveType primTypeFromEntityType();

public:

  PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount, ComputeInterface* compute = NULL);

  ~PrimitiveArrayEntity();

  RayTracingEntity* createCopy()const;

  void createBox(const real dim[]);

  void createSphere(const real radius);

  void createMesh(const string fileName);

  RayTracingEntityId getIdentity()const;

  void setIdentity(RayTracingEntityId identity);

  void update();
};

#endif
