#ifndef SHARED_ALLOCATOR_H
#define SHARED_ALLOCATOR_H

#include <Core.h>
#include "Constrain.h"
#include "../Common/ParticleStruct.h"

template<class IndexType, class CoefficientType> struct ConstrainAllocator
{
  ComputeHeap constrainSectionsHeap;
  ComputeHeap constrainHeadersHeap;
  ComputeHeap constrainIndicesHeap;
  ComputeHeap constrainCoefficientsHeap;

  ConstrainAllocator(ComputeInterface* compute)
    :constrainSectionsHeap(compute),
    constrainHeadersHeap(compute),
    constrainIndicesHeap(compute),
    constrainCoefficientsHeap(compute)
  {}

  void create(uint initialNodes, uint initialIndices)
  {
    constrainSectionsHeap.create(initialNodes * sizeof(Constrain));
    constrainHeadersHeap.create(initialNodes * sizeof(Constrain));
    constrainIndicesHeap.create(initialIndices * sizeof(IndexType));
    constrainCoefficientsHeap.create(initialIndices * sizeof(CoefficientType));
  }
};

struct ParticleAllocator
{
  ComputeHeap partitions;
  ComputeHeap particlePredicted;
  ComputeHeap particleSharedHeap;
  ComputeHeap particleHeap;
  ComputeHeap particleDifferentialHeap;

  ComputeHeap particleRigidData;
  ComputeHeap particleCollisionData;
  ComputeHeap particleAuxData;

  ParticleAllocator(ComputeInterface* compute)
    :partitions(compute),
    particlePredicted(compute),
    particleSharedHeap(compute),
    particleHeap(compute),
    particleDifferentialHeap(compute),
    particleRigidData(compute),
    particleCollisionData(compute),
    particleAuxData(compute)
  {}

  void create(uint initialParticles)
  {
    partitions.create(initialParticles/256 * sizeof(ParticleSharedData));
    particlePredicted.create(initialParticles * sizeof(ParticleStruct));
    particleSharedHeap.create(initialParticles/256 * sizeof(ParticleSharedData));
    particleHeap.create(initialParticles * sizeof(ParticleStruct));
    particleDifferentialHeap.create(initialParticles * sizeof(ParticleDifferential));
    particleRigidData.create(initialParticles/64 * sizeof(ParticleRigidData));
    particleCollisionData.create(initialParticles * sizeof(ParticleCollisionData));
    particleAuxData.create(initialParticles * sizeof(ParticleAuxData));
  }
};

enum SharedComputeHeapEnum
{
  COMPUTE_HEAP_SECTIONS,

  COMPUTE_HEAP_CONSTRAIN_HEADERS,
  COMPUTE_HEAP_CONSTRAIN_INDICES,
  COMPUTE_HEAP_CONSTRAIN_COEFFICIENTS,

  COMPUTE_HEAP_PARTITIONS,
  COMPUTE_HEAP_PARTICLE,
  COMPUTE_HEAP_PARTICLE_PREDICTED,
  COMPUTE_HEAP_PARTICLE_SHARED,
  COMPUTE_HEAP_PARTICLE_DIFF,
  COMPUTE_HEAP_PARTICLE_RIGID,
  COMPUTE_HEAP_PARTICLE_COLLISION,
  COMPUTE_HEAP_PARTICLE_AUX
};

class SharedAllocator
{
public:
  ConstrainAllocator<uint, real>  constrainAllocator;

  ParticleAllocator               particleAllocator;

  SharedAllocator(ComputeInterface* compute)
    :constrainAllocator(compute),
    particleAllocator(compute)
  {}

  ComputeHeap* getHeap(SharedComputeHeapEnum type)
  {
    switch (type)
    {
    case COMPUTE_HEAP_SECTIONS:
      return &constrainAllocator.constrainSectionsHeap;
    case COMPUTE_HEAP_CONSTRAIN_HEADERS:
      return &constrainAllocator.constrainHeadersHeap;
    case COMPUTE_HEAP_CONSTRAIN_INDICES:
      return &constrainAllocator.constrainIndicesHeap;
    case COMPUTE_HEAP_CONSTRAIN_COEFFICIENTS:
      return &constrainAllocator.constrainCoefficientsHeap;

    case COMPUTE_HEAP_PARTITIONS:
      return &particleAllocator.partitions;
    case COMPUTE_HEAP_PARTICLE:
      return &particleAllocator.particleHeap;
    case COMPUTE_HEAP_PARTICLE_PREDICTED:
      return &particleAllocator.particlePredicted;
    case COMPUTE_HEAP_PARTICLE_SHARED:
      return &particleAllocator.particleSharedHeap;
    case COMPUTE_HEAP_PARTICLE_DIFF:
      return &particleAllocator.particleDifferentialHeap;
    case COMPUTE_HEAP_PARTICLE_RIGID:
      return &particleAllocator.particleRigidData;
    case COMPUTE_HEAP_PARTICLE_COLLISION:
      return &particleAllocator.particleCollisionData;
    case COMPUTE_HEAP_PARTICLE_AUX:
      return &particleAllocator.particleAuxData;
    }
    return NULL;
  }
};

#endif
