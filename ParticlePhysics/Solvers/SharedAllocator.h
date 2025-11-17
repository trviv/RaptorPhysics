/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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
  ComputeHeap particleForceHeap;

  ComputeHeap particleRigidData;
  ComputeHeap particleCollisionData;
  ComputeHeap particleCouplingData;

  ParticleAllocator(ComputeInterface* compute)
    :partitions(compute),
    particlePredicted(compute),
    particleSharedHeap(compute),
    particleHeap(compute),
    particleDifferentialHeap(compute),
    particleForceHeap(compute),
    particleRigidData(compute),
    particleCollisionData(compute),
    particleCouplingData(compute)
  {}

  void create(uint initialParticles)
  {
    partitions.create(initialParticles/256 * sizeof(ParticleSharedData));
    particlePredicted.create(initialParticles * sizeof(ParticleStruct));
    particleSharedHeap.create(initialParticles/256 * sizeof(ParticleSharedData));
    particleHeap.create(initialParticles * sizeof(ParticleStruct));
    particleDifferentialHeap.create(initialParticles * sizeof(ParticleDifferential));
    particleForceHeap.create(initialParticles * sizeof(ParticleForce));
    particleRigidData.create(initialParticles/64 * sizeof(ParticleRigidData));
    particleCollisionData.create(initialParticles * sizeof(ParticleCollisionData));
    particleCouplingData.create(initialParticles * sizeof(ParticleCouplingData));
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
  COMPUTE_HEAP_PARTICLE_FORCE,
  COMPUTE_HEAP_PARTICLE_RIGID,
  COMPUTE_HEAP_PARTICLE_COLLISION,
  COMPUTE_HEAP_PARTICLE_COUPLING
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
    case COMPUTE_HEAP_PARTICLE_FORCE:
      return &particleAllocator.particleForceHeap;
    case COMPUTE_HEAP_PARTICLE_RIGID:
      return &particleAllocator.particleRigidData;
    case COMPUTE_HEAP_PARTICLE_COLLISION:
      return &particleAllocator.particleCollisionData;
    case COMPUTE_HEAP_PARTICLE_COUPLING:
      return &particleAllocator.particleCouplingData;
    }
    return NULL;
  }
};

#endif
