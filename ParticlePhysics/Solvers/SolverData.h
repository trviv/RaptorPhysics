/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef SOLVER_DATA_H
#define SOLVER_DATA_H

#include <Core.h>
#include "SharedAllocator.h"

//forward declaration
class PhysicsSystem;

/*!
@class Class containing data for all entity solvers.
*/
template<class IndexType, class CoefficientType, class VariableType>
class SolverData
{
  // datatype for host constrain
  typedef vector<IndexType>       SingleConstrain;

  // datatype for host coefficient
  typedef vector<CoefficientType> SingleCoefficient;

  friend class PhysicsSystem;

protected:

  // related to equation solver
  DeviceArray<Constrain>            constrainHeaders;
  DeviceArray<IndexType>            constrainIndices;
  DeviceArray<CoefficientType>      constrainCoefficients;
  DeviceArray<VariableType>         constrainConstants;
  DeviceArray<VariableType>         constrainVariableAux[2];

  // related to particle instances
  DeviceArray<ParticleStruct>       particles;
  DeviceArray<ParticleStruct>       particlesPredicted;
  DeviceArray<ParticleDifferential> particleDifferential;
  DeviceArray<ParticleForce>        particleForce;
  DeviceArray<ParticleRigidData>    particleRigidData;
  DeviceArray<ParticleStruct>       particlesTemp[2];
  DeviceArray<ParticleCollisionData>particleCollisionData;
  DeviceArray<ParticleCouplingData> particleCouplingData;

  DeviceArray<PartitionInfo>        partitions;
  DeviceArray<uint>                 partitionsCount;

  vector<SingleConstrain>           rawConstrainConnections;
  vector<SingleCoefficient>         rawConstrainCoefficients;

  /*!@member Per entity simulation property.*/
  DeviceArray<ParticleSharedData>   entitySharedData;

  /*!@member Per entity shared data info.*/
  DeviceArray<EntityLocation>       entityLocations;

  vector<EntityLocation>            updates;

  template<class BaseType> void expand(IndexType index, vector<BaseType>& list)
  {
    while (list.size() <= index) list.push_back(BaseType());
  }

  void addCoefficient(IndexType index, CoefficientType coefficient);

  void addConstrain(IndexType index, IndexType connection);

  SolverData();

  uint nodes()const;

  uint connectionCount()const;

  void addConnection(IndexType index, IndexType connection, CoefficientType coefficient);

  void setConstant(IndexType index, VariableType value);

public:

  const PartitionInfo lastPartition()const;

  const DeviceArray<ParticleStruct>& getParticles()const {return particles;}

  const DeviceArray<PartitionInfo>& getPartitions()const {return partitions;}

  const DeviceArray<uint>& getPartitionsCount()const {return partitionsCount;}

  const DeviceArray<ParticleSharedData>& getEntitySharedData()const {return entitySharedData;}

  const DeviceArray<ParticleCollisionData>& getParticleCollisionData()const {return particleCollisionData;}
};

#endif
