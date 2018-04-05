#ifndef SOLVER_DATA_H
#define SOLVER_DATA_H

#include <Core.h>
#include "SharedAllocator.h"

//forward declaration
class PhysicsSystem;

template<class IndexType, class CoefficientType, class VariableType> class SolverData
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
  DeviceArray<IdentityInfo>         particleIdentities;
  DeviceArray<ParticleStruct>       particleDeltas;
  DeviceArray<ParticleDifferential> particleDifferential;
  DeviceArray<ParticleAuxData>      particleAuxData;
  DeviceArray<ParticleRigidData>    particleRigidData;
  DeviceArray<ParticleStruct>       particlesTemp[2];

  DeviceArray<PartitionInfo>        partitions;
  DeviceArray<uint>                 partitionsCount;

  vector<SingleConstrain>           rawConstrainConnections;
  vector<SingleCoefficient>         rawConstrainCoefficients;

  /*@member Per entity simulation property.*/
  DeviceArray<ParticleSharedData>   entitySharedData;

  /*@member Per entity shared data info.*/
  DeviceArray<EntityLocation>       entityLocations;

  vector<EntityLocation>            updates;

  template<class BaseType> void expand(IndexType index, vector<BaseType>& list)
  {
    while (list.size() <= index) list.push_back(BaseType());
  }

  void addCoefficient(IndexType index, CoefficientType coefficient);

  void addConstrain(IndexType index, IndexType connection);

public:

  SolverData();

  uint nodes()const;

  uint connectionCount()const;

  const PartitionInfo lastPartition()const;

  void addConnection(IndexType index, IndexType connection, CoefficientType coefficient);

  void setConstant(IndexType index, VariableType value);

};

#endif