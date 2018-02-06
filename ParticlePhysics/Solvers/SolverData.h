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

  // related to particle simulation
  DeviceArray<ParticleSharedData>   particleSharedData;
  DeviceArray<ParticleStruct>       particles;
  DeviceArray<ParticleStruct>       particleDeltas;
  DeviceArray<ParticleDifferential> particleDiff;
  DeviceArray<ParticleAuxData>      particleAuxData;
  DeviceArray<ParticleStruct>       particlesTemp[2];

  vector<SingleConstrain>           rawConstrainConnections;
  vector<SingleCoefficient>         rawConstrainCoefficients;

  // sections in the device array where each entity is present
  vector<SectionData>               deviceSections;

  uint nodeOffset;
  uint connectionOffset;

  template<class BaseType> void expand(IndexType index, vector<BaseType>& list)
  {
    while (list.size() <= index) list.push_back(BaseType());
  }

  void addCoefficient(IndexType index, CoefficientType coefficient);

  void addConstrain(IndexType index, IndexType connection);

public:

  SolverData();

  uint nodes()const
  {
    return nodeOffset;
  }

  void addConnection(IndexType index, IndexType connection, CoefficientType coefficient);

  void setConstant(IndexType index, VariableType value);

};

#endif