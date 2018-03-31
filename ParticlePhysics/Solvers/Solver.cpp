#include "Solver.h"

template<class IndexType, class CoefficientType, class VariableType>
Solver<IndexType, CoefficientType, VariableType>::Solver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type) :
SolverData(), compute(compute), allocator(allocator), type(type)
{
  constrainHeaders.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_HEADERS), true);
  constrainIndices.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_INDICES), true);
  constrainCoefficients.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_COEFFICIENTS), true);
  constrainConstants.create(compute, NULL, true);
  constrainVariableAux[0].create(compute, NULL, false);
  constrainVariableAux[1].create(compute, NULL, false);

  entityParticleSharedData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED), true);

  particles.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE), true);
  particleIdentities.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY), true);
  particleDeltas.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA), false);
  particleDifferential.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF), false);
  particleAuxData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX), true);
  particleRigidData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_RIGID), true);

#ifdef DEBUG_SOLVERS
  particlesTemp[0].create(compute, NULL, true);
  particlesTemp[1].create(compute, NULL, true);
#else
  particlesTemp[0].create(compute, NULL, false);
  particlesTemp[1].create(compute, NULL, false);
#endif

  partitions.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTITIONS), true);
  partitionsCount.create(compute, NULL, true);
  partitionsCount.host()->reserve(1);
  partitionsCount.host()->resize(1);
  deviceSections.create(compute, allocator->getHeap(COMPUTE_HEAP_SECTIONS), true);

  iterations = 1;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ConstrainStruct.h");
  includeFiles.push_back("ParticleStruct.h");
}

template<class IndexType, class CoefficientType, class VariableType>
Solver<IndexType, CoefficientType, VariableType>::~Solver()
{
}

template<class IndexType, class CoefficientType, class VariableType>
void Solver<IndexType, CoefficientType, VariableType>::commit(const SectionData& sectionData)
{
  flatArray<CoefficientType>(*constrainCoefficients.host(), rawConstrainCoefficients);

  SectionData updateInfo;

  updateInfo.node.offset = nodes();
  updateInfo.node.count = constrainConstants.host()->size() - nodes();

  updateInfo.connection.offset = connectionCount();
  updateInfo.connection.count = constrainCoefficients.host()->size() - connectionCount();

  updates.push_back(updateInfo);
  deviceSections.host()->push_back(updateInfo);
}

template<class IndexType, class CoefficientType, class VariableType>
void Solver<IndexType, CoefficientType, VariableType>::update()
{
  // arrays to be exported to device
  for (const SectionData& section : updates)
  {
    if (rawConstrainConnections.size() && section.connection.count)
    {
      uint indexOffset = 0;

      // create flat constrain array for device
      for (uint i = 0; i < section.node.count; i++)
      {
        Constrain newConstrain(indexOffset, 0);

        for (uint j = 0; j < rawConstrainConnections[constrainHeaders.host()->size()].size(); j++)
        {
          constrainIndices.host()->push_back(rawConstrainConnections[constrainHeaders.host()->size()][j]);
          indexOffset++;
        }
        newConstrain.setCount(rawConstrainConnections[constrainHeaders.host()->size()].size());
        constrainHeaders.host()->push_back(newConstrain);
      }

      // send values to device
      if (section.node.count)
      {
        constrainHeaders.syncDevice(section.node);
        constrainConstants.syncDevice(section.node);
      }
      if (section.connection.count)
      {
        constrainIndices.syncDevice(section.connection);
        constrainCoefficients.syncDevice(section.connection);
      }
    }
  }
  rawConstrainConnections.clear();

  const uint count = lastPartition().end();
  // reset aux arrays
  constrainVariableAux[0].resize(count, false);
  constrainVariableAux[1].resize(count, false);

  entityParticleSharedData.syncDevice();
  particles.syncDevice();
  particleIdentities.syncDevice();
  particleDeltas.resize(particles.size(), false);
  particleAuxData.syncDevice();

  partitions.syncDevice();
  (*partitionsCount.host())[0] = partitions.host()->size();
  partitionsCount.syncDevice();
  deviceSections.syncDevice();

  updates.clear();
}

template<class IndexType, class CoefficientType, class VariableType>
uint Solver<IndexType, CoefficientType, VariableType>::newEntityId()
{
  return deviceSections.host()->size();
}

template<class IndexType, class CoefficientType, class VariableType>
uint Solver<IndexType, CoefficientType, VariableType>::newEntityInstanceId()const
{
  return partitions.host()->size();
}

#define classPrefix(x, y, z) template void Solver<x, y, z>

#define declareFunctions(x, y, z) \
  template Solver<x, y, z>::Solver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type); \
  template Solver<x, y, z>::~Solver(); \
  classPrefix(x, y, z)::update(); \
  classPrefix(x, y, z)::commit(const SectionData& sectionData); \
  template uint Solver<x, y, z>::newEntityId(); \
  template uint Solver<x, y, z>::newEntityInstanceId()const;

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)