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

  particleSharedData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED), true);
  particles.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE), true);
  particleIdentities.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY), true);
  particleDeltas.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA), false);
  particleDifferential.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF), false);
  particleAuxData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX), true);
  particleRigidData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_RIGID), true);
#ifdef DEBUG_SOLVERS
  particlesTemp[0].create(compute, NULL, true);
  particlesTemp[1].create(compute, NULL, true);
  entityOffsets.create(compute, NULL, true);
  entityOffsetCount.create(compute, NULL, true);
#else
  particlesTemp[0].create(compute, NULL, false);
  particlesTemp[1].create(compute, NULL, false);
  entityOffsets.create(compute, NULL, false);
  entityOffsetCount.create(compute, NULL, false);
#endif

  deviceSections.create(compute, NULL, true);

  iterations = 1;

  includeFiles.push_back("ComputeHeader.shader");
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

  updateInfo.offsets[SECTION_DATA_NODE] = nodes();
  updateInfo.counts[SECTION_DATA_NODE] = constrainConstants.host()->size() - nodes();

  updateInfo.offsets[SECTION_DATA_CONNECTION] = connectionCount();
  updateInfo.counts[SECTION_DATA_CONNECTION] = constrainCoefficients.host()->size() - connectionCount();

  updateInfo.offsets[SECTION_DATA_COMMON_NODE] = commonNodeCount();
  updateInfo.counts[SECTION_DATA_COMMON_NODE] = constrainConstants.host()->size() / getInstanceId(sectionData.identity) - commonNodeCount();

  updates.push_back(updateInfo);
}

template<class IndexType, class CoefficientType, class VariableType>
void Solver<IndexType, CoefficientType, VariableType>::update()
{
  // arrays to be exported to device
  for (const SectionData& section : updates)
  {
    if (rawConstrainConnections.size() && section.counts[SECTION_DATA_CONNECTION])
    {
      uint indexOffset = 0;

      // create flat constrain array for device
      for (uint i = 0; i < section.counts[SECTION_DATA_COMMON_NODE]; i++)
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
      if (section.counts[SECTION_DATA_COMMON_NODE])
      {
        constrainHeaders.syncDevice(section.offsets[SECTION_DATA_COMMON_NODE], section.counts[SECTION_DATA_COMMON_NODE]);
        constrainConstants.syncDevice(section.offsets[SECTION_DATA_COMMON_NODE], section.counts[SECTION_DATA_COMMON_NODE]);
      }
      if (section.counts[SECTION_DATA_CONNECTION])
      {
        constrainIndices.syncDevice(section.offsets[SECTION_DATA_CONNECTION], section.counts[SECTION_DATA_CONNECTION]);
        constrainCoefficients.syncDevice(section.offsets[SECTION_DATA_CONNECTION], section.counts[SECTION_DATA_CONNECTION]);
      }
    }
  }
  rawConstrainConnections.clear();

  // reset aux arrays
  constrainVariableAux[0].resize(nodes(), false);
  constrainVariableAux[1].resize(nodes(), false);

  updates.clear();
}

template<class IndexType, class CoefficientType, class VariableType>
uint Solver<IndexType, CoefficientType, VariableType>::newEntityId()
{
  return deviceSections.host()->size();
}

template<class IndexType, class CoefficientType, class VariableType>
uint Solver<IndexType, CoefficientType, VariableType>::uniqueEntityCount()const
{
  return deviceSections.host()->size();
}

template<class IndexType, class CoefficientType, class VariableType>
uint Solver<IndexType, CoefficientType, VariableType>::totalEntityCount()const
{
  if (deviceSections.host()->size())
  {
    return getEntityOffset(deviceSections.host()->back().identity) + getInstanceId(deviceSections.host()->back().identity);
  }

  return 0;
}

#define classPrefix(x, y, z) template void Solver<x, y, z>

#define declareFunctions(x, y, z) \
  template Solver<x, y, z>::Solver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type); \
  template Solver<x, y, z>::~Solver(); \
  classPrefix(x, y, z)::update(); \
  classPrefix(x, y, z)::commit(const SectionData& sectionData); \
  template uint Solver<x, y, z>::newEntityId(); \
  template uint Solver<x, y, z>::uniqueEntityCount()const; \
  template uint Solver<x, y, z>::totalEntityCount()const;

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)