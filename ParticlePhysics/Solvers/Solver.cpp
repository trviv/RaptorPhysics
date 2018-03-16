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
void Solver<IndexType, CoefficientType, VariableType>::commit()
{
}

template<class IndexType, class CoefficientType, class VariableType>
void Solver<IndexType, CoefficientType, VariableType>::update()
{
  // arrays to be exported to device
  for (const SectionData& section : updates)
  {
    if (rawConstrainConnections.size())
    {
      // create flat constrain array for device
      for (uint i = section.offsets[SECTION_DATA_NODE]; i < section.offsets[SECTION_DATA_NODE] + section.counts[SECTION_DATA_NODE]; i++)
      {
        constrainHeaders.host()->push_back(Constrain(constrainIndices.host()->size(), 0));
        for (uint j = 0; j < rawConstrainConnections[i].size(); j++)
        {
          constrainIndices.host()->push_back(rawConstrainConnections[i][j]);
        }
        (*constrainHeaders.host())[i].setCount(rawConstrainConnections[i].size()); // the constrain header
      }

      rawConstrainConnections.clear();

      // send values to device
      if (section.counts[SECTION_DATA_NODE])
      {
        constrainHeaders.syncDevice(section.offsets[SECTION_DATA_NODE], section.counts[SECTION_DATA_NODE]);
        constrainConstants.syncDevice(section.offsets[SECTION_DATA_NODE], section.counts[SECTION_DATA_NODE]);
      }
      if (section.counts[SECTION_DATA_CONNECTION])
      {
        constrainIndices.syncDevice(section.offsets[SECTION_DATA_CONNECTION], section.counts[SECTION_DATA_CONNECTION]);
        constrainCoefficients.syncDevice(section.offsets[SECTION_DATA_CONNECTION], section.counts[SECTION_DATA_CONNECTION]);
      }
    }
  }

  // reset aux arrays
  constrainVariableAux[0].resize(nodes(), false);
  constrainVariableAux[1].resize(nodes(), false);

  updates.clear();
}

#define classPrefix(x, y, z) template void Solver<x, y, z>

#define declareFunctions(x, y, z) \
  template Solver<x, y, z>::Solver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type); \
  template Solver<x, y, z>::~Solver(); \
  classPrefix(x, y, z)::update(); \
  classPrefix(x, y, z)::commit();

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)