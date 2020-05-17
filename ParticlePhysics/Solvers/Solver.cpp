#include "Solver.h"

Solver::Solver(ComputeInterface* compute, SharedAllocator* allocator) :
  compute(compute), allocator(allocator)
{

}

template<class IndexType, class CoefficientType, class VariableType>
EntitySolver<IndexType, CoefficientType, VariableType>::EntitySolver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type) :
  Solver(compute, allocator), SolverData<IndexType, CoefficientType, VariableType>(), type(type)
{
  this->constrainHeaders.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_HEADERS), true);
  this->constrainIndices.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_INDICES), true);
  this->constrainCoefficients.create(compute, allocator->getHeap(COMPUTE_HEAP_CONSTRAIN_COEFFICIENTS), true);
  this->constrainConstants.create(compute, NULL, true);
  this->constrainVariableAux[0].create(compute, NULL, false);
  this->constrainVariableAux[1].create(compute, NULL, false);

  this->entitySharedData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED), true);

  this->particles.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE), true);
  this->particlesPredicted.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED), true);
  this->particleDifferential.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF), false);
  this->particleForce.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_FORCE), false);
  this->particleAuxData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX), true);
  this->particleRigidData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_RIGID), true);

#ifdef DEBUG_SOLVERS
  this->particlesTemp[0].create(compute, NULL, true);
  this->particlesTemp[1].create(compute, NULL, true);
#else
  this->particlesTemp[0].create(compute, NULL, false);
  this->particlesTemp[1].create(compute, NULL, false);
#endif

  this->particleCollisionData.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION), true);

  this->partitions.create(compute, allocator->getHeap(COMPUTE_HEAP_PARTITIONS), true);
  this->partitionsCount.create(compute, NULL, true);
  this->partitionsCount.host()->reserve(1);
  this->partitionsCount.host()->resize(1);
  this->entityLocations.create(compute, allocator->getHeap(COMPUTE_HEAP_SECTIONS), true);

  this->iterations = 1;

  this->includeFiles.push_back("ComputeHeader.shader");
  this->includeFiles.push_back("ComputeShared.h");
  this->includeFiles.push_back("ConstrainStruct.h");
  this->includeFiles.push_back("ParticleStruct.h");
}

template<class IndexType, class CoefficientType, class VariableType>
EntitySolver<IndexType, CoefficientType, VariableType>::~EntitySolver()
{
}

template<class IndexType, class CoefficientType, class VariableType>
void EntitySolver<IndexType, CoefficientType, VariableType>::commit()
{
  flatArray<CoefficientType>(*this->constrainCoefficients.host(), this->rawConstrainCoefficients);

  EntityLocation updateInfo;

  updateInfo.node.offset = this->nodes();
  updateInfo.node.count = (uint)this->constrainConstants.host()->size() - this->nodes();

  updateInfo.connection.offset = this->connectionCount();
  updateInfo.connection.count = (uint)this->constrainCoefficients.host()->size() - this->connectionCount();

  this->updates.push_back(updateInfo);
  this->entityLocations.host()->push_back(updateInfo);
}

template<class IndexType, class CoefficientType, class VariableType>
void EntitySolver<IndexType, CoefficientType, VariableType>::update()
{
  // arrays to be exported to device
  for (const EntityLocation& section : this->updates)
  {
    if (this->rawConstrainConnections.size() && section.connection.count)
    {
      uint indexOffset = 0;

      // create flat constrain array for device
      for (uint i = 0; i < section.node.count; i++)
      {
        Constrain newConstrain(indexOffset, 0);

        for (uint j = 0; j < this->rawConstrainConnections[this->constrainHeaders.host()->size()].size(); j++)
        {
          this->constrainIndices.host()->push_back(this->rawConstrainConnections[this->constrainHeaders.host()->size()][j]);
          indexOffset++;
        }
        newConstrain.setCount((uint)this->rawConstrainConnections[this->constrainHeaders.host()->size()].size());
        this->constrainHeaders.host()->push_back(newConstrain);
      }

      // send values to device
      if (section.node.count)
      {
        this->constrainHeaders.syncDevice(section.node);
        this->constrainConstants.syncDevice(section.node);
      }
      if (section.connection.count)
      {
        this->constrainIndices.syncDevice(section.connection);
        this->constrainCoefficients.syncDevice(section.connection);
      }
    }
  }
  this->rawConstrainConnections.clear();

  const uint count = this->lastPartition().end();
  // reset aux arrays
  this->constrainVariableAux[0].resize(count, false);
  this->constrainVariableAux[1].resize(count, false);

  this->entitySharedData.syncDevice();
  this->particles.syncDevice();
  this->particlesPredicted.resize(this->particles.size(), false);
  this->compute->copyBuffer(this->particles.device(), this->particlesPredicted.device(), 0, 0, this->particles.size() * sizeof(ParticleStruct));
  this->particleDifferential.resize(this->particles.size(), false);
  this->particleForce.resize(this->particles.size(), false);
  this->particleCollisionData.syncDevice();
  this->particleAuxData.syncDevice();

  this->partitions.syncDevice();
  (*this->partitionsCount.host())[0] = (uint)this->partitions.host()->size();
  this->partitionsCount.syncDevice();
  this->entityLocations.syncDevice();

  this->updates.clear();
}

template<class IndexType, class CoefficientType, class VariableType>
uint EntitySolver<IndexType, CoefficientType, VariableType>::newEntityId()
{
  return (uint)this->entityLocations.host()->size();
}

template<class IndexType, class CoefficientType, class VariableType>
uint EntitySolver<IndexType, CoefficientType, VariableType>::newEntityInstanceId()const
{
  return (uint)this->partitions.host()->size();
}

#define classPrefix(x, y, z) template void EntitySolver<x, y, z>

#define declareFunctions(x, y, z) \
  template EntitySolver<x, y, z>::EntitySolver(ComputeInterface* compute, SharedAllocator* allocator, SolverType type); \
  template EntitySolver<x, y, z>::~EntitySolver(); \
  classPrefix(x, y, z)::update(); \
  classPrefix(x, y, z)::commit(); \
  template uint EntitySolver<x, y, z>::newEntityId(); \
  template uint EntitySolver<x, y, z>::newEntityInstanceId()const;

declareFunctions(ushort, real, real)
declareFunctions(uint, real, real)
declareFunctions(ushort, real, Real3)
declareFunctions(uint, real, Real3)
