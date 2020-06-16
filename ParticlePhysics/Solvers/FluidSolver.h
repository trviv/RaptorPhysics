#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include "LinearSolver.h"
#include "FluidSolverCommon.h"
#include "../Common/ParticleStruct.h"
#include "Collision/UniformGridCollisionSolver.h"

/*!
@class Class to solve fluid constraints.
*/
class FluidSolver : public EntitySolver<uint, real, Real3>, protected UniformGridCollisionSolver
{
protected:

  friend class PhysicsSystem;

  /* Kernels common to fluid solvers */
  ComputeKernel reorderFluidParticles;
  ComputeKernel calculateDensity;
  ComputeKernel createBoundingBoxesCoupling;
  ComputeKernel calculateCouplingData;
  ComputeKernel reorderCouplingParticles;

  DeviceArray <float> particlesDensity;
  DeviceArray <float> particlesLambda;

  DeviceArray <ParticleStruct>&particlesCopy = particlesTemp[1];
  DeviceArray <ParticleStruct>&particlesPredictedCopy = UniformGridCollisionSolver::particlesBufferTemp;
  DeviceArray <ParticleStruct>&particleDifferentialCopy = particlesTemp[0];

  /* Physics system related variables, used for fluid-solid coupling */
  DeviceArray <uint>  systemGridParticleCellIndex;
  DeviceArray <uint> &systemGridCellParticleOffsets;
  DeviceArray <uint>  systemGridCellParticleIndices;
  DeviceArray <uint>  systemGridCellParticleCount;
  DeviceArray <uint>  systemGridParticleSystemIndex;

  DeviceArray <ParticleStruct>  systemParticlePositionsCopy;
  DeviceArray <ParticleStruct>  systemParticleDifferentialCopy;
  ComputeMemory *systemParticlePositions;
  ComputeMemory *systemParticleDifferential;
  ComputeMemory *systemParticleCollisionData;
  ComputeMemory *systemParticleForce;
  ComputeMemory *systemSettings;
  uint systemParticleCount;
  uint systemNonFluidParticleCount;

  void rearrangeParticles(uint particleCount);

  void update();

  void updateRadius();

  void constructGrid();

  float calculateGradientConstant(const ParticleSharedData& entitySharedData)const;

  void constructBoundaryGrid();

  void allocateBoundingBoxes(const uint particleCount);

  void allocatePositionBuffers(const uint particleCount);

  void allocateIndexBuffers(const uint particleCount);

  void allocateBuffers(const uint particleCount);

  FluidSolver(ComputeInterface* compute, SharedAllocator* allocator, bool noCreate);

public:

  FluidSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);

  void calculateParticleCouplingData(DeviceArray<ParticleCouplingData> &particleCouplingData,
                                     DeviceArray<ParticleStruct> &particles,
                                     DeviceArray<ParticleCollisionData> &particleCollisionData,
                                     uint particleCount);
};

#endif
