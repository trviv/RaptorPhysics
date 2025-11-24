/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include "LinearSolver.h"
#include "FluidSolverCommon.h"
#include "../Common/ParticleStruct.h"
#include "Collision/UniformGridCollisionSolver.h"

/*!
@class Class to solve fluid constraints.
*/
class FluidSolver : public EntitySolverType, protected UniformGridCollisionSolver
{
protected:

  friend class PhysicsSystem;

  /* Kernels common to fluid solvers */
  ComputeKernel reorderFluidParticles;
  ComputeKernel calculateDensity;
  ComputeKernel createBoundingBoxesCoupling;
  ComputeKernel calculateCouplingData;
  ComputeKernel reorderCouplingParticles;
  ComputeKernel reorderBoundaryParticles;

  DeviceArray <float> particlesDensity;
  DeviceArray <float> particlesLambda;

  DeviceArray <ParticleStruct>&particlesCopy = particlesTemp[1];
  DeviceArray <ParticleStruct>&particlesPredictedCopy = UniformGridCollisionSolver::particlesBufferTemp;
  DeviceArray <ParticleStruct>&particleDifferentialCopy = particlesTemp[0];

  /* Physics system related variables, used for fluid-solid coupling */
  DeviceArray <uint>  boundaryGridParticleCellIndex;
  DeviceArray <uint> &boundaryGridCellParticleOffsets;
  DeviceArray <uint>  boundaryGridCellParticleIndices;
  DeviceArray <uint>  boundaryGridCellParticleCount;
  DeviceArray <uint>  boundaryGridParticleSystemIndex;

  DeviceArray <ParticleStruct>  boundaryParticlePositions;
  DeviceArray <ParticleStruct>  boundaryParticleDifferential;
  DeviceArray <ParticleCouplingData> boundaryParticleCouplingData;

  ComputeMemory *systemParticlePositions;
  ComputeMemory *systemParticleDifferential;
  ComputeMemory *systemParticleCouplingData;
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

  const DeviceArray<float>& getParticlesDensity()const {return particlesDensity;}
};

#endif
