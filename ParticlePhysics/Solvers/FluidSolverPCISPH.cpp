#include "FluidSolverPCISPH.h"

//#define DEBUG_FLUID_PCISPH_SOLVER

#define FLUID_COLLISION_SOLVER_PCISPH_PREDICT             0
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE       1
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES         2

FluidSolverPCISPH::FluidSolverPCISPH(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), FluidSolver(compute, allocator, true), invBeta(0)
{
  FluidSolverPCISPH::create(compute);

  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  particlesPressure.create(compute, solverHeap, true);
#else
  particlesPressure.create(compute, solverHeap);
#endif
}

void FluidSolverPCISPH::create(ComputeInterface* compute)
{
  registerShader(compute, "FluidSolverPCISPH.shader", NULL, NULL);

  createBoundingBoxes     = programs[0].createKernel("createBoundingBoxes");
  createGridCellHistogram = programs[0].createKernel("createGridCellHistogram");
  createGridCellArrays    = programs[0].createKernel("createGridCellArrays");
  reorderFluidParticles   = programs[0].createKernel("reorderFluidParticles");
  calculateDensity        = programs[0].createKernel("calculateDensity");
  calculateCouplingData   = programs[0].createKernel("calculateCouplingData");
  reorderCouplingParticles    = programs[0].createKernel("reorderCouplingParticles");
  createBoundingBoxesCoupling = programs[0].createKernel("createBoundingBoxesCollision");
  kernels.push_back(programs[0].createKernel("predictionStep"));
  kernels.push_back(programs[0].createKernel("calculatePressure"));
  kernels.push_back(programs[0].createKernel("calculateForces"));

  invMaxRadius.host()->push_back(-1.f);

  createUtilities();
}

void FluidSolverPCISPH::solve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  updateRadius();

  if (invBeta == 0)
  {
    invBeta = 1.f/(2.f * mSqr(timeStep) * mSqr(1.f/entitySharedData.host()->at(0).sharedInvMass) * mSqr(entitySharedData.host()->at(0).invRestDensity));
  }

  allocateBuffers(particleCount);

  ComputeUtil::get(0)->clearBuffer(compute, particlesNextPosition.device(), particleCount * 4);
  ComputeUtil::get(0)->clearBuffer(compute, particlesNextVelocity.device(), particleCount * 4);
  ComputeUtil::get(0)->clearBuffer(compute, particleForce.device(), particleCount * 4);
  ComputeUtil::get(0)->clearBuffer(compute, particlesPressure.device(), particleCount);

  constructGrid();

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, particleCount);

  for (int i=0; i<iterations; i++)
  {
    {
      ComputeMemory* buffers[] = {
        particlesNextPosition.device(),
        particlesNextVelocity.device(),
        particlesPredicted.device(),
        particleDifferential.device(),
        particleForce.device(),
        entitySharedData.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_PREDICT].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_PREDICT].setArg<uint>(&particleCount, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_PREDICT].setArg<float>(&timeStep, bufferCount + 1);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_PREDICT], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particlesNextPosition.syncHost();
    particlesNextVelocity.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particlesDensity.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        particlesNextPosition.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      calculateDensity.setArgs(buffers, bufferCount);
      calculateDensity.setArg<ushort>(&gridSize, bufferCount);
      calculateDensity.setArg<ushort>(&gridSizeExp, bufferCount + 1);
      calculateDensity.setArg<uint>(&particleCount, bufferCount + 2);
      calculateDensity.setArg<float>(&timeStep, bufferCount + 3);
      calculateDensity.setArg<float>(&invBeta, bufferCount + 4);

      compute->execute(calculateDensity, workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particlesDensity.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particlesPressure.device(),
        particlesDensity.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        particlesNextPosition.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<ushort>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<ushort>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<uint>(&particleCount, bufferCount + 2);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<float>(&timeStep, bufferCount + 3);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<float>(&invBeta, bufferCount + 4);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particlesPressure.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particleForce.device(),
        particlesDensity.device(),
        particlesPressure.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        particlesNextPosition.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArg<ushort>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArg<ushort>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particleForce.syncHost();
    compute->sync();
#endif
  }
}

void FluidSolverPCISPH::update()
{
  FluidSolver::update();

  for (auto& esd : *entitySharedData.host())
  {
    esd.fluidSolverData.fluidKernelFunctionConstant[0] = poly6FunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[1] = spikyFunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[2] = 1.f/calculateGradientConstant(esd);
  }

  entitySharedData.syncDevice();
}
