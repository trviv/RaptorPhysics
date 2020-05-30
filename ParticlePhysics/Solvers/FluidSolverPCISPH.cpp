#include "FluidSolverPCISPH.h"

//#define DEBUG_FLUID_PCISPH_SOLVER

#define FLUID_COLLISION_SOLVER_REORDER                    3
#define FLUID_COLLISION_SOLVER_PCISPH_PREDICT             4
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY       5
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE       6
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES         7

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
  includeFiles.push_back("UniformGridCollisionSolver.shader");
  includeFiles.push_back("FluidSolverCommon.h");

  registerShader(compute, "FluidSolverPCISPH.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("reorderFluidParticles"));
  kernels.push_back(programs[0].createKernel("predictionStep"));
  kernels.push_back(programs[0].createKernel("calculateDensity"));
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

  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t groupWorkgroupSize[3], groupWorkgroupCount[3];
  compute->configureSize(groupWorkgroupSize, groupWorkgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < groupWorkgroupSize[0] * groupWorkgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(groupWorkgroupSize[0] * groupWorkgroupCount[0]), false);
  }

  if (gridParticleCellIndex.size() < particleCount)
  {
    particlesNextPosition.resize(particleCount, false);
    particlesNextVelocity.resize(particleCount, false);
    particleDifferentialCopy.resize(particleCount, false);
    particlesPressure.resize(particleCount, false);
    particlesDensity.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
  }

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
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<uint>(&particleCount, bufferCount + 2);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<float>(&timeStep, bufferCount + 3);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<float>(&invBeta, bufferCount + 4);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY], workgroupSize, workgroupCount);
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
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<uint>(&gridSizeExp, bufferCount + 1);
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
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES].setArg<uint>(&gridSizeExp, bufferCount + 1);
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
