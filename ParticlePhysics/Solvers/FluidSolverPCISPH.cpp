#include "FluidSolverPCISPH.h"

//#define DEBUG_FLUID_PCISPH_SOLVER

#define FLUID_COLLISION_SOLVER_PCISPH_CREATE_BOUNDING_BOX 0
#define FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS         1
#define FLUID_COLLISION_SOLVER_PCISPH_CELL_ARRAYS         2
#define FLUID_COLLISION_SOLVER_PCISPH_PREDICT             3
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY       4
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE       5
#define FLUID_COLLISION_SOLVER_PCISPH_CALC_FORCES         6
#define FLUID_COLLISION_SOLVER_PCISPH_UPDATE_POSITION     7

FluidSolverPCISPH::FluidSolverPCISPH(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), FluidSolver(compute, allocator, true), beta(0)
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

  registerShader(compute, "FluidSolverPCISPH.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("predictionStep"));
  kernels.push_back(programs[0].createKernel("calculateDensity"));
  kernels.push_back(programs[0].createKernel("calculatePressure"));
  kernels.push_back(programs[0].createKernel("calculateForces"));
  kernels.push_back(programs[0].createKernel("updatePositions"));

  invMaxRadius.host()->push_back(-1.f);

  createUtilities();
}

void FluidSolverPCISPH::solve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  updateRadius();

  if (beta == 0)
  {
    beta = 2.f * mSqr(timeStep) * mSqr(1.f/entitySharedData.host()->at(0).sharedInvMass) * mSqr(entitySharedData.host()->at(0).invRestDensity);
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
    particlesPressureForce.resize(particleCount, false);
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
  ComputeUtil::get(0)->clearBuffer(compute, particlesPressureForce.device(), particleCount * 4);
  ComputeUtil::get(0)->clearBuffer(compute, particlesPressure.device(), particleCount);

  {
    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleGroupBoundingBoxes.device(),
      particlesPredicted.device(),
      entitySharedData.device(),
      particleAuxData.device(),
      partitions.device(),
      entityLocations.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CREATE_BOUNDING_BOX].setArg<uint>(&nodeBatchCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CREATE_BOUNDING_BOX].setArg<uint>(&particleCount, bufferCount + 1);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CREATE_BOUNDING_BOX], groupWorkgroupSize, groupWorkgroupCount);
  }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

  // find bounding box for the simulation space
  ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(groupWorkgroupSize[0] * groupWorkgroupCount[0]));

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  systemBoundingBox.syncHost();
  compute->sync();
#endif

  // clear index offset buffer
  ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, gridCellParticleCount.device(), gridElements);

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      gridCellParticleCount.device(),
      gridParticleCellIndex.device(),
      particlesPredicted.device(),
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS].setArg<uint>(&particleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS].setArg<uint>(&gridSizeExp, bufferCount + 2);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount, compute->simdSize());

    ComputeMemory* buffers[] = {
      gridCellParticleIndices.device(),
      gridCellParticleOffsets.device(),
      gridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_ARRAYS].setArg<uint>(&particleCount, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

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
        particlesPressureForce.device(),
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
        gridCellParticleIndices.device(),
        gridParticleCellIndex.device(),
        particlesPredicted.device(),
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
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY].setArg<float>(&beta, bufferCount + 4);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_DENSITTY], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particlesDensity.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particlesPressure.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
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
      kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE].setArg<float>(&beta, bufferCount + 4);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_CALC_PRESSURE], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
    particlesPressure.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particlesPressureForce.device(),
        particlesDensity.device(),
        particlesPressure.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
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
    particlesPressureForce.syncHost();
    compute->sync();
#endif
  }

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particlesPredicted.device(),
      particlesPressureForce.device(),
      entitySharedData.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_UPDATE_POSITION].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_UPDATE_POSITION].setArg<uint>(&particleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PCISPH_UPDATE_POSITION].setArg<float>(&timeStep, bufferCount + 1);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PCISPH_UPDATE_POSITION], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PCISPH_SOLVER
  compute->sync();
#endif
}
