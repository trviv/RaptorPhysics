#include "FluidSolver.h"

//#define DEBUG_FLUID_SOLVER

#define FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX  0
#define FLUID_COLLISION_SOLVER_CELL_COUNTS          1
#define FLUID_COLLISION_SOLVER_CELL_ARRAYS          2
#define FLUID_COLLISION_SOLVER_REORDER              3
#define FLUID_COLLISION_SOLVER_CALC_DENSITY         4
#define FLUID_COLLISION_SOLVER_CALC_FORCES          5

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : FluidSolver(compute, allocator, false)
{}

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator, bool noCreate)
  : Solver(compute, allocator), EntitySolver<uint, real, Real3>(compute, allocator, SOLVER_FLUID), UniformGridCollisionSolver(compute, allocator)
{
  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

  if (!noCreate)
  {
    create(compute);
  }

  particlesDensity.create(compute, solverHeap, true);
#ifdef DEBUG_FLUID_SOLVER
  particlesLambda.create(compute, solverHeap, true);
  particlesTemp[0].create(compute, solverHeap, true);
  particlesTemp[1].create(compute, solverHeap, true);
#else
  particlesLambda.create(compute, solverHeap);
  particlesTemp[0].create(compute, solverHeap);
  particlesTemp[1].create(compute, solverHeap);
#endif
}

void FluidSolver::create(ComputeInterface* compute)
{
  includeFiles.push_back("UniformGridCollisionSolver.shader");
  includeFiles.push_back("FluidSolverCommon.h");

  registerShader(compute, "FluidSolver.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("reorderFluidParticles"));
  kernels.push_back(programs[0].createKernel("calculateDensity"));
  kernels.push_back(programs[0].createKernel("calculateForces"));

  invMaxRadius.host()->push_back(-1.f);

  createUtilities();
}

void FluidSolver::updateRadius()
{
  // update kernel radius
  float kernelRadius = 0.f;
  for (auto& esd : *entitySharedData.host())
  {
    kernelRadius = max(esd.fluidSolverData.fluidKernelRadius, kernelRadius);
  }

  if (1.f/kernelRadius != invMaxRadius.host()->at(0))
  {
    (*invMaxRadius.host())[0] = 1.f/kernelRadius;
    invMaxRadius.syncDevice();
  }
}

void FluidSolver::rearrangeParticles(uint particleCount)
{
  size_t workgroupSize[3], workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, particleCount);

  ComputeUtil::get(0)->copyBuffer(compute, particles.device(), particlesCopy.device(), 0, 0, sizeof(ParticleStruct)*particleCount);
  ComputeUtil::get(0)->copyBuffer(compute, particlesPredicted.device(), particlesPredictedCopy.device(), 0, 0, sizeof(ParticleStruct)*particleCount);
  ComputeUtil::get(0)->copyBuffer(compute, particleDifferential.device(), particleDifferentialCopy.device(), 0, 0, sizeof(ParticleDifferential)*particleCount);
  ComputeUtil::get(0)->copyBuffer(compute, gridParticleCellIndex.device(), particlesLambda.device(), 0, 0, sizeof(uint)*particleCount);

  ComputeMemory* buffers[] = {
    particles.device(),
    particlesCopy.device(),
    particleDifferential.device(),
    particleDifferentialCopy.device(),
    particlesPredicted.device(),
    particlesPredictedCopy.device(),
    gridParticleCellIndex.device(),
    particlesLambda.device(),
    gridCellParticleIndices.device(),
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[FLUID_COLLISION_SOLVER_REORDER].setArgs(buffers, bufferCount);
  kernels[FLUID_COLLISION_SOLVER_REORDER].setArg<uint>(&particleCount, bufferCount);

  compute->execute(kernels[FLUID_COLLISION_SOLVER_REORDER], workgroupSize, workgroupCount);
}

void FluidSolver::solve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  updateRadius();

  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(workgroupSize[0] * workgroupCount[0]), false);
  }

  if (gridParticleCellIndex.size() < particleCount)
  {
    particlesCopy.resize(particleCount, false);
    particlesPredictedCopy.resize(particleCount, false);
    particleDifferentialCopy.resize(particleCount, false);
    particlesDensity.resize(particleCount, false);
    particlesLambda.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
  }

  for (int i=0; i<iterations; i++)
  {
    {
      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

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
      kernels[FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&nodeBatchCount, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&particleCount, bufferCount + 1);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    particleGroupBoundingBoxes.syncHost();
    compute->sync();
#endif

    // find bounding box for the simulation space
    ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(workgroupSize[0] * workgroupCount[0]));

#ifdef DEBUG_FLUID_SOLVER
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
      kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&particleCount, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSizeExp, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    gridCellParticleCount.syncHost();
    gridParticleCellIndex.syncHost();
    compute->sync();
#endif

    // get prefix sum for each
    ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_SOLVER
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
      kernels[FLUID_COLLISION_SOLVER_CELL_ARRAYS].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CELL_ARRAYS].setArg<uint>(&particleCount, bufferCount);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    gridCellParticleIndices.syncHost();
    gridCellParticleOffsets.syncHost();
    compute->sync();
  #endif

    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    rearrangeParticles(particleCount);

    {
      ComputeMemory* buffers[] = {
        particlesDensity.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        particlesPredicted.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    particlesDensity.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particleForce.device(),
        particlesDensity.device(),
        particleDifferential.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        particlesPredicted.device(),
        entitySharedData.device(),
        particleCollisionData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&particleCount, bufferCount + 2);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<float>(&timeStep, bufferCount + 3);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CALC_FORCES], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    particleForce.syncHost();
    compute->sync();
#endif
  }
}

void FluidSolver::update()
{
  if (!updates.size()) return;

  EntitySolver::update();

  for (auto& esd : *entitySharedData.host())
  {
    esd.fluidSolverData.fluidKernelFunctionConstant[0] = poly6FunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[1] = spikyFunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[2] = viscosityFunctionConstant(esd.fluidSolverData.fluidKernelRadius);
  }

  entitySharedData.syncDevice();
}
