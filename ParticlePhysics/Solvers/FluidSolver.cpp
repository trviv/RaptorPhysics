#include "FluidSolver.h"

//#define DEBUG_FLUID_SOLVER

#define FLUID_COLLISION_SOLVER_CALC_FORCES          0
#define FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS   1
#define FLUID_COLLISION_SOLVER_SYSTEM_CELL_ARRAYS   2
#define FLUID_COLLISION_SOLVER_SYSTEM_REORDER       3

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : FluidSolver(compute, allocator, false)
{}

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator, bool noCreate)
  : Solver(compute, allocator), EntitySolver<uint, real, Real3>(compute, allocator, SOLVER_FLUID), UniformGridCollisionSolver(compute, allocator), systemGridCellParticleOffsets(systemGridCellParticleCount), systemParticleCount(0), systemNonFluidParticleCount(0)
{
  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

  includeFiles.push_back("UniformGridCollisionSolver.shader");
  includeFiles.push_back("FluidSolverCommon.h");

  if (!noCreate)
  {
    create(compute);
  }

  particlesDensity.create(compute, solverHeap, true);
#ifdef DEBUG_FLUID_SOLVER
  particlesLambda.create(compute, solverHeap, true);
  particlesTemp[0].create(compute, solverHeap, true);
  particlesTemp[1].create(compute, solverHeap, true);
  systemGridParticleCellIndex.create(compute, solverHeap, true);
  systemGridCellParticleIndices.create(compute, solverHeap, true);
  systemGridCellParticleCount.create(compute, solverHeap, true);
  systemParticlePositionsCopy.create(compute, solverHeap, true);
  systemParticleDifferentialCopy.create(compute, solverHeap, true);
  systemGridParticleSystemIndex.create(compute, solverHeap, true);
#else
  particlesLambda.create(compute, solverHeap);
  particlesTemp[0].create(compute, solverHeap);
  particlesTemp[1].create(compute, solverHeap);
  systemGridParticleCellIndex.create(compute, solverHeap);
  systemGridCellParticleIndices.create(compute, solverHeap);
  systemGridCellParticleCount.create(compute, solverHeap);
  systemParticlePositionsCopy.create(compute, solverHeap);
  systemParticleDifferentialCopy.create(compute, solverHeap);
  systemGridParticleSystemIndex.create(compute, solverHeap);
#endif
}

void FluidSolver::create(ComputeInterface* compute)
{
  registerShader(compute, "FluidSolver.shader", NULL, NULL);

  createBoundingBoxes     = programs[0].createKernel("createBoundingBoxes");
  createGridCellHistogram = programs[0].createKernel("createGridCellHistogram");
  createGridCellArrays    = programs[0].createKernel("createGridCellArrays");
  reorderFluidParticles   = programs[0].createKernel("reorderFluidParticles");
  calculateDensity        = programs[0].createKernel("calculateDensity");
  calculateCouplingData   = programs[0].createKernel("calculateCouplingData");
  reorderCouplingParticles    = programs[0].createKernel("reorderCouplingParticles");
  createBoundingBoxesCoupling = programs[0].createKernel("createBoundingBoxesCollision");
  kernels.push_back(programs[0].createKernel("calculateForces"));
  kernels.push_back(programs[0].createKernel("createBoundaryGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createBoundaryGridCellArrays"));
  kernels.push_back(programs[0].createKernel("reorderBoundaryParticles"));

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

float FluidSolver::calculateGradientConstant(const ParticleSharedData& entitySharedData)const
{
  // create prototype neighbourhood
  int kernelFactor = entitySharedData.fluidSolverData.fluidKernelRadius / (2.f * entitySharedData.sharedRadius);
  vector<Real3> neighbourParticles;
  neighbourParticles.reserve((2*kernelFactor+1) * (2*kernelFactor+1) * (2*kernelFactor+1));
  for (int i=-kernelFactor; i<=kernelFactor; i++)
  {
    for (int j=-kernelFactor; j<=kernelFactor; j++)
    {
      for (int k=-kernelFactor; k<=kernelFactor; k++)
      {
        neighbourParticles.push_back(Real3(i, j, k));
      }
    }
  }

  // calculate gradient magnitude contribution from neighbours
  float sumGradientMagnitude = 0.f;
  for (const auto& position : neighbourParticles)
  {
    const Real3 collisionVector = position * 2.f * entitySharedData.sharedRadius;
    const Real3 gradient = collisionVector * spikyFunctionGradientVariable(collisionVector.length(), entitySharedData.fluidSolverData.fluidKernelRadius);
    sumGradientMagnitude += gradient.lengthSq();
  }

  return sumGradientMagnitude * mSqr(spikyFunctionConstant(entitySharedData.fluidSolverData.fluidKernelRadius));
}

void FluidSolver::rearrangeParticles(uint particleCount)
{
  size_t workgroupSize[3], workgroupCount[3];

  uint multiplier = 1;
#ifdef FLUID_SOLVER_SORTED_REARRANGE
  multiplier = 2;
#endif

  compute->configureSize(workgroupSize, workgroupCount, mAlignBy(particleCount, multiplier));

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
#ifdef FLUID_SOLVER_SORTED_REARRANGE
    systemBoundingBox.device(),
    invMaxRadius.device(),
#endif
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  reorderFluidParticles.setArgs(buffers, bufferCount);
  reorderFluidParticles.setArg<uint>(&particleCount, bufferCount);
#ifdef FLUID_SOLVER_SORTED_REARRANGE
  reorderFluidParticles.setArg<ushort>(&gridSize, bufferCount + 1);
  reorderFluidParticles.setArg<ushort>(&gridSizeExp, bufferCount + 2);
  reorderFluidParticles.setSharedMemArg(sizeof(uint)*4*workgroupSize[0]*workgroupSize[1]*workgroupSize[2]*multiplier, bufferCount + 3);
#endif

  compute->execute(reorderFluidParticles, workgroupSize, workgroupCount);
}

void FluidSolver::constructGrid()
{
  uint particleCount = lastPartition().end();
  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleGroupBoundingBoxes.device(),
      particlesPredicted.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    createBoundingBoxes.setArgs(buffers, bufferCount);
    createBoundingBoxes.setArg<uint>(&nodeBatchCount, bufferCount);
    createBoundingBoxes.setArg<uint>(&particleCount, bufferCount + 1);

    compute->execute(createBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_FLUID_SOLVER
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

    // find bounding box for the simulation space
    ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(workgroupSize[0] * workgroupCount[0]));
  }

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
    createGridCellHistogram.setArgs(buffers, bufferCount);
    createGridCellHistogram.setArg<uint>(&particleCount, bufferCount);
    createGridCellHistogram.setArg<ushort>(&gridSize, bufferCount + 1);
    createGridCellHistogram.setArg<ushort>(&gridSizeExp, bufferCount + 2);

    compute->execute(createGridCellHistogram, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

//  constructBoundaryGrid();

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
    createGridCellArrays.setArgs(buffers, bufferCount);
    createGridCellArrays.setArg<uint>(&particleCount, bufferCount);

    compute->execute(createGridCellArrays, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  rearrangeParticles(particleCount);
}

void FluidSolver::allocateBoundingBoxes(const uint particleCount)
{
  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(workgroupSize[0] * workgroupCount[0]), false);
  }
}

void FluidSolver::allocatePositionBuffers(const uint particleCount)
{
  if (systemParticlePositionsCopy.size() < systemNonFluidParticleCount)
  {
    systemParticlePositionsCopy.resize(systemNonFluidParticleCount, false);
    systemParticleDifferentialCopy.resize(systemNonFluidParticleCount, false);
  }

  if (gridParticleCellIndex.size() < particleCount)
  {
    particlesCopy.resize(particleCount, false);
    particlesPredictedCopy.resize(particleCount, false);
    particleDifferentialCopy.resize(particleCount, false);
  }
}

void FluidSolver::allocateIndexBuffers(const uint particleCount)
{
  if (gridParticleCellIndex.size() < particleCount)
  {
    particlesDensity.resize(particleCount, false);
    particlesLambda.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (particlesLambda.size() < systemNonFluidParticleCount)
  {
    particlesLambda.resize(systemNonFluidParticleCount, false);
  }

  if (systemGridParticleCellIndex.size() < systemNonFluidParticleCount)
  {
    systemGridParticleCellIndex.resize(systemNonFluidParticleCount, false);
    systemGridCellParticleIndices.resize(systemNonFluidParticleCount, false);
    systemGridParticleSystemIndex.resize(systemNonFluidParticleCount, false);
  }

  const uint gridElements = gridSize * gridSize * gridSize;

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
    systemGridCellParticleCount.resize(systemNonFluidParticleCount, false);
  }
}

void FluidSolver::allocateBuffers(const uint particleCount)
{
  systemNonFluidParticleCount = systemParticleCount - particleCount;
  allocateBoundingBoxes(particleCount);
  allocatePositionBuffers(particleCount);
  allocateIndexBuffers(particleCount);
}

void FluidSolver::solve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  updateRadius();
  allocateBuffers(particleCount);

  for (int i=0; i<iterations; i++)
  {
    constructGrid();

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

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
      calculateDensity.setArgs(buffers, bufferCount);
      calculateDensity.setArg<ushort>(&gridSize, bufferCount);
      calculateDensity.setArg<ushort>(&gridSizeExp, bufferCount + 1);
      calculateDensity.setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(calculateDensity, workgroupSize, workgroupCount);
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
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<ushort>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<ushort>(&gridSizeExp, bufferCount + 1);
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

void FluidSolver::constructBoundaryGrid()
{
  if (!systemNonFluidParticleCount) return;

  const uint gridElements = gridSize * gridSize * gridSize;

  ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, systemGridCellParticleCount.device(), gridElements);

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount);

    ComputeMemory* buffers[] = {
      systemGridCellParticleCount.device(),
      systemGridParticleCellIndex.device(),
      systemParticlePositions,
      systemSettings,
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS].setArg<uint>(&systemNonFluidParticleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS].setArg<ushort>(&gridSize, bufferCount + 1);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS].setArg<ushort>(&gridSizeExp, bufferCount + 2);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  systemGridCellParticleCount.syncHost();
  systemGridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, systemGridCellParticleOffsets.device(), systemGridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_SOLVER
  systemGridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount, compute->simdSize());

    ComputeMemory* buffers[] = {
      systemGridCellParticleIndices.device(),
      systemGridCellParticleOffsets.device(),
      systemGridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_ARRAYS].setArg<uint>(&systemNonFluidParticleCount, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_SYSTEM_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  systemGridCellParticleIndices.syncHost();
  systemGridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(0)->copyBuffer(compute, systemGridParticleCellIndex.device(), particlesLambda.device(), 0, 0, sizeof(uint)*systemNonFluidParticleCount);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount);

    ComputeMemory* buffers[] = {
      systemParticlePositionsCopy.device(),
      systemParticlePositions,
      systemParticleDifferentialCopy.device(),
      systemParticleDifferential,
      systemGridParticleCellIndex.device(),
      particlesLambda.device(),
      systemGridCellParticleIndices.device(),
      systemGridParticleSystemIndex.device(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_REORDER].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_SYSTEM_REORDER].setArg<uint>(&systemNonFluidParticleCount, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_SYSTEM_REORDER], workgroupSize, workgroupCount);
  }
}

void FluidSolver::calculateParticleCouplingData(DeviceArray<ParticleCouplingData> &particleCouplingData,
                                                DeviceArray<ParticleStruct> &particles,
                                                DeviceArray<ParticleCollisionData> &particleCollisionData,
                                                uint particleCount)
{
  updateRadius();
  allocateBuffers(particleCount);

  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleGroupBoundingBoxes.device(),
      particles.device(),
      particleCollisionData.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    createBoundingBoxesCoupling.setArgs(buffers, bufferCount);
    createBoundingBoxesCoupling.setArg<uint>(&particleCount, bufferCount);

    compute->execute(createBoundingBoxesCoupling, workgroupSize, workgroupCount);

#ifdef DEBUG_FLUID_SOLVER
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

    // find bounding box for the simulation space
    ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(workgroupSize[0] * workgroupCount[0]));
  }

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
      particles.device(),
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    createGridCellHistogram.setArgs(buffers, bufferCount);
    createGridCellHistogram.setArg<uint>(&particleCount, bufferCount);
    createGridCellHistogram.setArg<ushort>(&gridSize, bufferCount + 1);
    createGridCellHistogram.setArg<ushort>(&gridSizeExp, bufferCount + 2);

    compute->execute(createGridCellHistogram, workgroupSize, workgroupCount);
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
    createGridCellArrays.setArgs(buffers, bufferCount);
    createGridCellArrays.setArg<uint>(&particleCount, bufferCount);

    compute->execute(createGridCellArrays, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particlesCopy.device(),
      particles.device(),
      particlesLambda.device(),
      gridParticleCellIndex.device(),
      gridCellParticleIndices.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    reorderCouplingParticles.setArgs(buffers, bufferCount);
    reorderCouplingParticles.setArg<uint>(&particleCount, bufferCount);

    compute->execute(reorderCouplingParticles, workgroupSize, workgroupCount);
  }

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particleCouplingData.device(),
      gridCellParticleOffsets.device(),
      particlesLambda.device(),
      gridCellParticleIndices.device(),
      particlesCopy.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    calculateCouplingData.setArgs(buffers, bufferCount);
    calculateCouplingData.setArg<FluidSolverData>(&(entitySharedData.host()->at(0).fluidSolverData), bufferCount);
    calculateCouplingData.setArg<ushort>(&gridSize, bufferCount + 1);
    calculateCouplingData.setArg<ushort>(&gridSizeExp, bufferCount + 2);
    calculateCouplingData.setArg<uint>(&particleCount, bufferCount + 3);

    compute->execute(calculateCouplingData, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  particleCouplingData.syncHost();
  compute->sync();
#endif
}
