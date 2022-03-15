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
  : Solver(compute, allocator), EntitySolverType(compute, allocator, SOLVER_FLUID), UniformGridCollisionSolver(compute, allocator), boundaryGridCellParticleOffsets(boundaryGridCellParticleCount), systemParticleCount(0), systemNonFluidParticleCount(0)
{
  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

  if (!noCreate)
  {
    create(compute);
  }

  particlesDensity.create(compute, solverHeap);
  particlesLambda.create(compute, solverHeap);
  boundaryGridParticleCellIndex.create(compute, solverHeap);
  boundaryGridCellParticleIndices.create(compute, solverHeap);
  boundaryGridCellParticleCount.create(compute, solverHeap);
  boundaryParticlePositions.create(compute, solverHeap);
  boundaryParticleDifferential.create(compute, solverHeap);
  boundaryGridParticleSystemIndex.create(compute, solverHeap);
  boundaryParticleCouplingData.create(compute, solverHeap);
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
  createBoundaryGridCellHistogram = programs[0].createKernel("createBoundaryGridCellHistogram");
  createBoundaryGridCellArrays    = programs[0].createKernel("createBoundaryGridCellArrays");
  reorderBoundaryParticles        = programs[0].createKernel("reorderBoundaryParticles");

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

float FluidSolver::calculateGradientConstant(const ParticleSharedData& entitySharedData)const
{
  // TODO: Find proper way of calculating prototype neighborhood, right now it has to be made dense for fluid to work
  const float scale = 8;
  // create prototype neighbourhood
  int kernelFactor = scale * ceil(entitySharedData.fluidSolverData.fluidKernelRadius / entitySharedData.sharedRadius);
  vector<Real3> neighbourParticles;
  neighbourParticles.reserve((2*kernelFactor+1) * (2*kernelFactor+1) * (2*kernelFactor+1));
  for (int i=-kernelFactor; i<=kernelFactor; i++)
  {
    for (int j=-kernelFactor; j<=kernelFactor; j++)
    {
      for (int k=-kernelFactor; k<=kernelFactor; k++)
      {
        neighbourParticles.push_back(Real3(i, j, k) / scale);
      }
    }
  }

  // calculate gradient magnitude contribution from neighbours
  float sumGradientMagnitude = 0.f;
  for (const auto& position : neighbourParticles)
  {
    const Real3 collisionVector = position * 2.f * entitySharedData.sharedRadius;
    if (collisionVector.length() < entitySharedData.fluidSolverData.fluidKernelRadius)
    {
      const Real3 gradient = collisionVector * spikyFunctionGradientVariable(collisionVector.length(), entitySharedData.fluidSolverData.fluidKernelRadius);
      sumGradientMagnitude += gradient.lengthSq();
    }
  }

  return sumGradientMagnitude * mSqr(spikyFunctionConstant(entitySharedData.fluidSolverData.fluidKernelRadius));
}

void FluidSolver::rearrangeParticles(uint particleCount)
{
  size_t workgroupSize[3], workgroupCount[3];

  uint multiplier = 1;
#ifdef FLUID_SOLVER_SORTED_REARRANGE
  multiplier = FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER;
#endif

  compute->configureSize(workgroupSize, workgroupCount, mAlignBy(particleCount, multiplier), compute->maxThreadsPerGroup() / multiplier);

  ComputeUtil::get(ComputeUtil::getUInt4Util(compute))->copyBuffer(compute, particles.device(), particlesCopy.device(), 0, 0, sizeof(ParticleStruct)*particleCount);
  ComputeUtil::get(ComputeUtil::getUInt4Util(compute))->copyBuffer(compute, particlesPredicted.device(), particlesPredictedCopy.device(), 0, 0, sizeof(ParticleStruct)*particleCount);
  ComputeUtil::get(ComputeUtil::getUInt4Util(compute))->copyBuffer(compute, particleDifferential.device(), particleDifferentialCopy.device(), 0, 0, sizeof(ParticleDifferential)*particleCount);
  ComputeUtil::get(ComputeUtil::getUIntUtil(compute))->copyBuffer(compute, gridParticleCellIndex.device(), particlesLambda.device(), 0, 0, sizeof(uint)*particleCount);

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
  reorderFluidParticles.setSharedMemArg(sizeof(uint)*4*(workgroupSize[0]*workgroupSize[1]*workgroupSize[2]*multiplier+1), bufferCount + 1);
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

  constructBoundaryGrid();

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
  if (boundaryParticlePositions.size() < mMax(systemNonFluidParticleCount, (uint)1))
  {
    boundaryParticlePositions.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
    boundaryParticleDifferential.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
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

  if (boundaryGridParticleCellIndex.size() < mMax(systemNonFluidParticleCount, (uint)1))
  {
    boundaryGridParticleCellIndex.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
    boundaryGridCellParticleIndices.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
    boundaryGridParticleSystemIndex.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
    boundaryParticleCouplingData.resize(mMax(systemNonFluidParticleCount, (uint)1), false);
  }

  const uint gridElements = gridSize * gridSize * gridSize;

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
    boundaryGridCellParticleCount.resize(gridElements, false);
    // do this atleast once, so its all 0s when there are no boundary particles
    ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, boundaryGridCellParticleCount.device(), gridElements);
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
        boundaryGridCellParticleOffsets.device(),
        boundaryParticleCouplingData.device(),
        boundaryParticlePositions.device(),
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

  ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, boundaryGridCellParticleCount.device(), gridElements);

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount);

    ComputeMemory* buffers[] = {
      boundaryGridCellParticleCount.device(),
      boundaryGridParticleCellIndex.device(),
      systemParticlePositions,
      systemSettings,
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    createBoundaryGridCellHistogram.setArgs(buffers, bufferCount);
    createBoundaryGridCellHistogram.setArg<uint>(&systemNonFluidParticleCount, bufferCount);
    createBoundaryGridCellHistogram.setArg<ushort>(&gridSize, bufferCount + 1);
    createBoundaryGridCellHistogram.setArg<ushort>(&gridSizeExp, bufferCount + 2);

    compute->execute(createBoundaryGridCellHistogram, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  boundaryGridCellParticleCount.syncHost();
  boundaryGridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, boundaryGridCellParticleOffsets.device(), boundaryGridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_SOLVER
  boundaryGridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount, compute->simdSize());

    ComputeMemory* buffers[] = {
      boundaryGridCellParticleIndices.device(),
      boundaryGridCellParticleOffsets.device(),
      boundaryGridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    createBoundaryGridCellArrays.setArgs(buffers, bufferCount);
    createBoundaryGridCellArrays.setArg<uint>(&systemNonFluidParticleCount, bufferCount);

    compute->execute(createBoundaryGridCellArrays, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  boundaryGridCellParticleIndices.syncHost();
  boundaryGridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(ComputeUtil::getUIntUtil(compute))->copyBuffer(compute, boundaryGridParticleCellIndex.device(), particlesLambda.device(), 0, 0, sizeof(uint)*systemNonFluidParticleCount);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, systemNonFluidParticleCount);

    ComputeMemory* buffers[] = {
      boundaryParticlePositions.device(),
      systemParticlePositions,
      boundaryParticleDifferential.device(),
      systemParticleDifferential,
      boundaryParticleCouplingData.device(),
      systemParticleCouplingData,
      boundaryGridParticleCellIndex.device(),
      particlesLambda.device(),
      boundaryGridCellParticleIndices.device(),
      boundaryGridParticleSystemIndex.device(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    reorderBoundaryParticles.setArgs(buffers, bufferCount);
    reorderBoundaryParticles.setArg<uint>(&systemNonFluidParticleCount, bufferCount);

    compute->execute(reorderBoundaryParticles, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_SOLVER
  boundaryParticleCouplingData.syncHost();
  compute->sync();
#endif
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
      particlesCopy.device(),
      systemBoundingBox.device(),
      invMaxRadius.device()
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
