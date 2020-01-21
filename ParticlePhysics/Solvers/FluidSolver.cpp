#include "FluidSolver.h"

//#define DEBUG_FLUID_SOLVER

#define FLUID_COLLISION_SOLVER_CREATE_BOUNDING_BOX  0
#define FLUID_COLLISION_SOLVER_CELL_COUNTS          1
#define FLUID_COLLISION_SOLVER_CELL_ARRAYS          2
#define FLUID_COLLISION_SOLVER_CALC_DENSITY         3
#define FLUID_COLLISION_SOLVER_CALC_FORCES          4

static uint gridXABComputeUtilId;
static uint gridComputeUtilId;

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), EntitySolver<uint, real, Real3>(compute, allocator, SOLVER_FLUID), UniformGridCollisionSolver(compute, allocator)
{
  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

  create(compute);

#ifdef DEBUG_FLUID_SOLVER
  particlesDensity.create(compute, solverHeap, true);
  particlesLambda.create(compute, solverHeap, true);
  particlesTemp[0].create(compute, solverHeap, true);
  particlesTemp[1].create(compute, solverHeap, true);
#else
  particlesDensity.create(compute, solverHeap);
  particlesLambda.create(compute, solverHeap);
  particlesTemp[0].create(compute, solverHeap);
  particlesTemp[1].create(compute, solverHeap);
#endif
}

void FluidSolver::create(ComputeInterface* compute)
{
  includeFiles.push_back("UniformGridCollisionSolver.shader");

  registerShader(compute, "FluidSolver.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("calculateDensity"));
  kernels.push_back(programs[0].createKernel("calculateForces"));

  invMaxRadius.host()->push_back(-1.f);

  // create utility classes
  const vector<string> utilInclude = {"ParticleStruct.h"};

  map<ComputeUtilKey, string> lbvhXABSetting;
  lbvhXABSetting[ComputeUtilBatchSize] = "1";
  lbvhXABSetting[ComputeUtilStructType] = "XAB";
  lbvhXABSetting[ComputeUtilStructSize] = "32";
  lbvhXABSetting[ComputeUtilOnlyReduce] = "1";
  lbvhXABSetting[ComputeUtilCustomAddFunction] = "mergeXAB";
  lbvhXABSetting[ComputeUtilCustomDivFunction] = "divXAB";
  lbvhXABSetting[ComputeUtilCustomCopyFunction] = "copyXAB";
  lbvhXABSetting[ComputeUtilCustomClearFunction] = "clearXAB";
  lbvhXABSetting[ComputeUtilCustomReduceFunction] = "reduceXAB";
  lbvhXABSetting[ComputeUtilSkipParallelPrimitives] = "1";
  gridXABComputeUtilId = ComputeUtil::create(compute, lbvhXABSetting, &utilInclude);

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";

  gridComputeUtilId = ComputeUtil::create(compute, utilSetting);
}

void FluidSolver::solve()
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  // update kernel radius
  float kernelRadius = 0.f;
  for (auto& esd : *entitySharedData.host())
  {
    kernelRadius = max(esd.fluidKernelRadius, kernelRadius);
  }

  if (1.f/kernelRadius != invMaxRadius.host()->at(0))
  {
    (*invMaxRadius.host())[0] = 1.f/kernelRadius;
    invMaxRadius.syncDevice();
  }

  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(workgroupSize[0] * workgroupCount[0]), false);
  }

  // 4 byte aligned for indirect dispatch
  if (gridCompactCellCount.size() == 0)
  {
    gridCompactCellCount.resize(4, false);
  }

  if (gridParticleCellIndex.size() < particleCount)
  {
    UniformGridCollisionSolver::particlesBufferTemp.resize(particleCount, false);
    particlesTemp[0].resize(particleCount, false);
    particlesTemp[1].resize(particleCount, false);
    particlesDensity.resize(particleCount, false);
    particlesLambda.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (gridCompactCellIndices.size() < gridElements)
  {
    gridCompactCellIndices.resize(gridElements, false);
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

    ComputeUtil::get(gridComputeUtilId)->compactSparseArray(compute, gridCompactCellCount.device(), gridCompactCellIndices.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_SOLVER
    gridCompactCellIndices.syncHost();
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

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, gridCompactCellCount.device(), 0);
    }

#ifdef DEBUG_FLUID_SOLVER
    gridCellParticleIndices.syncHost();
    gridCellParticleOffsets.syncHost();
    compute->sync();
  #endif

    ComputeUtil::get(0)->copyBuffer(compute, particlesPredicted.device(), UniformGridCollisionSolver::particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*particleCount);
    ComputeUtil::get(0)->copyBuffer(compute, particleDifferential.device(), particlesTemp[0].device(), 0, 0, sizeof(ParticleDifferential)*particleCount);
    ComputeUtil::get(0)->copyBuffer(compute, particles.device(), particlesTemp[1].device(), 0, 0, sizeof(ParticleStruct)*particleCount);

    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    {
      ComputeMemory* buffers[] = {
        particlesDensity.device(),
        particlesLambda.device(),
        gridCompactCellIndices.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
        gridParticleCellIndex.device(),
        UniformGridCollisionSolver::particlesBufferTemp.device(),
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
    particlesLambda.syncHost();
    gridCompactCellCount.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particles.device(),
        particlesTemp[1].device(),
        particlesPredicted.device(),
        particlesDensity.device(),
        particlesLambda.device(),
        particleDifferential.device(),
        particlesTemp[0].device(),
        gridCompactCellIndices.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
        gridParticleCellIndex.device(),
        UniformGridCollisionSolver::particlesBufferTemp.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_CALC_FORCES], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_SOLVER
    particlesPredicted.syncHost();
    particlesDensity.syncHost();
    particlesLambda.syncHost();
    gridCompactCellCount.syncHost();
    particleDifferential.syncHost();
    particlesTemp[0].syncHost();
    compute->sync();
#endif
  }
}

void FluidSolver::update()
{
  if (!updates.size()) return;

  EntitySolver::update();
}
