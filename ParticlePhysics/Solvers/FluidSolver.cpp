#include "FluidSolver.h"

#define DEBUG_FLUID_SOLVER

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

  create(compute);

#ifdef DEBUG_FLUID_SOLVER
  particlesDensity.create(compute, solverHeap, true);
  particlesLambda.create(compute, solverHeap, true);
#else
  particlesDensity.create(compute, solverHeap);
  particlesLambda.create(compute, solverHeap);
#endif
}

void FluidSolver::create(ComputeInterface* compute)
{
  const vector<string> oldType = {"SOLVER_FLUID"};
  const vector<string> newType = {to_string(SOLVER_FLUID)};

  includeFiles.push_back("UniformGridCollisionSolver.shader");

  registerShader(compute, "FluidSolver.shader", &oldType, &newType);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("calculateDensity"));
  kernels.push_back(programs[0].createKernel("calculateForces"));

  maxRadius.host()->push_back(-1.f);

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

  float kernelRadius = 0.f;
  for (auto& esd : *entitySharedData.host())
  {
    kernelRadius = max(esd.fluidKernelRadius, kernelRadius);
  }

  if (kernelRadius != maxRadius.host()->at(0))
  {
    (*maxRadius.host())[0] = kernelRadius;
    maxRadius.syncDevice();
  }

  uint nodeBatchSize = 8;
  uint nodeBatchCount = (particleCount + nodeBatchSize - 1) / nodeBatchSize;

  const uint gridElements = gridSize * gridSize * gridSize;

//  if (particleGroupBoundingBoxes.size() < nodeBatchCount)
//  {
//    particleGroupBoundingBoxes.resize(nodeBatchCount, false);
//  }

  // 4 byte aligned for indirect dispatch
  if (gridCompactCellCount.size() == 0)
  {
    gridCompactCellCount.resize(4, false);
  }

  if (gridParticleCellIndex.size() < particleCount)
  {
    UniformGridCollisionSolver::particlesTemp.resize(particleCount, false);
    particlesDensity.resize(particleCount, false);
    particlesLambda.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (gridCompactCellIndices.size() < gridElements)
  {
    gridCompactCellIndices.resize(gridElements, false);
    gridCellParticleCount.resize(gridElements, false);
    gridCellParticleOffsets.resize(gridElements, false);
  }

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
  ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), nodeBatchCount);

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
      maxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&particleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);

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

  ComputeUtil::get(gridComputeUtilId)->compactSparseArray(compute, gridCompactCellCount.device(), gridCompactCellIndices.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_SOLVER
  gridCompactCellIndices.syncHost();
#endif

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

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

  compute->copyBuffer(particlesPredicted.device(), UniformGridCollisionSolver::particlesTemp.device(), 0, 0, sizeof(ParticleStruct)*particleCount);

  {
    size_t workgroupSize[3] = {1, 1, 1};
    workgroupSize[0] = this->compute->maxThreadsPerGroup();

    ComputeMemory* buffers[] = {
      particlesDensity.device(),
      particlesLambda.device(),
      gridCompactCellIndices.device(),
      gridCellParticleOffsets.device(),
      gridCellParticleCount.device(),
      gridCellParticleIndices.device(),
      UniformGridCollisionSolver::particlesTemp.device(),
      entitySharedData.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY].setArg<uint>(&gridSize, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_CALC_DENSITY], workgroupSize, gridCompactCellCount.device(), 0);
  }

#ifdef DEBUG_FLUID_SOLVER
  particlesDensity.syncHost();
  particlesLambda.syncHost();
  gridCompactCellCount.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3] = {1, 1, 1};
    workgroupSize[0] = this->compute->maxThreadsPerGroup();

    ComputeMemory* buffers[] = {
      particlesPredicted.device(),
      particlesDensity.device(),
      particlesLambda.device(),
      particleDifferential.device(),
      gridCompactCellIndices.device(),
      gridCellParticleOffsets.device(),
      gridCellParticleCount.device(),
      gridCellParticleIndices.device(),
      UniformGridCollisionSolver::particlesTemp.device(),
      entitySharedData.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_CALC_FORCES].setArg<uint>(&gridSize, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_CALC_FORCES], workgroupSize, gridCompactCellCount.device(), 0);
  }
#ifdef DEBUG_FLUID_SOLVER
  particlesPredicted.syncHost();
  particlesDensity.syncHost();
  particlesLambda.syncHost();
  gridCompactCellCount.syncHost();
  compute->sync();
#endif
}

void FluidSolver::update()
{
  if (!updates.size()) return;

  EntitySolver::update();
}
