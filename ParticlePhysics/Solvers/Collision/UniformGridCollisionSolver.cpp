#include "UniformGridCollisionSolver.h"

//#define DEBUG_GRID_SOLVER

#define GRID_COLLISION_SOLVER_GET_MAX_RADIUS      0
#define GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX 1
#define GRID_COLLISION_SOLVER_CELL_COUNTS         2
#define GRID_COLLISION_SOLVER_CELL_ARRAYS         3
#define GRID_COLLISION_SOLVER_APPLY_COLLISIONS    4

static uint gridXABComputeUtilId;
static uint gridComputeUtilId;
static uint gridGetSystemRadiusUtilId;

UniformGridCollisionSolver::UniformGridCollisionSolver(ComputeInterface* compute, SharedAllocator* allocator) :
  Solver(compute, allocator), CollisionSolver(compute, allocator)
{
  gridSize = 64;

  solverHeap = new ComputeHeap(compute);

  // allocate for 3*grid size + 2*number of max particles + 1* max particles particle structure
  solverHeap->create((3 * gridSize * gridSize * gridSize + 2 * (1 * 1024 * 1024) + 8 * (1 * 1024 * 1024)) * sizeof(uint) + (1 * 1024 * 1024) * sizeof(XAB));

  gridCompactCellCount.create(compute, solverHeap, true);
#ifdef DEBUG_GRID_SOLVER
  gridCompactCellIndices.create(compute, solverHeap, true);
  gridParticleCellIndex.create(compute, solverHeap, true);
  gridCellParticleCount.create(compute, solverHeap, true);
  gridCellParticleOffsets.create(compute, solverHeap, true);
  gridCellParticleIndices.create(compute, solverHeap, true);
  particlesPredictedTemp.create(compute, solverHeap, true);
  systemBoundingBox.create(compute, NULL, true);
#else
  gridCompactCellIndices.create(compute, solverHeap);
  gridParticleCellIndex.create(compute, solverHeap);
  gridCellParticleCount.create(compute, solverHeap);
  gridCellParticleOffsets.create(compute, solverHeap);
  gridCellParticleIndices.create(compute, solverHeap);
  particlesPredictedTemp.create(compute, solverHeap);
  systemBoundingBox.create(compute, NULL);
#endif

  particleGroupBoundingBoxes.create(compute, solverHeap, true);

  // allocate space for fixed sized data
  empty.create(compute, NULL, false);
  empty.resize(1, false);

  systemBoundingBox.resize(1, false);

  maxRadius.create(compute, NULL, true);
  maxRadius.resize(1, false);
}

UniformGridCollisionSolver::~UniformGridCollisionSolver()
{
}

DeviceArray<XAB>* UniformGridCollisionSolver::getBoundingBoxes()
{
  return &particleGroupBoundingBoxes;
}

void UniformGridCollisionSolver::init()
{
  const vector<string> oldType = {"COLLISION_SOLVER_USE_SYSTEM_OFFSETS"};
  const vector<string> newType = {""};

  registerShader(compute, "UniformGridCollisionSolver.shader", &oldType, &newType);

  kernels.push_back(programs[0].createKernel("getSystemMaxRadius"));
  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("applyCollisions"));

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

  utilSetting.clear();
  utilSetting[ComputeUtilOnlyReduce] = "1";
  utilSetting[ComputeUtilStructType] = "float";
  utilSetting[ComputeUtilCustomAddFunction] = "mergeFloat";
  utilSetting[ComputeUtilCustomReduceFunction] = "reduceFloat";
  utilSetting[ComputeUtilSkipParallelPrimitives] = "1";

  gridGetSystemRadiusUtilId = ComputeUtil::create(compute, utilSetting, &utilInclude);
}

void UniformGridCollisionSolver::build(uint instanceNodeCount, ComputeMemory* systemSettings)
{
  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(instanceNodeCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
  {
    particleGroupBoundingBoxes.resize(workgroupSize[0] * workgroupCount[0], false);
  }

  // 4 byte aligned for indirect dispatch
  if (gridCompactCellCount.size() == 0)
  {
    gridCompactCellCount.resize(4, false);
  }

  if (gridParticleCellIndex.size() < instanceNodeCount)
  {
    particlesPredictedTemp.resize(instanceNodeCount, false);
    gridParticleCellIndex.resize(instanceNodeCount, false);
    gridCellParticleIndices.resize(instanceNodeCount, false);
  }

  if (gridCompactCellIndices.size() < gridElements)
  {
    gridCompactCellIndices.resize(gridElements, false);
    gridCellParticleCount.resize(gridElements, false);
    gridCellParticleOffsets.resize(gridElements, false);
  }

  // TODO: Make a flag so that this is only done when needed
  if (maxRadius.host()->size() == 0)
  {
    uint nodeBatchCount = workgroupSize[0] * workgroupCount[0];

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particlesPredictedTemp.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_GET_MAX_RADIUS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_GET_MAX_RADIUS].setArg<uint>(&nodeBatchCount, bufferCount);
    kernels[GRID_COLLISION_SOLVER_GET_MAX_RADIUS].setArg<uint>(&instanceNodeCount, bufferCount + 1);

    compute->execute(kernels[GRID_COLLISION_SOLVER_GET_MAX_RADIUS], workgroupSize, workgroupCount);

#ifdef DEBUG_GRID_SOLVER
    particlesPredictedTemp.syncHost();
    compute->sync();
#endif

    ComputeUtil::get(gridGetSystemRadiusUtilId)->sum1D(compute, maxRadius.device(), particlesPredictedTemp.device(), nodeBatchCount);

#ifdef DEBUG_GRID_SOLVER
    maxRadius.syncHost();
    compute->sync();
#endif
  }

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleGroupBoundingBoxes.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&nodeBatchCount, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&instanceNodeCount, bufferCount + 1);

    compute->execute(kernels[GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_GRID_SOLVER
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

  // find bounding box for the simulation space
  ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), workgroupSize[0] * workgroupCount[0]);

#ifdef DEBUG_GRID_SOLVER
  systemBoundingBox.syncHost();
  compute->sync();
#endif

  // clear index offset buffer
  ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, gridCellParticleCount.device(), gridElements);

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleCount.device(),
      gridParticleCellIndex.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      systemBoundingBox.device(),
      maxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);

    compute->execute(kernels[GRID_COLLISION_SOLVER_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_GRID_SOLVER
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_GRID_SOLVER
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(gridComputeUtilId)->compactSparseArray(compute, gridCompactCellCount.device(), gridCompactCellIndices.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_GRID_SOLVER
  gridCompactCellIndices.syncHost();
#endif

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleIndices.device(),
      gridCellParticleOffsets.device(),
      gridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, gridCompactCellCount.device(), 0);
  }

#ifdef DEBUG_GRID_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif
}

void UniformGridCollisionSolver::solve(uint instanceNodeCount, ComputeMemory* systemSettings)
{
  const int iterations = 1;

  ComputeMemory* second;

  for (int i=0; i<iterations; i++)
  {
    second = (i==(iterations-1)) ? empty.device() : allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get();

    build(instanceNodeCount, systemSettings);

    compute->copyBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(), particlesPredictedTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);

    size_t workgroupSize[3] = {1, 1, 1};
    workgroupSize[0] = this->compute->maxThreadsPerGroup();

    ComputeMemory* buffers[] = {
      gridCompactCellIndices.device(),
      gridCellParticleOffsets.device(),
      gridCellParticleCount.device(),
      gridCellParticleIndices.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      second,
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      particlesPredictedTemp.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&gridSize, bufferCount);
    uint stablizationPass = (second != empty.device());
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&stablizationPass, bufferCount + 1);

    compute->execute(kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, gridCompactCellCount.device(), 0);
  }
}
