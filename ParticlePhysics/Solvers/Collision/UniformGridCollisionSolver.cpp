#include "UniformGridCollisionSolver.h"

//#define DEBUG_GRID_SOLVER
//#define GRID_COLLISION_SOLVE_PAIR_ONCE
//#define GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
#define GRID_SOLVER_HASH_FUNCTION
//#define GRID_COLLISION_SOLVER_SCATTER_PARTICLES

#define GRID_COLLISION_SOLVER_GET_MAX_RADIUS      0
#define GRID_COLLISION_SOLVER_CREATE_BOUNDING_BOX 1
#define GRID_COLLISION_SOLVER_CELL_COUNTS         2
#define GRID_COLLISION_SOLVER_CELL_ARRAYS         3
#define GRID_COLLISION_SOLVER_APPLY_COLLISIONS    4
#define GRID_COLLISION_SOLVER_APPLY_DELTA         5

static uint gridXABComputeUtilId;
static uint gridComputeUtilId;
static uint gridGetSystemRadiusUtilId;

UniformGridCollisionSolver::UniformGridCollisionSolver(ComputeInterface* compute, SharedAllocator* allocator) :
  Solver(compute, allocator), CollisionSolver(compute, allocator), gridCellParticleOffsets(gridCellParticleCount)
{
  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

  solverHeap = new ComputeHeap(compute);

  const uint maxParticles = (uint)allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get()->getSize() / sizeof(ParticleStruct);

  // allocate for 3*grid size (for grid indices) + 2*number of max particles(for particle indices) + 2*max particles particle structure (for temp buffers)
#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
  solverHeap->create((gridSize * gridSize * gridSize + 2 * 8 * maxParticles + 4 * maxParticles) * sizeof(uint) + maxParticles * sizeof(XAB));
#else
  solverHeap->create((gridSize * gridSize * gridSize + 2 * maxParticles + 4 * maxParticles) * sizeof(uint) + maxParticles * sizeof(XAB));
#endif

#ifdef DEBUG_GRID_SOLVER
  gridParticleCellIndex.create(compute, solverHeap, true);
  gridCellParticleIndices.create(compute, solverHeap, true);
  particlesBufferTemp.create(compute, solverHeap, true);
#else
  gridParticleCellIndex.create(compute, solverHeap);
  gridCellParticleIndices.create(compute, solverHeap);
  particlesBufferTemp.create(compute, solverHeap);
#endif
  gridCellParticleCount.create(compute, solverHeap, true);

  particleGroupBoundingBoxes.create(compute, solverHeap, true);

  systemBoundingBox.create(compute, NULL, true);
  systemBoundingBox.resize(1, false);

  invMaxRadius.create(compute, NULL, true);
  invMaxRadius.resize(1, false);
}

UniformGridCollisionSolver::~UniformGridCollisionSolver()
{
}

void UniformGridCollisionSolver::init()
{
  vector<string> oldType = {"COLLISION_SOLVER_USE_SYSTEM_OFFSETS"};
  vector<string> newType = {""};

#ifdef GRID_COLLISION_SOLVE_PAIR_ONCE
  oldType.push_back("GRID_COLLISION_SOLVE_PAIR_ONCE");
  newType.push_back("");
#endif

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
  oldType.push_back("GRID_COLLISION_SOLVER_USE_SHARED_MEMORY");
  newType.push_back("");
#endif

#ifdef GRID_SOLVER_HASH_FUNCTION
  oldType.push_back("GRID_SOLVER_HASH_FUNCTION");
  newType.push_back("");
#endif

#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
  oldType.push_back("GRID_COLLISION_SOLVER_SCATTER_PARTICLES");
  newType.push_back("");
#endif

  registerShader(compute, "UniformGridCollisionSolver.shader", &oldType, &newType);

  kernels.push_back(programs[0].createKernel("getSystemMaxRadius"));
  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("applyCollisions"));
  kernels.push_back(programs[0].createKernel("applyDeltas"));

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

void UniformGridCollisionSolver::build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer)
{
  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(instanceNodeCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(workgroupSize[0] * workgroupCount[0]), false);
  }

#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
  if (gridParticleCellIndex.size() < instanceNodeCount * 8)
  {
    particlesBufferTemp.resize(instanceNodeCount, false);
    gridParticleCellIndex.resize(instanceNodeCount * 8, false);
    gridCellParticleIndices.resize(instanceNodeCount * 8, false);
  }
#else
  if (gridParticleCellIndex.size() < instanceNodeCount)
  {
    particlesBufferTemp.resize(instanceNodeCount, false);
    gridParticleCellIndex.resize(instanceNodeCount, false);
    gridCellParticleIndices.resize(instanceNodeCount, false);
  }
#endif

  if (gridCellParticleCount.size() < gridElements)
  {
    // add 3 to use it directly as an indirect dispatch buffer
    gridCellParticleCount.resize(gridElements + 3, false);
    uint ones[3] = {1, 1, 1};
    compute->copyFromHost(gridCellParticleCount.device(), sizeof(uint)*gridElements, sizeof(uint)*3, ones, true);
  }

  // TODO: Make a flag so that this is only done when needed
  if (invMaxRadius.host()->size() == 0 || invMaxRadius.host()->at(0) == 0.f)
  {
    uint nodeBatchCount = (uint)(workgroupSize[0] * workgroupCount[0]);

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particlesBufferTemp.device(),
      particleBuffer,
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
    particlesBufferTemp.syncHost();
    compute->sync();
#endif

    ComputeUtil::get(gridGetSystemRadiusUtilId)->sum1D(compute, invMaxRadius.device(), particlesBufferTemp.device(), nodeBatchCount);
    invMaxRadius.syncHost();
    compute->sync();
    if (invMaxRadius.host()->at(0) != 0.f)
    {
      invMaxRadius.host()->at(0) = 1.f/invMaxRadius.host()->at(0);
      invMaxRadius.syncDevice();
    }

#ifdef DEBUG_GRID_SOLVER
    compute->sync();
#endif
  }

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleGroupBoundingBoxes.device(),
      particleBuffer,
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
  ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(workgroupSize[0] * workgroupCount[0]));

#ifdef DEBUG_GRID_SOLVER
  systemBoundingBox.syncHost();
  compute->sync();
#endif

  // clear index offset buffer
  ComputeUtil::get(gridComputeUtilId)->clearBuffer(compute, gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_GRID_SOLVER
  gridCellParticleCount.syncHost();
  compute->sync();
#endif

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleCount.device(),
      gridParticleCellIndex.device(),
      particleBuffer,
#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      systemSettings,
#endif
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);
    kernels[GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSizeExp, bufferCount + 2);

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

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount, compute->simdSize());

    ComputeMemory* buffers[] = {
      gridCellParticleIndices.device(),
      gridCellParticleOffsets.device(),
      gridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[GRID_COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_GRID_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif
}

void UniformGridCollisionSolver::solve(uint instanceNodeCount, ComputeMemory* systemSettings)
{
  for (int i=0; i<iterations; i++)
  {
    uint stablizationPass = (i < (iterations-1));

    ComputeMemory* particleBuffer = stablizationPass ? allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get() : allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get();

    build(instanceNodeCount, systemSettings, particleBuffer);

#ifndef GRID_COLLISION_SOLVE_PAIR_ONCE
    if (stablizationPass)
    {
      ComputeUtil::get(0)->copyBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(), particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);
    }
    else
    {
      ComputeUtil::get(0)->copyBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(), particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);
    }
#else
    ComputeUtil::get(0)->clearBuffer(compute, particlesBufferTemp.device(), instanceNodeCount * sizeof(ParticleStruct)/sizeof(uint), 0);
#endif

    size_t workgroupSize[3] = {1, 1, 1};
    size_t workgroupCount[3];
    const uint maxWorkgroupSize = compute->simdSize();

    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount, maxWorkgroupSize);

    ComputeMemory* buffers[] = {
      gridCellParticleOffsets.device(),
      gridCellParticleIndices.device(),
      gridParticleCellIndex.device(),
#ifdef GRID_COLLISION_SOLVE_PAIR_ONCE
      particlesBufferTemp.device(),
#else
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
#endif
      allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
#ifdef GRID_COLLISION_SOLVE_PAIR_ONCE
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
#else
      particlesBufferTemp.device(),
#endif
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      systemSettings,
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArgs(buffers, bufferCount);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&gridSize, bufferCount);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&gridSizeExp, bufferCount + 1);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&stablizationPass, bufferCount + 2);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&instanceNodeCount, bufferCount + 3);
#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(CollisionSolverData)*maxWorkgroupSize, bufferCount + 4);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(ParticleCollisionData)*maxWorkgroupSize, bufferCount + 5);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(ParticleStruct)*maxWorkgroupSize, bufferCount + 6);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(ParticleStruct)*maxWorkgroupSize, bufferCount + 7);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(float)*4*maxWorkgroupSize, bufferCount + 8);
    kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setSharedMemArg(sizeof(ParticleDifferential)*maxWorkgroupSize, bufferCount + 9);
#endif

#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
    compute->execute(kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, gridCellParticleOffsets.device(), (gridSize * gridSize * gridSize - 1) * sizeof(uint));
#else
    compute->execute(kernels[GRID_COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, workgroupCount);
#endif

#ifdef GRID_COLLISION_SOLVE_PAIR_ONCE
    {
      compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

      ComputeMemory* buffers[] = {
        allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
        allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
        particlesBufferTemp.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[GRID_COLLISION_SOLVER_APPLY_DELTA].setArgs(buffers, bufferCount);
      kernels[GRID_COLLISION_SOLVER_APPLY_DELTA].setArg<uint>(&stablizationPass, bufferCount);
      kernels[GRID_COLLISION_SOLVER_APPLY_DELTA].setArg<uint>(&instanceNodeCount, bufferCount + 1);

      compute->execute(kernels[GRID_COLLISION_SOLVER_APPLY_DELTA], workgroupSize, workgroupCount);
    }
#endif
  }
}
