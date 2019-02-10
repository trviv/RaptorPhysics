#include "CollisionSolver.h"

#define COLLISION_SOLVER_CELL_COUNTS      0
#define COLLISION_SOLVER_CELL_ARRAYS      1
#define COLLISION_SOLVER_APPLY_COLLISIONS 2
#define COLLISION_SOLVER_KERNEL_BOUNDARY  3
#define COLLISION_LBVH_SOLVER_MORTON_CODE 4

#define COLLISION_DEBUG_GRID

uint collisionSolverComputeUtilId;

CollisionSolver::~CollisionSolver()
{
  delete solverHeap;
}

void CollisionSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;
  gridSize = 32;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");

  registerShader(compute, "CollisionSolver.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("applyCollisions"));
  kernels.push_back(programs[0].createKernel("boundaryCollisionKernel"));
  kernels.push_back(programs[0].createKernel("assignMortonCodeKernel"));

  solverHeap = new ComputeHeap(compute);

  // allocate for 3*grid size + 2*number of max particles
  solverHeap->create((3 * gridSize * gridSize + 3 * 4 * 1024 * 1024)*sizeof(uint));

  gridCompactCellCount.create(compute, solverHeap, true);
#ifdef COLLISION_DEBUG_GRID
  gridCompactCellIndices.create(compute, solverHeap, true);
  gridParticleCellIndex.create(compute, solverHeap, true);
  gridCellParticleCount.create(compute, solverHeap, true);
  gridCellParticleOffsets.create(compute, solverHeap, true);
  gridCellParticleIndices.create(compute, solverHeap, true);
#else
  gridCompactCellIndices.create(compute, solverHeap);
  gridParticleCellIndex.create(compute, solverHeap);
  gridCellParticleCount.create(compute, solverHeap);
  gridCellParticleOffsets.create(compute, solverHeap);
  gridCellParticleIndices.create(compute, solverHeap);
#endif

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";

  collisionSolverComputeUtilId = ComputeUtil::create(compute, utilSetting);
}

void CollisionSolver::build(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  const uint gridElements = gridSize * gridSize;
  const uint gridSizeInBytes = gridElements * sizeof(uint);

  if (gridCompactCellCount.size() == 0)
  {
    gridCompactCellCount.resize(1, false);
  }

  if (gridCompactCellIndices.size() < gridElements)
  {
    gridCompactCellIndices.resize(gridElements, false);
  }

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
  }

  if (gridCellParticleOffsets.size() < gridElements)
  {
    gridCellParticleOffsets.resize(gridElements, false);
  }

  if (gridParticleCellIndex.size() < instanceNodeCount)
  {
    gridParticleCellIndex.resize(instanceNodeCount, false);
  }

  if (gridCellParticleIndices.size() < instanceNodeCount)
  {
    gridCellParticleIndices.resize(instanceNodeCount, false);
    return;
  }

  uint zero = 0;

  // clear index offset buffer
  compute->setBuffer(gridCellParticleCount.device(), 0, gridSizeInBytes, &zero, sizeof(uint));

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleCount.device(),
      gridParticleCellIndex.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);

    compute->execute(kernels[COLLISION_SOLVER_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef COLLISION_DEBUG_GRID
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(collisionSolverComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef COLLISION_DEBUG_GRID
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(collisionSolverComputeUtilId)->compactSparseArray(compute, gridCompactCellCount.device(), gridCompactCellIndices.device(), gridCellParticleCount.device(), gridElements);

#ifdef COLLISION_DEBUG_GRID
  gridCompactCellIndices.syncHost();
#endif

  gridCompactCellCount.syncHost();
  compute->sync();

  { // put particle indices in cell array
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleIndices.device(),
      gridCellParticleOffsets.device(),
      gridParticleCellIndex.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[COLLISION_SOLVER_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[COLLISION_SOLVER_CELL_ARRAYS].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef COLLISION_DEBUG_GRID
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  { // apply collisions
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, (*gridCompactCellCount.host())[0]);

    ComputeMemory* buffers[] = {
      gridCellParticleIndices.device(),
      gridCellParticleCount.device(),
      gridCellParticleOffsets.device(),
      gridParticleCellIndex.device(),
      gridCompactCellIndices.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[COLLISION_SOLVER_APPLY_COLLISIONS].setArgs(buffers, bufferCount);
    kernels[COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&(*gridCompactCellCount.host())[0], bufferCount + 1);

    compute->execute(kernels[COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, workgroupCount);
  }
}

void CollisionSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  build(instanceNodeCount, globalOffsets);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
    globalOffsets
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArgs(buffers, bufferCount);
  kernels[COLLISION_SOLVER_KERNEL_BOUNDARY].setArg<uint>(&instanceNodeCount, bufferCount);

  compute->execute(kernels[COLLISION_SOLVER_KERNEL_BOUNDARY], workgroupSize, workgroupCount);
}