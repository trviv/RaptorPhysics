#include "UniformGridCollisionSolver.h"

#define UNIFORM_GRID_COLLISION_SOLVER_CELL_COUNTS       0
#define UNIFORM_GRID_COLLISION_SOLVER_CELL_ARRAYS       1
#define UNIFORM_GRID_COLLISION_SOLVER_APPLY_COLLISIONS  2
#define UNIFORM_GRID_COLLISION_SOLVER_KERNEL_BOUNDARY   3

//#define DEBUG_COLLISION_UNIFORM_GRID

uint uniformGridCollisionSolverComputeUtilId;

UniformGridCollisionSolver::~UniformGridCollisionSolver()
{
}

void UniformGridCollisionSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;
  gridSize = 64;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");

  registerShader(compute, "UniformGridCollisionSolver.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("applyCollisions"));
  kernels.push_back(programs[0].createKernel("boundaryCollisionKernel"));

  solverHeap = new ComputeHeap(compute);

  // allocate for 3*grid size + 2*number of max particles + 1* max particles particle structure
  solverHeap->create((3 * gridSize * gridSize * gridSize + 2 * (1 * 1024 * 1024) + 4 * (1 * 1024 * 1024)) * sizeof(uint));

  gridCompactCellCount.create(compute, solverHeap, true);
#ifdef DEBUG_COLLISION_UNIFORM_GRID
  gridCompactCellIndices.create(compute, solverHeap, true);
  gridParticleCellIndex.create(compute, solverHeap, true);
  gridCellParticleCount.create(compute, solverHeap, true);
  gridCellParticleOffsets.create(compute, solverHeap, true);
  gridCellParticleIndices.create(compute, solverHeap, true);
  particlesTemp.create(compute, solverHeap, true);
#else
  gridCompactCellIndices.create(compute, solverHeap);
  gridParticleCellIndex.create(compute, solverHeap);
  gridCellParticleCount.create(compute, solverHeap);
  gridCellParticleOffsets.create(compute, solverHeap);
  gridCellParticleIndices.create(compute, solverHeap);
  particlesTemp.create(compute, solverHeap);
#endif

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";

  uniformGridCollisionSolverComputeUtilId = ComputeUtil::create(compute, utilSetting);
}

void UniformGridCollisionSolver::build(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  const uint gridElements = gridSize * gridSize * gridSize;

  if (gridCompactCellCount.size() == 0)
  {
    gridCompactCellCount.resize(1, false);
  }

  if (gridCompactCellIndices.size() < gridElements)
  {
    gridCompactCellIndices.resize(gridElements, false);
    gridCellParticleCount.resize(gridElements, false);
    gridCellParticleOffsets.resize(gridElements, false);
  }

  if (gridParticleCellIndex.size() < instanceNodeCount)
  {
    gridParticleCellIndex.resize(instanceNodeCount, false);
    gridCellParticleIndices.resize(instanceNodeCount, false);
    particlesTemp.resize(instanceNodeCount, false);
  }

  // clear index offset buffer
  ComputeUtil::get(uniformGridCollisionSolverComputeUtilId)->clearIntegerBuffer(compute, gridCellParticleCount.device(), gridElements);
  compute->copyBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(), particlesTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);

  { // get count for each grid cell
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      gridCellParticleCount.device(),
      gridParticleCellIndex.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);

    compute->execute(kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_COLLISION_UNIFORM_GRID
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(uniformGridCollisionSolverComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_COLLISION_UNIFORM_GRID
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(uniformGridCollisionSolverComputeUtilId)->compactSparseArray(compute, gridCompactCellCount.device(), gridCompactCellIndices.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_COLLISION_UNIFORM_GRID
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
    kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_ARRAYS].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[UNIFORM_GRID_COLLISION_SOLVER_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_COLLISION_UNIFORM_GRID
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  if ((*gridCompactCellCount.host())[0])
  { // apply collisions
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, (*gridCompactCellCount.host())[0] * 256);

    ComputeMemory* buffers[] = {
      gridCompactCellIndices.device(),
      gridCellParticleOffsets.device(),
      gridCellParticleCount.device(),
      gridCellParticleIndices.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      particlesTemp.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArgs(buffers, bufferCount);
    kernels[UNIFORM_GRID_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&(*gridCompactCellCount.host())[0], bufferCount);

    compute->execute(kernels[UNIFORM_GRID_COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_COLLISION_UNIFORM_GRID
  compute->sync();
#endif
}

void UniformGridCollisionSolver::solve(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  build(instanceNodeCount, globalOffsets);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
    globalOffsets
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[UNIFORM_GRID_COLLISION_SOLVER_KERNEL_BOUNDARY].setArgs(buffers, bufferCount);
  kernels[UNIFORM_GRID_COLLISION_SOLVER_KERNEL_BOUNDARY].setArg<uint>(&instanceNodeCount, bufferCount);

  compute->execute(kernels[UNIFORM_GRID_COLLISION_SOLVER_KERNEL_BOUNDARY], workgroupSize, workgroupCount);
}
