#include "CollisionSolver.h"

#define COLLISION_SOLVER_CELL_COUNTS      0
#define COLLISION_SOLVER_BUILD            1
#define COLLISION_SOLVER_KERNEL_BOUNDARY  2
#define COLLISION_LBVH_SOLVER_MORTON_CODE 3

#define COLLISION_DEBUG_GRID

uint collisionSolverComputeUtilId;

CollisionSolver::~CollisionSolver()
{
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
  kernels.push_back(programs[0].createKernel("createGridArrays"));
  kernels.push_back(programs[0].createKernel("boundaryCollisionKernel"));
  kernels.push_back(programs[0].createKernel("assignMortonCodeKernel"));

#ifdef COLLISION_DEBUG_GRID
  gridParticleIndices.create(compute, 0, true);
  gridCellParticleCount.create(compute, 0, true);
  gridCellParticleOffsets.create(compute, 0, true);
#else
  gridParticleIndices.create(compute);
  gridCellParticleCount.create(compute);
  gridCellParticleOffsets.create(compute);
#endif

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";

  collisionSolverComputeUtilId = ComputeUtil::create(compute, utilSetting);
}

void CollisionSolver::build(uint instanceNodeCount, ComputeMemory* globalOffsets)
{
  // resize index array if necessary
  if (gridParticleIndices.size() < instanceNodeCount)
  {
    gridParticleIndices.resize(instanceNodeCount, false);
  }

  // resize offset array if necessary
  if (gridCellParticleCount.size() < gridSize*gridSize)
  {
    gridCellParticleCount.resize(gridSize*gridSize, false);
  }

  // resize offset array if necessary
  if (gridCellParticleOffsets.size() < gridSize*gridSize)
  {
    gridCellParticleOffsets.resize(gridSize*gridSize, false);
    return;
  }

  uint zero = 0;

  // clear index offset buffer
  compute->setBuffer(gridCellParticleCount.device(), 0, gridSize*gridSize*sizeof(uint), &zero, sizeof(uint));

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
    gridCellParticleCount.device(),
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

#ifdef COLLISION_DEBUG_GRID
  gridCellParticleCount.syncHost();
#endif

  ComputeUtil::get(collisionSolverComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridSize*gridSize);

#ifdef COLLISION_DEBUG_GRID
  gridCellParticleOffsets.syncHost();
#endif

  /*size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg(allocator->getHeap(COMPUTE_HEAP_PARTICLE), 0);
  kernels[COLLISION_LBVH_SOLVER_MORTON_CODE].setArg<uint>(&instanceNodeCount, 1);
  compute->execute(kernels[COLLISION_LBVH_SOLVER_MORTON_CODE], workgroupSize, workgroupCount);*/
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