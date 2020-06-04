#include "LBVHSolver.h"

//#define DEBUG_LBVH_SOLVER

#define LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX 0
#define LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE  1
#define LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE  2
#define LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX   3
#define LBVH_COLLISION_SOLVER_APPLY_COLLISIONS    4

static uint lbvhXABComputeUtilId;
static uint lbvhSortComputeUtilId;

LBVHSolver::LBVHSolver(ComputeInterface* compute, SharedAllocator* allocator) :
  Solver(compute, allocator), CollisionSolver(compute, allocator)
{
  solverHeap = new ComputeHeap(compute);

#ifdef DEBUG_LBVH_SOLVER
  particlesBufferTemp.create(compute, NULL, true);
  treeInternalNodes.create(compute, NULL, true);
  particleLeafData.create(compute, NULL, true);
  particleLeafDataSorted.create(compute, NULL, true);
  visitedInternalNodes.create(compute, NULL, true);
  leafParentNodeIndices.create(compute, NULL, true);
  nodeParentNodeIndices.create(compute, NULL, true);
  particleBoundingBoxes.create(compute, NULL, true);
#else
  particlesBufferTemp.create(compute, NULL);
  treeInternalNodes.create(compute, NULL);
  particleLeafData.create(compute, NULL);
  particleLeafDataSorted.create(compute, NULL);
  visitedInternalNodes.create(compute, NULL);
  leafParentNodeIndices.create(compute, NULL);
  nodeParentNodeIndices.create(compute, NULL);
  particleBoundingBoxes.create(compute, NULL);
#endif
  particleGroupBoundingBoxes.create(compute, NULL, true);

  // allocate space for fixed sized data
  treeInternalNodeBoundingBoxes.create(compute, NULL, true);

  systemBoundingBox.create(compute, NULL, true);
  systemBoundingBox.resize(1, false);
}

LBVHSolver::~LBVHSolver()
{
}

void LBVHSolver::init()
{
  const vector<string> oldType = {"COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES", "COLLISION_SOLVER_USE_SYSTEM_OFFSETS"};
  const vector<string> newType = {"", ""};

  registerShader(compute, "LBVHSolver.shader", &oldType, &newType);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("assignMortonCode"));
  kernels.push_back(programs[0].createKernel("constructBinaryTree"));
  kernels.push_back(programs[0].createKernel("constructTreeBoundingBox"));
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
  lbvhXABComputeUtilId = ComputeUtil::create(compute, lbvhXABSetting, &utilInclude);

  map<ComputeUtilKey, string> lbvhSortSetting;
  lbvhSortSetting[ComputeUtilStructType] = "uint";
  lbvhSortSetting[ComputeUtilStructTypeIntegral] = "1";
  lbvhSortComputeUtilId = ComputeUtil::create(compute, lbvhSortSetting, &utilInclude);
}

void LBVHSolver::build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer)
{
  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(instanceNodeCount, nodeBatchSize);

  if (particleLeafData.size() != instanceNodeCount)
  {
    particlesBufferTemp.resize(instanceNodeCount, false);
    particleLeafData.resize(instanceNodeCount, false);
    particleLeafDataSorted.resize(instanceNodeCount, false);
    particleBoundingBoxes.resize(instanceNodeCount, false);
    visitedInternalNodes.resize(instanceNodeCount, false);
    leafParentNodeIndices.resize(instanceNodeCount, false);
    nodeParentNodeIndices.resize(instanceNodeCount, false);
    treeInternalNodes.resize(instanceNodeCount - 1, false);
    treeInternalNodeBoundingBoxes.resize(instanceNodeCount - 1, false);
  }

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeBatchCount);

    nodeBatchCount = (uint)(workgroupSize[0] * workgroupCount[0]);
    if (particleGroupBoundingBoxes.size() < workgroupSize[0] * workgroupCount[0])
    {
      particleGroupBoundingBoxes.resize((uint)(workgroupSize[0] * workgroupCount[0]), false);
    }

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      particleBoundingBoxes.device(),
      particleGroupBoundingBoxes.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&nodeBatchCount, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX].setArg<uint>(&instanceNodeCount, bufferCount + 1);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_CREATE_BOUNDING_BOX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  particleBoundingBoxes.syncHost();
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

  // find bounding box for the simulation space
  ComputeUtil::get(lbvhXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(workgroupSize[0] * workgroupCount[0]));

#ifdef DEBUG_LBVH_SOLVER
  systemBoundingBox.syncHost();
  compute->sync();
#endif

  {
    // assign morton code to the particle bounding boxes
    ComputeMemory* buffers[] = {
      particleLeafData.device(),
      systemBoundingBox.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_ASSIGN_MORTON_CODE], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  particleLeafData.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(lbvhSortComputeUtilId)->radixSort32Bit(compute, particleLeafDataSorted.device(), particleLeafData.device(), instanceNodeCount);

#ifdef DEBUG_LBVH_SOLVER
  particleLeafDataSorted.syncHost();
  compute->sync();
#endif

  {
    // create binary radix tree
    ComputeMemory* buffers[] = {
      treeInternalNodes.device(),
      visitedInternalNodes.device(),
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      particleLeafDataSorted.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_CREATE_BINARY_TREE], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  treeInternalNodes.syncHost();
  leafParentNodeIndices.syncHost();
  nodeParentNodeIndices.syncHost();
  compute->sync();
#endif

  {
    // calculate bounding boxes for the tree
    ComputeMemory* buffers[] = {
      treeInternalNodeBoundingBoxes.device(),
      visitedInternalNodes.device(),
      treeInternalNodes.device(),
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      particleBoundingBoxes.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX].setArg<uint>(&instanceNodeCount, bufferCount);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_TREE_BOUNDING_BOX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_LBVH_SOLVER
  treeInternalNodeBoundingBoxes.syncHost();
  visitedInternalNodes.syncHost();
  compute->sync();
#endif
}

void LBVHSolver::solve(uint instanceNodeCount, ComputeMemory* systemSettings)
{
  const int iterations = 1;

  for (int i=0; i<iterations; i++)
  {
    uint stablizationPass = (i < (iterations-1));

    ComputeMemory* particleBuffer = stablizationPass ? allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get() : allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get();

    build(instanceNodeCount, systemSettings, particleBuffer);

    if (stablizationPass)
    {
      compute->copyBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(), particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);
    }
    else
    {
      compute->copyBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(), particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*instanceNodeCount);
    }

    const uint batchesPerDispatch = 8;
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, mAlignBy(instanceNodeCount, batchesPerDispatch));

    // compute axis aligned bounding boxes for particles
    ComputeMemory* buffers[] = {
      visitedInternalNodes.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      particlesBufferTemp.device(),
      treeInternalNodes.device(),
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      treeInternalNodeBoundingBoxes.device(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      systemSettings
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[LBVH_COLLISION_SOLVER_APPLY_COLLISIONS].setArgs(buffers, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&instanceNodeCount, bufferCount);
    kernels[LBVH_COLLISION_SOLVER_APPLY_COLLISIONS].setArg<uint>(&stablizationPass, bufferCount + 1);

    compute->execute(kernels[LBVH_COLLISION_SOLVER_APPLY_COLLISIONS], workgroupSize, workgroupCount);
#ifdef DEBUG_LBVH_SOLVER
    compute->sync();
#endif
  }
}
