#include "FluidSolverPBF.h"

//#define DEBUG_FLUID_PBF_SOLVER

#define FLUID_COLLISION_SOLVER_PBF_CREATE_BOUNDING_BOX  0
#define FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS          1
#define FLUID_COLLISION_SOLVER_PBF_CELL_ARRAYS          2
#define FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA          3
#define FLUID_COLLISION_SOLVER_PBF_CALC_FORCES          4
#define FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA           5
#define FLUID_COLLISION_SOLVER_PBF_VORT_VISC            6

FluidSolverPBF::FluidSolverPBF(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), FluidSolver(compute, allocator, true)
{
  FluidSolverPBF::create(compute);

  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);
}

void FluidSolverPBF::create(ComputeInterface* compute)
{
  includeFiles.push_back("UniformGridCollisionSolver.shader");

  registerShader(compute, "FluidSolverPBF.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("calculateLambda"));
  kernels.push_back(programs[0].createKernel("calculateForces"));
  kernels.push_back(programs[0].createKernel("vorticityOmega"));
  kernels.push_back(programs[0].createKernel("vorticityConfinementXSPHViscosity"));

  invMaxRadius.host()->push_back(-1.f);

  createUtilities();
}

void FluidSolverPBF::solve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  updateRadius();

  uint nodeBatchSize = 8;
  uint nodeBatchCount = mAlignBy(particleCount, nodeBatchSize);

  const uint gridElements = gridSize * gridSize * gridSize;

  size_t groupWorkgroupSize[3], groupWorkgroupCount[3];
  compute->configureSize(groupWorkgroupSize, groupWorkgroupCount, nodeBatchCount);

  if (particleGroupBoundingBoxes.size() < groupWorkgroupSize[0] * groupWorkgroupCount[0])
  {
    particleGroupBoundingBoxes.resize((uint)(groupWorkgroupSize[0] * groupWorkgroupCount[0]), false);
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

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
  }

  {
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
    kernels[FLUID_COLLISION_SOLVER_PBF_CREATE_BOUNDING_BOX].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_CREATE_BOUNDING_BOX].setArg<uint>(&nodeBatchCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_CREATE_BOUNDING_BOX].setArg<uint>(&particleCount, bufferCount + 1);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_CREATE_BOUNDING_BOX], groupWorkgroupSize, groupWorkgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  particleGroupBoundingBoxes.syncHost();
  compute->sync();
#endif

  // find bounding box for the simulation space
  ComputeUtil::get(gridXABComputeUtilId)->sum1D(compute, systemBoundingBox.device(), particleGroupBoundingBoxes.device(), (uint)(groupWorkgroupSize[0] * groupWorkgroupCount[0]));

#ifdef DEBUG_FLUID_PBF_SOLVER
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
#ifdef GRID_COLLISION_SOLVER_SCATTER_PARTICLES
      entitySharedData.device(),
      particleAuxData.device(),
      partitions.device(),
      entityLocations.device(),
      systemSettings,
#endif
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS].setArg<uint>(&particleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS].setArg<uint>(&gridSize, bufferCount + 1);
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS].setArg<uint>(&gridSizeExp, bufferCount + 2);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_CELL_COUNTS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  gridCellParticleCount.syncHost();
  gridParticleCellIndex.syncHost();
  compute->sync();
#endif

  // get prefix sum for each
  ComputeUtil::get(gridComputeUtilId)->prefixScan1D(compute, gridCellParticleOffsets.device(), gridCellParticleCount.device(), gridElements);

#ifdef DEBUG_FLUID_PBF_SOLVER
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
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_ARRAYS].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_CELL_ARRAYS].setArg<uint>(&particleCount, bufferCount);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_CELL_ARRAYS], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  gridCellParticleIndices.syncHost();
  gridCellParticleOffsets.syncHost();
  compute->sync();
#endif

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, particleCount);

  for (int i=0; i<iterations; i++)
  {
    ComputeUtil::get(0)->copyBuffer(compute, particles.device(), particlesTemp[1].device(), 0, 0, sizeof(ParticleStruct)*particleCount);

    {
      ComputeMemory* buffers[] = {
        particlesDensity.device(),
        particlesLambda.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
        gridParticleCellIndex.device(),
        particlesPredicted.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PBF_SOLVER
    particlesDensity.syncHost();
    particlesLambda.syncHost();
    compute->sync();
#endif

    {
      ComputeMemory* buffers[] = {
        particles.device(),
        particlesTemp[1].device(),
        particlesPredicted.device(),
        particlesDensity.device(),
        particlesLambda.device(),
        gridCellParticleOffsets.device(),
        gridCellParticleIndices.device(),
        gridParticleCellIndex.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_FORCES].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_FORCES].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_FORCES].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PBF_CALC_FORCES].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_CALC_FORCES], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PBF_SOLVER
    particlesPredicted.syncHost();
    particlesDensity.syncHost();
    particlesLambda.syncHost();
    compute->sync();
#endif
  }

  ComputeUtil::get(0)->copyBuffer(compute, particleDifferential.device(), particlesTemp[0].device(), 0, 0, sizeof(ParticleDifferential)*particleCount);
  ComputeUtil::get(0)->copyBuffer(compute, particlesPredicted.device(), UniformGridCollisionSolver::particlesBufferTemp.device(), 0, 0, sizeof(ParticleStruct)*particleCount);

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particlesTemp[1].device(),
      particleDifferential.device(),
      particlesPredicted.device(),
      gridCellParticleOffsets.device(),
      gridCellParticleIndices.device(),
      gridParticleCellIndex.device(),
      entitySharedData.device(),
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA].setArg<uint>(&gridSize, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA].setArg<uint>(&gridSizeExp, bufferCount + 1);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA].setArg<uint>(&particleCount, bufferCount + 2);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  particlesTemp[1].syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particlesPredicted.device(),
      UniformGridCollisionSolver::particlesBufferTemp.device(),
      particleDifferential.device(),
      particlesTemp[0].device(),
      particlesTemp[1].device(),
      gridCellParticleOffsets.device(),
      gridCellParticleIndices.device(),
      gridParticleCellIndex.device(),
      entitySharedData.device(),
      systemBoundingBox.device(),
      invMaxRadius.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC].setArg<uint>(&gridSize, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC].setArg<uint>(&gridSizeExp, bufferCount + 1);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC].setArg<uint>(&particleCount, bufferCount + 2);
    kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC].setArg<float>(&timeStep, bufferCount + 3);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_VORT_VISC], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  particleDifferential.syncHost();
  compute->sync();
#endif
}
