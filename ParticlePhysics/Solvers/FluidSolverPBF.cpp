#include "FluidSolverPBF.h"

//#define DEBUG_FLUID_PBF_SOLVER

#define FLUID_COLLISION_SOLVER_REORDER                  3
#define FLUID_COLLISION_SOLVER_PBF_CALC_LAMBDA          4
#define FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION     5
#define FLUID_COLLISION_SOLVER_PBF_UPDATE_VELOCITY      6
#define FLUID_COLLISION_SOLVER_PBF_VORT_OMEGA           7
#define FLUID_COLLISION_SOLVER_PBF_VORT_VISC            8

FluidSolverPBF::FluidSolverPBF(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator), FluidSolver(compute, allocator, true)
{
  FluidSolverPBF::create(compute);

  iterations = 1;
  gridSize = 64;
  gridSizeExp = mCeilExpOf2(gridSize);

#ifdef DEBUG_FLUID_PBF_SOLVER
  particlesLambda.create(compute, solverHeap, true);
#else
  particlesLambda.create(compute, solverHeap);
#endif
}

void FluidSolverPBF::create(ComputeInterface* compute)
{
  includeFiles.push_back("UniformGridCollisionSolver.shader");
  includeFiles.push_back("FluidSolverCommon.h");

  registerShader(compute, "FluidSolverPBF.shader", NULL, NULL);

  kernels.push_back(programs[0].createKernel("createBoundingBoxes"));
  kernels.push_back(programs[0].createKernel("createGridCellHistogram"));
  kernels.push_back(programs[0].createKernel("createGridCellArrays"));
  kernels.push_back(programs[0].createKernel("reorderFluidParticles"));
  kernels.push_back(programs[0].createKernel("calculateLambda"));
  kernels.push_back(programs[0].createKernel("applyCorrection"));
  kernels.push_back(programs[0].createKernel("updateVelocities"));
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
    particlesCopy.resize(particleCount, false);
    particlesPredictedCopy.resize(particleCount, false);
    particleDifferentialCopy.resize(particleCount, false);
    particlesDensity.resize(particleCount, false);
    particlesLambda.resize(particleCount, false);
    gridParticleCellIndex.resize(particleCount, false);
    gridCellParticleIndices.resize(particleCount, false);
  }

  if (gridCellParticleCount.size() < gridElements)
  {
    gridCellParticleCount.resize(gridElements, false);
  }

  constructGrid();

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, particleCount);

  for (int i=0; i<iterations; i++)
  {
    ComputeUtil::get(0)->copyBuffer(compute, particlesPredicted.device(), particlesPredictedCopy.device(), 0, 0, sizeof(ParticleStruct)*particleCount);

    {
      ComputeMemory* buffers[] = {
        particlesDensity.device(),
        particlesLambda.device(),
        gridCellParticleOffsets.device(),
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
        particlesPredicted.device(),
        particlesPredictedCopy.device(),
        particlesDensity.device(),
        particlesLambda.device(),
        gridCellParticleOffsets.device(),
        gridParticleCellIndex.device(),
        entitySharedData.device(),
        systemBoundingBox.device(),
        invMaxRadius.device()
      };
      uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION].setArgs(buffers, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION].setArg<uint>(&gridSize, bufferCount);
      kernels[FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION].setArg<uint>(&gridSizeExp, bufferCount + 1);
      kernels[FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION].setArg<uint>(&particleCount, bufferCount + 2);

      compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_APPLY_CORRECTION], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_FLUID_PBF_SOLVER
    particlesPredicted.syncHost();
    compute->sync();
#endif
  }
}

void FluidSolverPBF::postCollisionSolve(float timeStep)
{
  uint particleCount = lastPartition().end();

  if (!particleCount) return;

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, particleCount);

  {
    ComputeMemory* buffers[] = {
      particleDifferential.device(),
      particlesPredicted.device(),
      particles.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[FLUID_COLLISION_SOLVER_PBF_UPDATE_VELOCITY].setArgs(buffers, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_UPDATE_VELOCITY].setArg<uint>(&particleCount, bufferCount);
    kernels[FLUID_COLLISION_SOLVER_PBF_UPDATE_VELOCITY].setArg<float>(&timeStep, bufferCount + 1);

    compute->execute(kernels[FLUID_COLLISION_SOLVER_PBF_UPDATE_VELOCITY], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_FLUID_PBF_SOLVER
  particleDifferential.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particlesOmega.device(),
      particleDifferential.device(),
      particlesPredicted.device(),
      gridCellParticleOffsets.device(),
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
  particlesOmega.syncHost();
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, particleCount);

    ComputeMemory* buffers[] = {
      particleForce.device(),
      particlesPredicted.device(),
      particleDifferential.device(),
      particlesOmega.device(),
      particlesDensity.device(),
      gridCellParticleOffsets.device(),
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
  particlesPredicted.syncHost();
  compute->sync();
#endif
}

void FluidSolverPBF::update()
{
  FluidSolver::update();

  for (auto& esd : *entitySharedData.host())
  {
    esd.fluidSolverData.fluidKernelFunctionConstant[0] = poly6FunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[1] = spikyFunctionConstant(esd.fluidSolverData.fluidKernelRadius);
    esd.fluidSolverData.fluidKernelFunctionConstant[2] = 1.f/(calculateGradientConstant(esd) * mSqr(esd.invRestDensity));
  }

  entitySharedData.syncDevice();
}
