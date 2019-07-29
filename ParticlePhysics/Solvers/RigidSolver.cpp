#include "RigidSolver.h"

static uint positionUtilId;
static uint matrix3x3UtilId;

//#define DEBUG_RIGID_SOLVER

#define RIGID_SOLVER_KERNEL_COVARIANCE          0
#define RIGID_SOLVER_KERNEL_DETERMINE_MATRIX    1
#define RIGID_SOLVER_KERNEL_SET_DELTA_POSITION  2

#define RIGID_SVD_SOLVER_ITERATIONS             4

RigidSolver::RigidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator, SOLVER_RIGID_BODY)
{
  iterations = 1;
  create(compute);

#ifdef DEBUG_RIGID_SOLVER
  covarianceMatrix.create(compute, NULL, true);
#else
  covarianceMatrix.create(compute, NULL, false);
#endif
}

void RigidSolver::create(ComputeInterface* compute)
{
  if (programs.empty())
  {
    vector<string> newType = { "uint", "float", "float3" };
    vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
    registerShader(compute, "RigidSolver.shader", &oldType, &newType);
    kernels.push_back(programs[0].createKernel("covarianceMatrix"));
    kernels.push_back(programs[0].createKernel("rigidSolver"));
    kernels.push_back(programs[0].createKernel("setDeltaPosition"));

    vector<string> include = { "ParticleStruct.h" };
    map<ComputeUtilKey, string> positionSetting;
    positionSetting[ComputeUtilStructType] = "ParticleStruct";
    positionSetting[ComputeUtilStructMember] = "position";
    positionSetting[ComputeUtilStructMemberType] = "float3";
    positionSetting[ComputeUtilStructMemberSize] = "16";
    positionSetting[ComputeUtilIdentityFunction] = "getInstanceId";
    positionSetting[ComputeUtilIdentityStructType] = "ParticleStruct";
    positionSetting[ComputeUtilIdentityStructMember] = "identity";

    map<ComputeUtilKey, string> matrix3x3Setting;
    matrix3x3Setting[ComputeUtilStructType] = "Matrix3x3";
    matrix3x3Setting[ComputeUtilStructSize] = "36";
    matrix3x3Setting[ComputeUtilIdentityFunction] = "getInstanceId";
    matrix3x3Setting[ComputeUtilCustomAddFunction] = "addMatrix3x3";
    matrix3x3Setting[ComputeUtilCustomDivFunction] = "divMatrix3x3";
    matrix3x3Setting[ComputeUtilCustomCopyFunction] = "copyMatrix3x3";
    matrix3x3Setting[ComputeUtilCustomClearFunction] = "clearMatrix3x3";
    matrix3x3Setting[ComputeUtilIdentityStructType] = "ParticleStruct";
    matrix3x3Setting[ComputeUtilIdentityStructMember] = "identity";
    matrix3x3Setting[ComputeUtilSkipParallelPrimitives] = "1";

    positionUtilId = ComputeUtil::create(compute, positionSetting, &include);
    matrix3x3UtilId = ComputeUtil::create(compute, matrix3x3Setting, &include);
  }
}

void RigidSolver::solve()
{
  uint count = lastPartition().end();

  if (!count) return;

  uint totalEntities = newEntityInstanceId();
  size_t workgroupSize[3], workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, count);

  //TODO: Perperly implement this loop. Which should perhaps fix wobbling
  for (uint iteration = 0; iteration < iterations; iteration++)
  {
    // calculate current COM
    ComputeUtil::get(positionUtilId)->sumIrregular2D(compute, particlesTemp[0].device(),
      particlesPredicted.device(), particlesPredicted.device(), partitions.device(), count, true);

#ifdef DEBUG_RIGID_SOLVER
    printf("\nMean:\n");
    ComputeUtil::get(matrix3x3UtilId)->showMatrix(compute, particlesTemp[0].device(), 3, 4, 3 * totalEntities);
    particlesTemp[0].syncHost();
    compute->sync();
#endif

    // calculate covariance matrix
    {
      ComputeMemory* buffers[] = {
        covarianceMatrix.device(),
        particlesPredicted.device(),
        particlesTemp[0].device(),
        particleRigidData.device(),
        partitions.device(),
        entityLocations.device()
      };

      uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[RIGID_SOLVER_KERNEL_COVARIANCE].setArgs(buffers, bufferOffset);
      kernels[RIGID_SOLVER_KERNEL_COVARIANCE].setArg<uint>(&count, bufferOffset);
      compute->execute(kernels[RIGID_SOLVER_KERNEL_COVARIANCE], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_RIGID_SOLVER
    covarianceMatrix.syncHost();
    compute->sync();
#endif

    // consolidate matrix for each body
    // add all n * 9 values to form = 3x3 matrix
    ComputeUtil::get(matrix3x3UtilId)->sumIrregular2D(compute, particlesTemp[0].device(), covarianceMatrix.device(),
      particlesPredicted.device(), partitions.device(), count, true);

#ifdef DEBUG_RIGID_SOLVER
    printf("\nM:\n");
    ComputeUtil::get(matrix3x3UtilId)->showMatrix(compute, particlesTemp[0].device(), 3, 3, 9 * totalEntities);
    covarianceMatrix.syncHost();
    particlesTemp[0].syncHost();
    compute->sync();

    {
      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, totalEntities);

      ComputeMemory* buffers[] = {
        particlesTemp[0].device()
      };

      uint svdIterations = RIGID_SVD_SOLVER_ITERATIONS;

      uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArgs(buffers, bufferOffset);
      kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArg<uint>(&svdIterations, bufferOffset);
      kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArg<uint>(&totalEntities, bufferOffset + 1);
      compute->execute(kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX], workgroupSize, workgroupCount);
    }

    printf("\nQ:\n");
    ComputeUtil::get(matrix3x3UtilId)->showMatrix(compute, particlesTemp[0].device(), 3, 3, 9 * totalEntities);
    compute->sync();
#endif

    {
      uint svdIterations = RIGID_SVD_SOLVER_ITERATIONS;

      ComputeMemory* buffers[] = {
        particlesPredicted.device(),
        particlesTemp[0].device(),
        particleRigidData.device(),
        particleCollisionData.device(),
        partitions.device(),
        entityLocations.device()
      };

      uint bufferOffset = sizeof(buffers) / sizeof(ComputeMemory*);
      kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION].setArgs(buffers, bufferOffset);
      kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION].setArg<uint>(&svdIterations, bufferOffset);
      kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION].setArg<uint>(&count, bufferOffset + 1);
      compute->execute(kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION], workgroupSize, workgroupCount);
    }

#ifdef DEBUG_RIGID_SOLVER
    printf("\nDeltas:\n");
    ComputeUtil::get(positionUtilId)->showMatrix(compute, particleDeltas.device(), 3, 4, count * 3);
    compute->sync();
#endif
  }
}

void RigidSolver::update()
{
  if (!updates.size()) return;

  Solver::update();

  particleRigidData.syncDevice();

  uint count = lastPartition().end();

  covarianceMatrix.resize(count * 9, false);

  // TODO: look why it has to be greater then partitions
  particlesTemp[0].resize(count, false);
}
