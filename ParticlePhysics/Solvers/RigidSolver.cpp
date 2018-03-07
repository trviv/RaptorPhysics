#include "RigidSolver.h"

static uint positionUtilId;
static uint matrix3x3UtilId;

//#define DEBUG_RIGID_SOLVER

#define RIGID_SOLVER_KERNEL_COVARIANCE 0
#define RIGID_SOLVER_KERNEL_DETERMINE_MATRIX 1
#define RIGID_SOLVER_KERNEL_SET_DELTA_POSITION 2

RigidSolver::RigidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator, SOLVER_RIGID_BODY)
{
  iterations = 2;
  create(compute);

  covarianceMatrix.create(compute, NULL, false);
}

void RigidSolver::commit()
{
  Solver::commit();

  flatArray<real>(*constrainCoefficients.host(), rawConstrainCoefficients);

  SectionData updateInfo;

  updateInfo.offsets[SECTION_DATA_NODE] = nodeOffset;
  updateInfo.counts[SECTION_DATA_NODE] = constrainConstants.host()->size() - nodeOffset;
  updateInfo.offsets[SECTION_DATA_CONNECTION] = connectionOffset;
  updateInfo.counts[SECTION_DATA_CONNECTION] = constrainCoefficients.host()->size() - connectionOffset;

  updates.push_back(updateInfo);

  nodeOffset += updateInfo.counts[SECTION_DATA_NODE];
  connectionOffset += updateInfo.counts[SECTION_DATA_CONNECTION];
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
    vector<ComputeUtilTuple> positionSetting;
    positionSetting.push_back(ComputeUtilTuple(ComputeUtilStructType, "ParticleStruct"));
    positionSetting.push_back(ComputeUtilTuple(ComputeUtilStructMember, "position"));

    vector<ComputeUtilTuple> matrix3x3Setting;
    matrix3x3Setting.push_back(ComputeUtilTuple(ComputeUtilStructType, "Matrix3x3"));
    matrix3x3Setting.push_back(ComputeUtilTuple(ComputeUtilCustomFunctionSuffix, "Matrix3x3"));;

    positionUtilId = ComputeUtil::create(compute, positionSetting, &include);
    matrix3x3UtilId = ComputeUtil::create(compute, matrix3x3Setting, &include);
  }
}

void RigidSolver::solve()
{
  if (updates.size()) // update arrays
  {
    update();
  }

  size_t workgroupSize[3], workgroupCount[3];
  uint count = nodes();
  compute->configureSize(workgroupSize, workgroupCount, count);

  covarianceMatrix.resize(count * 9, false);
  particlesTemp[0].resize(count, false);
  particlesTemp[1].resize(count, false);

  // copy to aux buffer to find new COM
  compute->copyBuffer(particles.device(), particlesTemp[0].device(), 0, 0, count * sizeof(ParticleStruct));

  // calculate current COM
  ComputeUtil::get(positionUtilId)->sum1D(compute, particlesTemp[0].device(), count, true);

#ifdef DEBUG_RIGID_SOLVER
  printf("\nMean:\n");
  ComputeUtil::get(positionUtilId)->showMatrix(compute, particlesTemp[0].device(), 3, 4, 3);
  compute->sync();
#endif

  // calculate covariance matrix
  {
    ComputeMemory* buffers[] = {
      covarianceMatrix.device(),
      particleDeltas.device(),
      particles.device(),
      particlesTemp[0].device(),
      particleRigidData.device()
    };
    kernels[RIGID_SOLVER_KERNEL_COVARIANCE].setArgs(buffers, 5);
    kernels[RIGID_SOLVER_KERNEL_COVARIANCE].setArg<uint>(&count, 5);
    compute->execute(kernels[RIGID_SOLVER_KERNEL_COVARIANCE], workgroupSize, workgroupCount);
  }

  // consolidate matrix for each body
  // add all n * 9 values to form = 3x3 matrix
  ComputeUtil::get(matrix3x3UtilId)->sum1D(compute, covarianceMatrix.device(), count, true);

  compute->copyBuffer(covarianceMatrix.device(), covarianceMatrix.device(), 0, 9 * sizeof(float), 9 * sizeof(float));

#ifdef DEBUG_RIGID_SOLVER
  printf("\nM:\n");
  ComputeUtil::get(matrix3x3UtilId)->showMatrix(compute, covarianceMatrix.device(), 3, 3, 9 * 2);
  compute->sync();
#endif

  {
    size_t workgroupSize[3], workgroupCount[3];
    uint rigidBodyCount = 1;
    compute->configureSize(workgroupSize, workgroupCount, rigidBodyCount);

    ComputeMemory* buffers[] = {
      covarianceMatrix.device(),
      particlesTemp[0].device(),
      particlesTemp[1].device()
    };

    uint step = 1;

    kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArgs(buffers, 3);
    kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArg<uint>(&step, 3);
    kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX].setArg<uint>(&rigidBodyCount, 4);
    compute->execute(kernels[RIGID_SOLVER_KERNEL_DETERMINE_MATRIX], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_RIGID_SOLVER
  printf("\nQ:\n");
  ComputeUtil::get(matrix3x3UtilId)->showMatrix(compute, covarianceMatrix.device(), 3, 3, 9 * 2);
  compute->sync();
#endif

  compute->copyBuffer(particles.device(), particlesTemp[0].device(), 0, 0, count * sizeof(ParticleStruct));

  {
    ComputeMemory* buffers[] = {
      particleDeltas.device(),
      covarianceMatrix.device(),
      particleRigidData.device()
    };

    kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION].setArgs(buffers, 3);
    kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION].setArg<uint>(&count, 3);
    compute->execute(kernels[RIGID_SOLVER_KERNEL_SET_DELTA_POSITION], workgroupSize, workgroupCount);
  }

#ifdef DEBUG_RIGID_SOLVER
  printf("\nDeltas:\n");
  ComputeUtil::get(positionUtilId)->showMatrix(compute, particleDeltas.device(), 3, 4, count * 3);
  compute->sync();
#endif

}

void RigidSolver::update()
{
  Solver::update();

  particleSharedData.syncDevice();
  particles.syncDevice();
  particleDeltas.resize(particles.size(), false);
  particleAuxData.syncDevice();
  particleRigidData.syncDevice();
}