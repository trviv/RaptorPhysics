#ifndef RIGID_SOLVER_SHADER
#define RIGID_SOLVER_SHADER

#include "ParticleStruct.h"

/*
@kernel Compute covariance matrix.
@param matrixData Matrix data output.
@param particlesPredicted Integrated particle position.
@param particlesTemp Current particle position.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param length Rigid body count.
*/
Kernel void covarianceMatrix(
  Device float*                   matrixData,
  Device ParticleStruct*          particlesPredicted,
  const Device ParticleStruct*    particlesTemp,
  const Device ParticleRigidData* rigidBodyData,
  const Device PartitionInfo*     partitions,
  const Device EntityLocation*    entityLocation,
  constantKernelInput(uint,       length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < length)
  {
    ParticleStruct predicted = particlesPredicted[index];
    const IdentityInfo identity = predicted.identity;
    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    float3 currentComOffset = predicted.position - particlesTemp[nodeIdentity.instanceId].position;

    // Update positions using deltas
    predicted.position -= currentComOffset;
    predicted.identity = identity;
    particlesPredicted[index] = predicted;

    const float3 initialComOffset = rigidBodyData[nodeLocator.commonNodeIndex].initialComOffset;

    matrixData += 9 * index;

    float4 ret1 = currentComOffset.xyzx * initialComOffset.xxxy;
    writeFloat4ToDeviceFloat(ret1, matrixData);

    float4 ret2 = currentComOffset.yzxy * initialComOffset.yyzz;
    writeFloat4ToDeviceFloat(ret2, matrixData + 4);

    matrixData[8] = currentComOffset.z * initialComOffset.z;
  }
}

void setAdjugateMatrix(
  Thread float* matrix2,
  const Thread float* matrixData)
{
  uchar offset = 0;

  for (uchar i = 0; i < 3; i++)
  {
    for (uchar j = 0; j < 3; j++)
    {
      // adjugate is the transpose of cofactor
      // https://en.wikipedia.org/wiki/Adjugate_matrix

      float cellValue =
        matrixData[((i + 1) % 3) * 3 + ((j + 1) % 3)] *
        matrixData[((i + 2) % 3) * 3 + ((j + 2) % 3)] -
        matrixData[((i + 1) % 3) * 3 + ((j + 2) % 3)] *
        matrixData[((i + 2) % 3) * 3 + ((j + 1) % 3)];

      matrix2[offset] = cellValue;
      offset++;
    }
  }
}

float getGamma(const Thread float* matrix2, const Thread float* matrixPtr, const float determinant)
{
  float mat_inf = 0,
    mat_one = 0,
    adj_inf = 0,
    adj_one = 0;

  for (uint x = 0; x < 3; x++)
  {
    float sum_mat_0 = 0,
      sum_mat_1 = 0,
      sum_adj_0 = 0,
      sum_adj_1 = 0;

    for (uint y = 0; y < 3; y++)
    {
      const uint offset0 = (x * 3) + y;
      const uint offset1 = (y * 3) + x;
      sum_mat_0 += fabs(matrixPtr[offset0]);
      sum_mat_1 += fabs(matrixPtr[offset1]);
      sum_adj_0 += fabs(matrix2[offset0]);
      sum_adj_1 += fabs(matrix2[offset1]);
    }

    if (sum_mat_0 > mat_inf)  mat_inf = sum_mat_0;
    if (sum_mat_1 > mat_one)  mat_one = sum_mat_1;
    if (sum_adj_0 > adj_inf)  adj_inf = sum_adj_0;
    if (sum_adj_1 > adj_one)  adj_one = sum_adj_1;
  }

  return sqrt(sqr((adj_one * adj_inf) / (mat_one * mat_inf)) / fabs(determinant));
}

//TODO: Use a better SVD solver, this one becomes unstable with com offset<1
/*
 @kernel Matrix SVD decomposition kernel.
 @param localMatrix Matrix data output.
 @param matrixData Matrix data input.
 @param iterations Iterations for the solver.
 @param instanceId Rigid body instance for offsetting to matrix data.
 @info Based on Computing the Polar Decomposition with Applications Nicholas J. Higham 1986
*/
inline void rigidSolverFunction(
  Thread float localMatrix[9],
  const Device float* matrixData,
  const uint iterations,
  uint instanceId)
{
  float matrix2[9];

  matrixData += instanceId * 9;

  readFromDevice4x(localMatrix, matrixData);
  readFromDevice4x(localMatrix+4, matrixData+4);
  localMatrix[8] = matrixData[8];

  for (uint it = 0; it < iterations; it++)
  {
    setAdjugateMatrix(matrix2, localMatrix);

    float determinant = matrix2[0] * localMatrix[0] + matrix2[1] * localMatrix[1] + matrix2[2] * localMatrix[2];
    determinant = select(max(determinant, 0.0001f), min(determinant, -0.0001f), determinant < 0.f);
    // TODO: Some issue with gamma calculation half is stable
    const float gamma = .5f;// getGamma(matrix2, localMatrix, determinant);
    const float g1 = gamma * .5f;
    const float g2 = .5f / (gamma * determinant);

    for (ushort i = 0; i < 9; i++)
    {
      localMatrix[i] = g1 * localMatrix[i] + g2 * matrix2[i];
    }
  }
}

/*
@kernel Matrix SVD decomposition kernel.
@param matrixData Matrix data output.
@param iterations Iterations for the solver.
@param length Rigid body count.
@info Based on Computing the Polar Decomposition with Applications Nicholas J. Higham 1986
*/
Kernel void rigidSolver(
  Device float* matrixData,
  constantKernelInput(uint, iterations),
  constantKernelInput(uint, length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  Thread float localMatrix[9];

  if (index < length)
  {
    rigidSolverFunction(localMatrix, matrixData, iterations, index);

    matrixData += index * 9;

    writeToDevice4x(matrixData, localMatrix);
    writeToDevice4x(matrixData+4, localMatrix+4);
    matrixData[8] = localMatrix[8];
  }
}

/*
@kernel Set delta position for rigid body.
@param particlesPredicted Integrated particle position.
@param matrixData Instance transformation matrix data.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param length Rigid body count.
*/
Kernel void setDeltaPosition(
  Device ParticleStruct*          particlesPredicted,
  const Device float*             matrixData,
  const Device ParticleRigidData* rigidBodyData,
  Device ParticleCollisionData*   particleCollisionData,
  const Device PartitionInfo*     partitions,
  const Device EntityLocation*    entityLocation,
  constantKernelInput(uint,       iterations),
  constantKernelInput(uint,       length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < length)
  {
    float localMatrix[9];

    const ParticleStruct delta = particlesPredicted[index];

    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(delta.identity);
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    rigidSolverFunction(localMatrix, matrixData, iterations, nodeIdentity.instanceId);
    const ParticleRigidData rigidData = rigidBodyData[nodeLocator.commonNodeIndex];

    float3 comOffsetCrossQ;
    Thread float* comOffsetCrossQPtr = (Thread float*)&comOffsetCrossQ;

    float sdfGradientOut[3];

    for (uint i = 0; i < 3; i++)
    {
      const Thread float* particleMatrix = localMatrix + i;
      const float3 column = constructFloat3(particleMatrix[0], particleMatrix[3], particleMatrix[6]);

      comOffsetCrossQPtr[i] = dot(rigidData.initialComOffset, column);
      sdfGradientOut[i] = dot(rigidData.initialSdfGradient, column);
    }

    const float3 normalizedSdfGradient = normalize(constructFloat3(sdfGradientOut[0], sdfGradientOut[1], sdfGradientOut[2]));

    particleCollisionData[index].transformedSdfGradient = encodeDirection(normalizedSdfGradient);
    particleCollisionData[index].gradientMagnitude = rigidData.gradientMagnitude;

    particlesPredicted[index].position += comOffsetCrossQ;
    particlesPredicted[index].identity = delta.identity;
  }
}

#endif
