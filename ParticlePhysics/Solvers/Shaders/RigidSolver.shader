#ifndef RIGID_SOLVER_SHADER
#define RIGID_SOLVER_SHADER

/*
@kernel Compute covariance matrix.
@param matrixData Matrix data output.
@param particleDeltas Change in particle position.
@param particlesPredicted Integrated particle position.
@param particlesTemp Current particle position.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param length Rigid body count.
*/
Kernel void covarianceMatrix(
  Device float*                   matrixData,
  Device ParticleStruct*          particleDeltas,
  const Device ParticleStruct*    particlesPredicted,
  const Device ParticleStruct*    particlesTemp,
  const Device ParticleRigidData* rigidBodyData,
  const Device PartitionInfo*     partitions,
  const Device EntityLocation*    entityLocation,
  const uint length)
{
  const uint index = threadIndex();

  float3 currentComOffset, initialComOffset;

  if (index < length)
  {
    const ParticleStruct predicted = particlesPredicted[index];
    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(predicted.identity);
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    currentComOffset = predicted.position - particlesTemp[nodeIdentity.instanceId].position;
    initialComOffset = rigidBodyData[nodeLocator.commonNodeIndex].initialComOffset;

    // batch write respecting data alignment
    float8 ret8;
    ret8.s012 = currentComOffset * initialComOffset.x;
    ret8.s345 = currentComOffset * initialComOffset.y;
    ret8.s67  = currentComOffset.xy * initialComOffset.z;

    Device float* matrixRow = (matrixData + 9 * index);

    ((Device float8*)matrixRow)[0] = ret8;
    matrixRow[8] = currentComOffset.z * initialComOffset.z;

    // set delta now because com is available, and will be overwritten later
    // refer unified particle physics
    ParticleStruct particleOut;
    particleOut.position = -currentComOffset;
    particleOut.identity = predicted.identity;
    particleDeltas[index] = particleOut;
  }
}

void setAdjugateMatrix(
  Thread float* matrix2,
  const Thread float* matrixData)
{
  uint offset = 0;

  for (uint i = 0; i < 3; i++)
  {
    for (uint j = 0; j < 3; j++)
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
  float localMatrix[9],
  const Device float* matrixData,
  const uint iterations,
  uint instanceId)
{
  float matrix2[9];

  instanceId *= 9;

  ((Thread float8*)localMatrix)[0] = ((Device float8*)(matrixData + instanceId))[0];
  localMatrix[8] = matrixData[instanceId + 8];

  for (uint it = 0; it < iterations; it++)
  {
    setAdjugateMatrix(matrix2, localMatrix);

    float determinant = matrix2[0] * localMatrix[0] + matrix2[1] * localMatrix[1] + matrix2[2] * localMatrix[2];
    determinant = select(max(determinant, 0.0001f), min(determinant, -0.0001f), determinant < 0.f);
    // TODO: Some issue with gamma calculation half is stable
    const float gamma = .5f;// getGamma(matrix2, localMatrix, determinant);
    const float g1 = gamma * .5f;
    const float g2 = .5f / (gamma * determinant);

    ((Thread float8*)localMatrix)[0] = g1 * ((Thread float8*)localMatrix)[0] + g2 * ((Thread float8*)matrix2)[0];
    localMatrix[8] = g1 * localMatrix[8] + g2 * matrix2[8];
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
  const uint iterations,
  const uint length)
{
  const uint index = threadIndex();

  float localMatrix[9];

  if (index < length)
  {
    rigidSolverFunction(localMatrix, matrixData, iterations, index);

    // batch write respecting data alignment
    ((Device float8*)(matrixData + index * 9))[0] = ((Thread float8*)localMatrix)[0];
    (matrixData + index * 9)[8] = localMatrix[8];
  }
}

/*
@kernel Set delta position for rigid body.
@param particleDeltas Change in particle position.
@param matrixData Instance transformation matrix data.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param length Rigid body count.
*/
Kernel void setDeltaPosition(
  Device ParticleStruct*          particleDeltas,
  const Device float*             matrixData,
  const Device ParticleRigidData* rigidBodyData,
  Device ParticleCollisionData*   particleCollisionData,
  const Device PartitionInfo*     partitions,
  const Device EntityLocation*    entityLocation,
  const uint                      iterations,
  const uint                      length)
{
  const uint index = threadIndex();

  if (index < length)
  {
    float3 initialComOffset;
    float localMatrix[9];

    ParticleStruct delta = particleDeltas[index];

    const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(delta.identity);
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    rigidSolverFunction(localMatrix, matrixData, iterations, nodeIdentity.instanceId);
//    ((Thread float8*)localMatrix)[0] = ((Device float8*)(matrixData + nodeIdentity.instanceId*9))[0];
//    localMatrix[8] = matrixData[nodeIdentity.instanceId*9 + 8];

    initialComOffset = rigidBodyData[nodeLocator.commonNodeIndex].initialComOffset;

    float3 comOffsetCrossQ;
    Thread float* comOffsetCrossQPtr = (Thread float*)&comOffsetCrossQ;
    ParticleCollisionData collisionData = particleCollisionData[nodeLocator.commonNodeIndex];
    const float3 sdfGradientIn = collisionData.initialSdfGradient;
    float sdfGradientOut[3];

    for (uint i = 0; i < 3; i++)
    {
      const Thread float* particleMatrix = localMatrix + i;
      const float3 column = constructFloat3(particleMatrix[0], particleMatrix[3], particleMatrix[6]);
      comOffsetCrossQPtr[i] = dot(initialComOffset, column);
      sdfGradientOut[i] = dot(sdfGradientIn, column);
    }

    delta.position += comOffsetCrossQ;
    particleDeltas[index].position = delta.position;

    particleCollisionData[index].transformedSdfGradient = constructFloat3(sdfGradientOut[0], sdfGradientOut[1], sdfGradientOut[2]);
    particleCollisionData[index].invMass = collisionData.invMass;
  }
}

#endif
