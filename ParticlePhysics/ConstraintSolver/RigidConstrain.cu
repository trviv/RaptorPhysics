#include "RigidConstrain.h"

void RigidConstrain::exportToDevice(__int8** device_additional_memory,
  int additional_size, int baseSize)
{
  __int8* device_memory = 0;

  Real3 com(0);
  std::vector<Real3> com_offset;
  for (Counter i = 0; i < constrain_values.size(); i++)
    com += constrain_values[i];
  com /= real(constrain_values.size());
  for (Counter i = 0; i < constrain_values.size(); i++)
    com_offset.push_back(constrain_values[i] - com);

  IndexType total_size = sizeof(Real3)*com_offset.size() +
    9 * sizeof(real)*constrain_values.size() +
    sizeof(Real3)*constrain_values.size();

  constrain_values[0] -= real(.3);
  DistanceConstrain::exportToDevice(&device_memory, total_size,
    sizeof(RigidConstrain));

  device_com_offset = (Real3*)device_memory;
  device_matrix = (real*)(device_com_offset + com_offset.size());
  device_del_pos = (Real3*)(device_matrix + 9 * constrain_values.size());

  DeviceEntity<Real3>::exportToDevice(&com_offset[0], device_com_offset,
    com_offset.size());
  DeviceEntity<real>::set(device_matrix, 0, 9 * getNodeCount());
  DeviceEntity<Real3>::set(device_del_pos, 0, getNodeCount());
  DeviceEntity<RigidConstrain>::exportToDevice(this,
    (RigidConstrain*)constrain_alloc);

  std::cout << "Rigid Alloc:\n";
  std::cout << "Base: " << baseSize <<
    "\tAddr: " << (__int64)constrain_alloc << "\n";
  std::cout << "Com offset: " <<
    sizeof(Real3)*com_offset.size() <<
    "\tAddr: " << (__int64)device_com_offset << "\n";
  std::cout << "Matrix: " <<
    9 * sizeof(real)*constrain_values.size() <<
    "\tAddr: " << (__int64)device_matrix <<
    "\tDiff: " << (__int64)device_matrix - (__int64)device_com_offset << "\n";
  std::cout << "Del pos: " <<
    sizeof(Real3)*constrain_values.size() <<
    "\tAddr: " << (__int64)device_del_pos <<
    "\tDiff: " << (__int64)device_del_pos - (__int64)device_matrix << "\n";
}

enum RigidSolverStep
{
  COVARIANCE_MATRIX,
  MORE,
  LESS
};

CU_KER void covarianceMatrix(
  RigidConstrain* constrain)
{
  Counter index = threadIndex;
  const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
  index = index / 9;
  if (index < constrain->getNodeCount())
  {
    const Real3 diff = constrain->getValue(index) -
      constrain->getValue(0, VAR1);
    constrain->getMatrix()[index + ((3 * i) + j)*constrain->getNodeCount()]
      = diff[i] * constrain->getComOffset()[index][j];
    //= diff[i] * diff[j];
    constrain->getDelPosition()[index] = constrain->getValue(0, VAR1) -
      constrain->getPosition()[index];
  }
}

CU_KER void rigidSolver(
  RigidConstrain* constrain,
  const Counter step,
  const Counter length)
{
  Counter index = threadIndex;
  const Counter grid = blockDim.x*blockDim.y;
  const Counter multiplier = ceil(length / real(grid));
  const Counter backIndex = index;
  for (Counter it = 0; it < step; it++)
  {
    for (Counter m = 0; m < multiplier; m++)
    {
      index = backIndex + grid*m;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real adjTra;
        const real* value = (constrain->getMatrix() + index);
        adjTra = value[((i + 1) % 3) * 3 + ((j + 1) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 2) % 3)] -
          value[((i + 1) % 3) * 3 + ((j + 2) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 1) % 3)];

        ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j] = adjTra;
        ((real*)constrain->getValueBuffer(VAR0))[index + i * 3 + j] =
          adjTra*value[i * 3 + j];
        //printf("%f ", ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j]);
        //if (i == 0 && j == 0) printf("\n");
      }
    }
    __syncthreads();

    for (Counter m = 0; m < multiplier; m++)
    {
      index = backIndex + grid*m;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real det = ((real*)constrain->getValueBuffer(VAR0))[index] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 1] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 2];
        real gamma;
        {
          real mat_inf = 0, mat_one = 0, adj_inf = 0, adj_one = 0;
          for (Counter x = 0; x < 3; x++)
          {
            real sum_mat_1 = 0, sum_adj_1 = 0,
              sum_mat_0 = 0, sum_adj_0 = 0;
            for (Counter y = 0; y < 3; y++)
            {
              sum_mat_0 += fabs(constrain->getMatrix()[index + (x * 3) + y]);
              sum_mat_1 += fabs(constrain->getMatrix()[index + (y * 3) + x]);
              sum_adj_0 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (x * 3) + y]);
              sum_adj_1 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (y * 3) + x]);
            }
            if (sum_mat_0 > mat_inf)  mat_inf = sum_mat_0;
            if (sum_mat_1 > mat_one)  mat_one = sum_mat_1;
            if (sum_adj_0 > adj_inf)  adj_inf = sum_adj_0;
            if (sum_adj_1 > adj_one)  adj_one = sum_adj_1;
          }
          gamma = mSqrt(mSqr((adj_one*adj_inf) / (mat_one*mat_inf))
            / fabs(det));
        }
        const real g1 = gamma*real(.5);
        const real g2 = real(.5) / (gamma*det);
        const Counter mat_index = index + (i * 3) + j;
        constrain->getMatrix()[mat_index] =
          g1*constrain->getMatrix()[mat_index] +
          g2*((real*)constrain->getValueBuffer(VAR1))[mat_index];
      }
    }
    __syncthreads();
  }
}

CU_KER void deltaPos(RigidConstrain* constrain)
{
  Counter index = threadIndex;
  const Counter i = (index % 3);
  index = index / 3;
  if (index < constrain->getNodeCount())
  {
    //real multiplier = -1;
    /*
    const real det =
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount()] +
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount() + 1] +
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount() + 2];
    //if (det > 0)
    */
    //multiplier = 1;
    /*
    real* mat = constrain->getMatrix() + i * 3;
    constrain->getPosition(index)[i] +=
    //constrain->getValue(index, VAR1)[i] +=
    constrain->getDelPosition(index)[i]
    + (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]);
    */
    real* mat = constrain->getMatrix() + i;
    real com_offset_cross_q =
      constrain->getComOffset()[index][0] * mat[0] +
      constrain->getComOffset()[index][1] * mat[3] +
      constrain->getComOffset()[index][2] * mat[6];
    /*
    real* mat = constrain->getMatrix() + i * 3;
    real com_offset_cross_q =
    constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2];
    */
    //constrain->getPosition()[index][i] +=
    //constrain->getDelPosition()[index][i] + com_offset_cross_q;
    constrain->getDelPosition()[index][i] += com_offset_cross_q;
    //constrain->getDelPosition()[index][i] = -constrain->getDelPosition()[index][i];

    /*
    printf("Del: %f %f\n", constrain->getDelPosition(index)[i],
    (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]));
    printf("%f %f\n", constrain->getPosition(index)[i],
    constrain->getDelPosition(index)[i]
    + (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]));
    */
    /*
    printf("Del:%f %f\n", constrain->getDelPosition(index)[i],
    (constrain->getComOffset()[index][0] * mat[0] +
    constrain->getComOffset()[index][1] * mat[3] +
    constrain->getComOffset()[index][2] * mat[6]));
    */
    printf("Del:%d %f %f\n", index * 3 + i,
      constrain->getDelPosition()[index][i], com_offset_cross_q);
    printf("%d %f %f\n", index * 3 + i, constrain->getPosition()[index][i],
      constrain->getDelPosition()[index][i] + com_offset_cross_q);
  }
}

void RigidConstrain::solve()
{
  integrate();
  dim3 threads_pre, blocks_pre;
  dim3 threads_post, blocks_post;
  dim3 threads_upd, blocks_upd;

  configureGrid(blocks_pre, threads_pre, getNodeCount() * 9);
  configureGrid(blocks_post, threads_post, 9);
  configureGrid(blocks_upd, threads_upd, getNodeCount() * 3);

  //for (Counter i = 0; i < iterations; i++)
  {
    //copy to aux buffer to find new COM
    DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition());
    mean<Real3>(getValueBuffer(VAR1), getNodeCount());
    covarianceMatrix << <blocks_pre, threads_pre >> >((RigidConstrain*)constrain_alloc);
    cudaDeviceSynchronize();
    CU_PROMPT;
    for (Counter j = 0; j < 9; j++)
    {
      mean<real>(device_matrix + (j*getNodeCount()), getNodeCount());
      DeviceEntity<real>::copy(device_matrix + j,
        device_matrix + (j*getNodeCount()));
      CU_PROMPT;
    }
    DeviceEntity<real>::copy(device_matrix + 9,
      device_matrix, 9);

    printf("M:\n");
    matrix << <1, 1 >> >(device_matrix);
    cudaDeviceSynchronize();
    CU_PROMPT;

    rigidSolver << <1, threads_post >> >
      ((RigidConstrain*)constrain_alloc, 16, 1);
    cudaDeviceSynchronize();
    CU_PROMPT;

    printf("Q:\n");
    matrix << <1, 1 >> >(device_matrix);
    cudaDeviceSynchronize();
    CU_PROMPT;

    DeviceEntity<Real3>::copy(getValueBuffer(VAR1),
      getPosition(), getNodeCount());

    deltaPos << <blocks_upd, threads_upd >> >((RigidConstrain*)constrain_alloc);
    cudaDeviceSynchronize();
    CU_PROMPT;
  }
  /*
  if ((getIterations() & 1) == 0)
  {
  DeviceEntity<Real3>::copy((Real3*)getValueBuffer(DEF),
  (Real3*)getValueBuffer(VAR0), getNodeCount());
  cudaDeviceSynchronize();
  }
  */
  //DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(VAR1),
  //getNodeCount());
  //DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition(),
  //getNodeCount());
  differentiate();

  //DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(VAR1),
  //getNodeCount());
  //DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition(),
  //getNodeCount());
}