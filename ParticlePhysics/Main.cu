#include "../Core/Core.h"
//#include "ConstraintSolver/PointConstrain.h"
#include <vector>
//#include "Objects/Cloth.h"
#include "Objects/PhysicsSystem.h"

#pragma comment(lib, "glew32.lib")
/*
template<class T>CU_KER void meanKernel(T* device_array, const Counter length,
const Counter iteration, const Counter max_iterations, const Counter divide)
{
Counter index = ((threadIndex) << (1 + iteration));
for (Counter i = 0; i < max_iterations; i++)
{
Counter index2 = index + (1 << (i + iteration));
if (index2 < length) device_array[index] += device_array[index2];
index <<= 1;
__syncthreads();
}
if (index == 0 && divide) device_array[index] /= length;
}

template<class T>void mean(T* device_array, const Counter length,
const bool only_sum = false)
{
Counter iterations = mExpOf2(length);
Counter max_block_parallelism = mExpOf2(2 * MAX_BLOCK_PARALLELISM);
for (Counter i = 0; i < iterations; i++)
{
dim3 threads, blocks;
configureGrid(blocks, threads, ceil(float(length) / ((1 << i) * 2)));
meanKernel << <blocks, threads >> >(device_array, length, i,
((iterations - i)>max_block_parallelism) ? 1 : (iterations - i),
((iterations - i) <= max_block_parallelism) ? (only_sum ? 0 : 1) : 0);
cudaDeviceSynchronize();
CU_PROMPT;
if (iterations - i <= max_block_parallelism) break;
}
}

template<class T>void sum(T* device_array, const Counter length)
{
mean(device_array, length, true);
}
*/
int main(int argc, char** argv)
{
  /*
  Cloth * cl = new Cloth();
  main_window = cl;
  main_window->init(argc, argv);
  cl->init(32, 32, Real3(-1, 1, 0), 2, -2);
  main_window->start();
  */

  /*
  const int elements = 1234589;
  float *arr = new float[elements];
  float sum = 0;
  for (int i = 0; i < elements; i++)
  {
  arr[i] = float(rand()) / RAND_MAX;
  sum += arr[i];
  }
  //sum /= elements;

  DeviceAllocator<float> da;
  da.alloc(elements);
  DeviceEntity<float>::exportToDevice(arr, da.get(), elements);
  mean(da.get(), elements);
  DeviceEntity<float>::importToHost(arr, da.get(), elements);
  std::cout << "\n";
  for (int i = 0; i < elements; i++)
  {
  std::cout << arr[i] << " ";
  }
  delete[]arr;
  */

  ConstrainSolver<int, float, float> cons;

  /*
  cons.add(0, 0, 1);
  cons.add(1, 1, 1);
  cons.add(2, 2, 1);
  cons.add(0, 1, 1);
  cons.add(1, 0, 1);
  cons.add(1, 2, 1);
  cons.add(2, 1, 1);

  cons.setValue(0, 1);
  cons.setValue(1, 2);
  cons.setValue(2, 3);
  */
  /*
  cons.addCC(0, 0, 10);
  cons.addCC(1, 1, 11);
  cons.addCC(2, 2, 10);
  cons.addCC(3, 3, 8);

  cons.addCC(0, 1, -1);
  cons.addCC(0, 2, 2);

  cons.addCC(1, 0, -1);
  cons.addCC(1, 2, -1);
  cons.addCC(1, 3, 3);

  cons.addCC(2, 0, 2);
  cons.addCC(2, 1, -1);
  cons.addCC(2, 3, -1);

  cons.addCC(3, 1, 3);
  cons.addCC(3, 2, -1);

  cons.setValue(0, 6);
  cons.setValue(1, 25);
  cons.setValue(2, -11);
  cons.setValue(3, 15);

  cons.pushObject(OBJ_EQUATION);
  cons.addCC(0, 0, 2);
  cons.addCC(1, 1, 7);
  cons.addCC(0, 1, 1);
  cons.addCC(1, 0, 5);


  cons.setValue(0, 11);
  cons.setValue(1, 13);

  cons.pushObject(OBJ_EQUATION);
  //cons.show();
  cons.solve();
  return 0;
  */
  physics_system = new PhysicsSystem();
  main_window = physics_system;
  main_window->init(argc, argv);
  main_window->start();

  //Constraint con;
  //con.setCount(17);
  //con.setOffset(273);

  //std::cout << con.count() << " " << con.offset() << "\n";
  cudaSharedMemConfig cn;
  cudaDeviceGetSharedMemConfig(&cn);
  //std::cout << cn << "\n";

  /*
  int count = pow(2, 23);
  float *numbers = new float[count];
  for (int i = 0; i < count; i++)
  {
  int n = rand();
  n = rand()*rand();
  numbers[i] = n;// count - i;
  }

  sort<float>(numbers, count);
  */
  /*
  for (int i = 0; i < count; i++)
  {
  //fl << numbers[i] << "\n";
  std::cout << numbers[i] << "\n";
  }
  */
  //delete numbers;
  /*
  //func();
  int count = pow(2, 8);
  //count += pow(2, 18);
  float *numbers = new float[count];
  float *device_numbers = NULL;
  for (int i = 0; i < count; i++)
  {
  int n = rand();
  n = rand()*rand();
  numbers[i] = n;// count - i;
  }
  cudaMalloc(&device_numbers, sizeof(float)*count);
  CU_PROMPT;
  cudaMemcpy(device_numbers, numbers, sizeof(float)*count, cudaMemcpyHostToDevice);
  CU_PROMPT;
  std::cout << "start" << "\n";
  int y = count / 64 > 32 ? 32 : count / 64;
  bitonicSort << < 1, dim3(32, y), 2 * 32 * y*sizeof(float) >> > (device_numbers, count);
  CU_PROMPT;
  cudaDeviceSynchronize();
  std::cout << "stop" << "\n";
  CU_PROMPT;
  cudaMemcpy(numbers, device_numbers, sizeof(float)*count, cudaMemcpyDeviceToHost);
  CU_PROMPT;
  cudaFree(device_numbers);
  CU_PROMPT;

  std::ofstream fl("array.txt", std::ios::binary);
  for (int i = 0; i < count; i++)
  {
  //fl << numbers[i] << "\n";
  //std::cout << numbers[i] << "\n";
  }
  fl.close();

  delete numbers;
  */
  return 0;
}