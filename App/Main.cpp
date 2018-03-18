#include "UnifiedPhysics.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface* compute;
static PhysicsSystem* physicsSystem;

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create(1);

  physicsSystem = new PhysicsSystem(compute);
  main_window = physicsSystem;
  main_window->init(argc, argv);

  real dim[3];
  real simSize = 50.f;

  Cloth* cloth = new Cloth();
  RigidBody* rigidBody = new RigidBody();

  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;

  vector<Matrix4> matrixTransforms;

  uint subdivision1[2] = { 8, 8 };
  cloth->initXY(dim, subdivision1, 1);
  int clothInstances = 512;
  for (int i = 0; i < clothInstances; i++)
  {
    Matrix4 matrix;
    float randx = (2 * float(rand()) / RAND_MAX) - 1;
    float randy = float(rand()) / RAND_MAX;
    float randz = (2 * float(rand()) / RAND_MAX) - 1;
    matrix.set(Matrix3::getIdentity(), Real3(randx * simSize, randy * simSize, randz * simSize));
    matrixTransforms.push_back(matrix);
  }
  physicsSystem->registerEntity(cloth, clothInstances, &matrixTransforms[0]);

  matrixTransforms.clear();

  uint subdivision2[3] = { 2, 2, 2 };
  rigidBody->initCube(dim, subdivision2, 1);
  int rbInstances = 512;
  for (int i = 0; i < rbInstances; i++)
  {
    Matrix4 matrix;
    float randx = (2 * float(rand()) / RAND_MAX) - 1;
    float randy = float(rand()) / RAND_MAX;
    float randz = (2 * float(rand()) / RAND_MAX) - 1;
    matrix.set(Matrix3::getIdentity(), Real3(randx * simSize, randy * simSize, randz * simSize));
    matrixTransforms.push_back(matrix);
  }
  physicsSystem->registerEntity(rigidBody, rbInstances, &matrixTransforms[0]);

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}