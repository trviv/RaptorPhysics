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
  real simSize = 2.f;

  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;

  vector<Matrix4> matrixTransforms;

  uint subdivision1[2] = { 8, 8 };

  for (int c = 0; c < 0; c++)
  {
    Cloth* cloth = new Cloth();

    matrixTransforms.clear();

    cloth->initXY(dim, subdivision1, 1);

    int clothInstances = 2;
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
    subdivision1[0]++;
  }

  uint subdivision2[3] = { 2, 2, 2 };

  for (int r = 0; r < 2; r++)
  {
    RigidBody* rigidBody = new RigidBody();

    matrixTransforms.clear();

    rigidBody->initCube(dim, subdivision2, 1);
    int rbInstances = 1; // 2048;
    for (int i = 0; i < rbInstances; i++)
    {
      Matrix4 matrix;
      float randx = 1 * i;// (2 * float(rand()) / RAND_MAX) - 1;
      float randy = 0;// float(rand()) / RAND_MAX;
      float randz = 0;// (2 * float(rand()) / RAND_MAX) - 1;
      matrix.set(Matrix3::getIdentity(), Real3(randx * simSize, randy * simSize, randz * simSize));
      matrixTransforms.push_back(matrix);
    }
    physicsSystem->registerEntity(rigidBody, rbInstances, &matrixTransforms[0]);
    subdivision2[0]++;
  }

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}