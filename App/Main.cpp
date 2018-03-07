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

  Cloth* cloth = new Cloth();
  RigidBody* rigidBody = new RigidBody();

  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;

  uint subdivision1[2] = { 8, 8 };
  cloth->initXY(dim, subdivision1, 1);
  vector<Matrix4> matrixTransforms;
  int clothInstances = 2;
  for (int i = 0; i < clothInstances; i++)
  {
    Matrix4 matrix;
    matrix.set(Matrix3::getIdentity(), Real3(.25*i, 0, .25*i));
    matrixTransforms.push_back(matrix);
  }
  physicsSystem->registerEntity(cloth, clothInstances, &matrixTransforms[0]);

  uint subdivision2[3] = { 2, 2, 2 };
  rigidBody->initCube(dim, subdivision2, 1);
  physicsSystem->registerEntity(rigidBody);

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}