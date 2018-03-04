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
  cloth->init(Matrix4(), dim, subdivision1, 1);
  physicsSystem->registerEntity(cloth);

  uint subdivision2[3] = { 2, 2, 2 };
  rigidBody->init(Matrix4(), dim, subdivision2, 1);
  physicsSystem->registerEntity(rigidBody);

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}