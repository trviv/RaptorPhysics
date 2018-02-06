#include "UnifiedPhysics.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface compute;
static PhysicsSystem* physicsSystem;

int main(int argc, char** argv)
{
  compute.create(1);

  physicsSystem = new PhysicsSystem(&compute);
  main_window = physicsSystem;
  main_window->init(argc, argv);

  real dim[3];

  Cloth* cloth = new Cloth();

  dim[0] = 1;
  dim[1] = 1;

  //cloth->init(Matrix4(), dim, .5f, 32);
  uint subdivision[2] = { 8, 8 };
  cloth->init(Matrix4(), dim, subdivision, 32);
  physicsSystem->registerEntity(cloth);

  main_window->start();

  return 0;

  SharedAllocator allocator(&compute);
  allocator.constrainAllocator.create(1024, 1024 * 16);
  allocator.particleAllocator.create(1024);

  LinearSolver<ushort, float, float> cons(&compute, &allocator);

  cons.create(&compute);

  cons.addConnection(0, 0, 10);
  cons.addConnection(1, 1, 11);
  cons.addConnection(2, 2, 10);
  cons.addConnection(3, 3, 8);

  cons.addConnection(0, 1, -1);
  cons.addConnection(0, 2, 2);

  cons.addConnection(1, 0, -1);
  cons.addConnection(1, 2, -1);
  cons.addConnection(1, 3, 3);

  cons.addConnection(2, 0, 2);
  cons.addConnection(2, 1, -1);
  cons.addConnection(2, 3, -1);

  cons.addConnection(3, 1, 3);
  cons.addConnection(3, 2, -1);

  cons.setConstant(0, 6);
  cons.setConstant(1, 25);
  cons.setConstant(2, -11);
  cons.setConstant(3, 15);

  cons.commit();

  cons.addConnection(0, 0, 2);
  cons.addConnection(1, 1, 7);
  cons.addConnection(0, 1, 1);
  cons.addConnection(1, 0, 5);

  cons.setConstant(0, 11);
  cons.setConstant(1, 13);

  cons.commit();

  cons.solve();
  return 0;
}