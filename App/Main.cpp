#include <SDL2/SDL_main.h>
#include "UnifiedPhysics.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface* compute;
static PhysicsSystem* physicsSystem;

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  physicsSystem = new PhysicsSystem(compute, 256 * 1024);
  main_window = physicsSystem;
  main_window->init(argc, argv, 1280, 1080);

  uint clothEntities = 0;
  uint clothInstances = 1;

  uint rigidEntities = 0;
  uint rigidInstances = 100;

  uint fluidEntities = 1;
  uint fluidInstances = 1;

  XAB systemBound;

//  systemBound.min = Real3(-1.f, 0.f, -7.f);
//  systemBound.max = Real3(1.f, 0.f, -5.f);

//  systemBound.min = Real3(-2.f, 0.f, -2.f);
//  systemBound.max = Real3(2.f, 0.f, 2.f);

  systemBound.min = Real3(-7.f, 0.f, -7.f);
  systemBound.max = Real3(7.f, 0.f, 7.f);

//  systemBound.min = Real3(-4.f, 0.f, -4.f);
//  systemBound.max = Real3(4.f, 0.f, 4.f);

//  systemBound.min = Real3(-10.f, 0.f, -10.f);
//  systemBound.max = Real3(10.f, 0.f, 10.f);
//
//  systemBound.min = Real3(-70.f, 0.f, -70.f);
//  systemBound.max = Real3(70.f, 0.f, 70.f);

  physicsSystem->setGravity(Real3(0.f, -9.8f, 0.f));
  physicsSystem->setSystemBoundary(systemBound);

  vector<Matrix4> matrixTransforms;

  uint clothSubdivision[2] = { 12, 24 };
  real clothDim[3] = {2.f, 4.f, 2.f};
  real clothSimSize = 2.f;

  real rigidDim[3] = {2.0f, 2.0f, 2.0f};
  real rigidSimSize = 2.f;

//  real fluidDim[3] = {.9f, .8f, .9f};
  real fluidDim[3] = {1.2f, 22.5f, 1.5f};
  real fluidSimSize = 2.f;

  for (int c = 0; c < clothEntities; c++)
  {
    Cloth* cloth = new Cloth();

    matrixTransforms.clear();

    cloth->initXY(clothDim, clothSubdivision, 1);

    for (int i = 0; i < clothInstances; i++)
    {
      Matrix4 matrix;
//      float randx = (2 * float(rand()) / RAND_MAX) - 1;
//      float randy = 2 + float(rand()) / RAND_MAX;
//      float randz = -5 + (2 * float(rand()) / RAND_MAX) - 1;

      float randx = 1.5 * c - clothEntities * 0.5f;
      float randy = 1 + 1.05 * i;
      float randz = -1;

      matrix.set(Matrix3::getIdentity(), Real3(randx * clothSimSize, randy * clothSimSize, randz * clothSimSize));
      matrixTransforms.push_back(matrix);
    }
    PhysicsEntityId entity = physicsSystem->registerEntity(cloth);
    physicsSystem->addEntityInstance(entity, clothInstances, &matrixTransforms[0]);

    clothSubdivision[0]++;
    clothSubdivision[1]++;
  }

  for (int r = 0; r < rigidEntities; r++)
  {
    RigidBody* rigidBody = new RigidBody();

    matrixTransforms.clear();

    rigidBody->initCube(rigidDim, .4 - .00005 * r, 1);

    for (int i = 0; i < rigidInstances; i++)
    {
      Matrix4 matrix;
      float randx = r * 1.5 - rigidEntities * 0.5f;
      float randy = 1 + 1.05 * i;
      float randz = -5;

//      float randx = (5 * float(rand()) / RAND_MAX) - 1;
//      float randy = 5 + float(rand()) / RAND_MAX;
//      float randz = (5 * float(rand()) / RAND_MAX) - 1;

      matrix.set(Matrix3::getIdentity(), Real3(randx * rigidSimSize, randy * rigidSimSize, randz * rigidSimSize));
//      Matrix mat;
//      mat.setIdentity();
//      mat.rotate(Real3(10.f, 10.f, 10.f));
//      matrix *= mat[TRANS];
      matrixTransforms.push_back(matrix);
    }
    PhysicsEntityId entity = physicsSystem->registerEntity(rigidBody);
    physicsSystem->addEntityInstance(entity, rigidInstances, &matrixTransforms[0]);
  }

  for (int f = 0; f < fluidEntities; f++)
  {
    Fluid* fluid = new Fluid();

    matrixTransforms.clear();

    fluid->initFluid(fluidDim, .04, 1, 0.12f);

    for (int i = 0; i < fluidInstances; i++)
    {
      Matrix4 matrix;
      float randx = 0;//f * 1.5 - fluidEntities * 0.5f;
      float randy = 10.5f + 0.05 * i;
      float randz = 0;

      //      float randx = (5 * float(rand()) / RAND_MAX) - 1;
      //      float randy = 5 + float(rand()) / RAND_MAX;
      //      float randz = (5 * float(rand()) / RAND_MAX) - 1;

      matrix.set(Matrix3::getIdentity(), Real3(randx * rigidSimSize, randy * rigidSimSize, randz * rigidSimSize));
      //      Matrix mat;
      //      mat.setIdentity();
      //      mat.rotate(Real3(10.f, 10.f, 10.f));
      //      matrix *= mat[TRANS];
      matrixTransforms.push_back(matrix);
    }
    PhysicsEntityId entity = physicsSystem->registerEntity(fluid);
    physicsSystem->addEntityInstance(entity, fluidInstances, &matrixTransforms[0]);
  }

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}
