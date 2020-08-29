#include <SDL2/SDL_main.h>
#include "UnifiedPhysics.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface* compute;
static PhysicsSystem* physicsSystem;
static CameraInterface* cameraInterface;

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  physicsSystem = new PhysicsSystem();
  main_window = physicsSystem;
  main_window->init(argc, argv, 1280, 1080);
  physicsSystem->init(compute, 128 * 1024);
  main_window->readCSVFile("AppConfig.csv");

  XAB systemBound;

  systemBound.min = physicsSystem->getParamAsFloat3("systemBoundMin");
  systemBound.max = physicsSystem->getParamAsFloat3("systemBoundMax");
  physicsSystem->setGravity(physicsSystem->getParamAsFloat3("gravity"));

  physicsSystem->setSystemBoundary(systemBound);

//  cameraInterface = new CameraInterface(compute);
//  cameraInterface->startSession();
//
//  physicsSystem->setCameraInterface(cameraInterface);

  vector<Matrix4> matrixTransforms;

  uint clothEntities = physicsSystem->getParamAsInt("clothEntities");
  uint clothInstances = physicsSystem->getParamAsInt("clothInstances");
  Int2 clothSubdivision = physicsSystem->getParamAsInt2("clothSubdivision");
  Real3 clothDim = physicsSystem->getParamAsFloat3("clothDimension");
  real clothSimSize = physicsSystem->getParamAsFloat("clothSimSize");

  for (int c = 0; c < clothEntities; c++)
  {
    Cloth* cloth = new Cloth();

    matrixTransforms.clear();

    cloth->initXY(&clothDim[0], (uint*)&clothSubdivision[0], 1);

    for (int i = 0; i < clothInstances; i++)
    {
      Matrix4 matrix;

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

  uint rigidEntities = physicsSystem->getParamAsInt("rigidEntities");
  uint rigidInstances = physicsSystem->getParamAsInt("rigidInstances");
  Real3 rigidDim = physicsSystem->getParamAsFloat3("rigidDimension");
  real rigidSimSize = physicsSystem->getParamAsFloat("rigidSimSize");

  for (int r = 0; r < rigidEntities; r++)
  {
    RigidBody* rigidBody = new RigidBody();

    matrixTransforms.clear();

    rigidBody->initCube(&rigidDim[0], .2 - .005 * r, .5*977.f);

    for (int i = 0; i < rigidInstances; i++)
    {
      Matrix4 matrix;
      float randx = r * 1.5 - rigidEntities * 0.5f;
      float randy = 4 + 1.2 * i;
      float randz = 2;

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

  uint fluidEntities = physicsSystem->getParamAsInt("fluidEntities");
  uint fluidInstances = 1;
  Real3 fluidDim = physicsSystem->getParamAsFloat3("fluidDimension");
  for (int f = 0; f < fluidEntities; f++)
  {
    Fluid* fluid = new Fluid();

    matrixTransforms.clear();

    fluid->initFluid(&fluidDim[0], physicsSystem->getParamAsFloat("fluidRadius"), fluidDim[0]*fluidDim[1]*fluidDim[2]*977.f, physicsSystem->getParamAsFloat("fluidRadius") * 4.f);

    for (int i = 0; i < fluidInstances; i++)
    {
      Matrix4 matrix;
      float randx = 0;
      float randy = 20.5f + 0.05 * i;
      float randz = 0;

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
//  delete cameraInterface;

  return 0;
}
