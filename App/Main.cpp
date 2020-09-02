#include <SDL2/SDL_main.h>
#include "UnifiedPhysics.h"
#include "ReaderScene.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface* compute;
static PhysicsSystem* physicsSystem;
//static CameraInterface* cameraInterface;

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  physicsSystem = new PhysicsSystem();
  main_window = physicsSystem;

//  cameraInterface = new CameraInterface(compute);
//  cameraInterface->startSession();
//
//  physicsSystem->setCameraInterface(cameraInterface);

  ReaderScene reader;
  reader.readFile(compute, physicsSystem, main_window, "Scene.xml");

  main_window->start();

  delete physicsSystem;
  delete compute;
//  delete cameraInterface;
  
  return 0;
}
