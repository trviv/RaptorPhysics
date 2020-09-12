#include <SDL2/SDL_main.h>
#include "MainSystem.h"

#pragma comment(lib, "glew32.lib")

int main(int argc, char** argv)
{
  // Initialize compute
  ComputeInterface compute;
  compute.create();

  // Create main system
  MainSystem mainSystem(&compute);
  // Load scene
  mainSystem.createFromFile("Scene_RT.xml");
  // Start system
  mainSystem.start();

  return 0;
}
