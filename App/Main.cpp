/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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
//  mainSystem.createFromFile("Scene_RT.xml");
  mainSystem.createFromFile("Scene_Fluid.xml");
//  mainSystem.createFromFile("Cornell.xml");
//  mainSystem.createFromFile("Scene_Mesh.xml");
  // Start system
  mainSystem.start();

  return 0;
}
