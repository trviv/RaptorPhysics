#include "PhysicsSystem.h"
#include "Cloth.h"
#include "RigidBody.h"

PhysicsSystem* physics_system = NULL;

PhysicsSystem::~PhysicsSystem()
{
  for (Counter i = 0; i < objects.size(); i++)
  {
    delete objects[i];
    objects[i] = NULL;
  }
}

void PhysicsSystem::init(int argc, char** argv, int width,
  int height, const char* name)
{
  Window::init(argc, argv, width, height);
  Counter count = 0;
  Counter subdivision[3];
  real dim[3];
  /*
  objects.push_back(new Cloth());
  subdivision[0] = 16;
  subdivision[1] = 16;
  dim[0] = 1;
  dim[1] = 1;

  objects[count]->init(Matrix4(), dim, .1, 32);
  count++;
  constrain.pushObject(OBJ_CLOTH);
  */

  objects.push_back(new RigidBody());
  subdivision[0] = 3;
  subdivision[1] = 3;
  subdivision[2] = 3;
  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;

  objects[count]->init(Matrix4(), dim, .25, 16);
  constrain.pushObject(OBJ_RIGID_BODY);

  constrain.exportToDevice();
}

void PhysicsSystem::render()
{
  for (Counter i = 0; i < objects.size(); i++)
  {
    objects[i]->render();
  }
}

void PhysicsSystem::step()
{
  for (Counter i = 0; i < objects.size(); i++)
  {
    objects[i]->step();
  }
}