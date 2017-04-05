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

  objects.push_back(new Cloth());
  Counter subdivision[] = { 16, 16 };
  float dim[] = { 1, 1 };
  objects[0]->init(Matrix4(), dim, subdivision);

  /*objects.push_back(new RigidBody());
  Counter subdivision[] = { 3, 3, 3 };
  float dim[] = { 1, 1, 1 };
  objects[0]->init(Matrix4(), dim, subdivision);
  */
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