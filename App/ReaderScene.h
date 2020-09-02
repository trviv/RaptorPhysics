#ifndef READER_SCENE
#define READER_SCENE

#include "Reader.h"

/*!
@class Class to read and initialize scene.
*/
class ReaderScene : public Reader
{
  unordered_map<string, PhysicsEntityId>  registeredEntities;

  /*!@function Read settings for physics and renderer.*/
  void readSettings(ComputeInterface* compute, PhysicsSystem* physicsSystem, Window* renderer, XMLElement* settings);

  /*!@function Read entities in the scene.*/
  void readEntities(ComputeInterface* compute, PhysicsSystem* physicsSystem, Window* renderer, XMLElement* entities);

  /*!@function Instantiate entities.*/
  void createInstances(ComputeInterface* compute, PhysicsSystem* physicsSystem, Window* renderer, XMLElement* instances);

  /*!@function Apply transformation by reading the element.*/
  void transform(const XMLElement* element, Matrix* matrix);

public:

  bool readFile(ComputeInterface* compute, PhysicsSystem* physicsSystem, Window* renderer, const char fileName[]);
};

#endif
