#ifndef READER_SCENE_H
#define READER_SCENE_H

#include <UnifiedPhysics.h>
#include <RayTracing.h>

using namespace tinyxml2;

class MainSystem;

/*!
@class Class to read and initialize scene.
*/
class ReaderScene
{
  unordered_map<string, PhysicsEntityId>    registeredEntities;
  unordered_map<string, RayTracingEntityId> registeredRTEntities;
  unordered_map<string, MaterialId>         registeredMaterials;

  /*!@function Read settings for physics and renderer.*/
  void readSettings(MainSystem* system, XMLElement* settings);

  /*!@function Read entities in the scene.*/
  void readEntities(MainSystem* system, XMLElement* entities);

  /*!@function Read materials in the scene.*/
  void readMaterials(MainSystem* system, XMLElement* materials);

  /*!@function Instantiate entities.*/
  void createInstances(MainSystem* system, XMLElement* instances);

  /*!@function Apply transformation by reading the element.*/
  void transform(const XMLElement* element, Matrix* matrix);

public:

  bool readFile(MainSystem* system, const char fileName[]);
};

#endif
