#include "ReaderScene.h"
#include "MainSystem.h"
#include <sstream>

void QueryFloat3Attribute(const XMLElement* element, float* value, const char* attributeName="xyz")
{
  if (!element)
  {
    return;
  }

  Real3 val;
  val.setNull();

  const XMLAttribute* attr = element->FindAttribute(attributeName);
  if (attr)
  {
    std::stringstream ss(attr->Value());
    std::string token;
    int i=0;
    while (std::getline(ss, token, ',') && i<3)
    {
      val[i] = atof(token.c_str());
      i++;
    }
    value[0] = val[0];
    value[1] = val[1];
    value[2] = val[2];
    return;
  }

  element->QueryFloatAttribute("x", &val.x);
  element->QueryFloatAttribute("y", &val.y);
  element->QueryFloatAttribute("z", &val.z);

  value[0] = val[0];
  value[1] = val[1];
  value[2] = val[2];
}

void QueryFloat3Attribute(const XMLElement* element, float3& value, const char* attributeName="xyz")
{
  QueryFloat3Attribute(element, &value.x, attributeName);
}

void QueryFloat3Attribute(const XMLElement* element, Real3& value, const char* attributeName="xyz")
{
  QueryFloat3Attribute(element, &value.x, attributeName);
}

void QueryTransformElement(const XMLElement* element, Matrix& matrix)
{
  matrix.setIdentity();

  if (!element)
  {
    return;
  }

  for (const XMLElement* childElem = element->FirstChildElement(); childElem; childElem = childElem->NextSiblingElement())
  {
    Real3 value = {0.f, 0.f, 0.f};
    QueryFloat3Attribute(childElem, value);

    if (strcmp(childElem->Name(), "translate") == 0)
    {
      matrix.translate(value);
    }
    else if (strcmp(childElem->Name(), "scale") == 0)
    {
      matrix.scale(value);
    }
    else if (strcmp(childElem->Name(), "rotate") == 0)
    {
      matrix.rotate(value);
    }
  }
}

void QueryInt3Attribute(const XMLElement* element, int value[3], const char* attributeName)
{
  if (!element)
  {
    return;
  }

  int val[3] = {0, 0, 0};

  const XMLAttribute* attr = element->FindAttribute(attributeName);
  if (attr)
  {
    std::stringstream ss(attr->Value());
    std::string token;
    int i=0;
    while (std::getline(ss, token, ',') && i<3)
    {
      val[i] = atof(token.c_str());
      i++;
    }
    value[0] = val[0];
    value[1] = val[1];
    value[2] = val[2];
    return;
  }

  value[0] = val[0];
  value[1] = val[1];
  value[2] = val[2];
}

void ReaderScene::readSettings(MainSystem* system, XMLElement* settings)
{
  Window* renderer = system;
  ComputeInterface* compute = system->compute;
  PhysicsSystem* physicsSystem = &system->physicsSystem;
  RayTracingSystem* rayTracingSystem = &system->rayTracingSystem;

  XMLError queryResult;

  XMLElement* render = settings->FirstChildElement("rendering");
  XMLElement* physics = settings->FirstChildElement("physics");
  XMLElement* rayTracing = settings->FirstChildElement("ray-tracing");

  uint width = 640, height = 480;
  render->QueryUnsignedAttribute("width", &width);
  render->QueryUnsignedAttribute("height", &height);
  renderer->init(0, NULL, width, height);

  uint maxParticles;
  queryResult = physics->QueryUnsignedAttribute("max-particles", &maxParticles);
  if (queryResult)
  {
    logComputeError("Physics Max Memory is required!");
  }
  physicsSystem->init(compute, maxParticles);

  if (physics->FirstChildElement("bound-min") || physics->FirstChildElement("bound-max"))
  {
    XAB systemBound;
    QueryFloat3Attribute(physics->FirstChildElement("bound-min"), systemBound.min);
    QueryFloat3Attribute(physics->FirstChildElement("bound-max"), systemBound.max);
    physicsSystem->setSystemBoundary(systemBound);
  }

  if (physics->FirstChildElement("gravity"))
  {
    Real3 gravity;
    QueryFloat3Attribute(physics->FirstChildElement("gravity"), gravity);
    physicsSystem->setGravity(gravity);
  }

  physics->FirstChildElement("simulation-iterations")->QueryIntAttribute("value", &physicsSystem->simulationIterations);
  physics->FirstChildElement("solver-iterations")->QueryIntAttribute("value", &physicsSystem->solverIterations);
  physics->FirstChildElement("frame-capture-start")->QueryUnsignedAttribute("value", &system->frameCaptureStart);
  physics->FirstChildElement("frame-capture-end")->QueryUnsignedAttribute("value", &system->frameCaptureEnd);

  if (physics->FirstChildElement("stream-max-capacity"))
  {
    uint sizeInBytes = 0;
    physics->FirstChildElement("stream-max-capacity")->QueryUnsignedAttribute("value", &sizeInBytes);
    system->particlePositionStream.setMaxCapacity(sizeInBytes);
  }

  render->FirstChildElement("render-particles-option")->QueryBoolAttribute("value", &renderer->optionFrame->getElement(RENDER_PARTICLES_OPTION)->boolValue);
  render->FirstChildElement("render-solids-option")->QueryBoolAttribute("value", &renderer->optionFrame->getElement(RENDER_SOLIDS_OPTION)->boolValue);
  render->FirstChildElement("render-bounding-boxes-option")->QueryBoolAttribute("value", &renderer->optionFrame->getElement(RENDER_BOUNDING_BOXES_OPTION)->boolValue);
  render->FirstChildElement("render-system-bound-option")->QueryBoolAttribute("value", &renderer->optionFrame->getElement(RENDER_SYSTEM_BOUND_OPTION)->boolValue);
  render->FirstChildElement("render-grid-heatmap-option")->QueryBoolAttribute("value", &renderer->optionFrame->getElement(RENDER_GRID_HEATMAP_OPTION)->boolValue);

  QueryFloat3Attribute(render->FirstChildElement("reset-camera-up"), renderer->cameraUp.end());
  renderer->cameraUp.begin() = renderer->cameraUp.end();
  QueryFloat3Attribute(render->FirstChildElement("reset-camera-front"), renderer->cameraFront.end());
  renderer->cameraFront.begin() = renderer->cameraFront.end();
  QueryFloat3Attribute(render->FirstChildElement("reset-camera-position"), renderer->cameraPosition.end());
  renderer->cameraPosition.begin() = renderer->cameraPosition.end();

  real scale = 1.f;
  rayTracingSystem->init(compute, renderer->width() * renderer->height() * 2);
  if (rayTracing->FirstChildElement("simple-camera"))
  {
    rayTracingSystem->camera = new Camera(compute);
    rayTracing->FirstChildElement("simple-camera")->QueryFloatAttribute("scale", &scale);
  }

  rayTracingSystem->camera->width  = renderer->width();
  rayTracingSystem->camera->height = renderer->height();
  rayTracingSystem->camera->setScale(scale);
}

struct ShapeData
{
  int subDivision[3] = {0, 0, 0};
  Real3 dim = {0.f, 0.f, 0.f};
  real size = 0.f;
  real kernelSize = 0.f;
  real mass = 0.f;
};

ShapeData readShape(XMLConstHandle shapeHandle)
{
  ShapeData shape;

  QueryInt3Attribute(shapeHandle.ToElement(), shape.subDivision, "sub-division");
  QueryFloat3Attribute(shapeHandle.ToElement(), shape.dim, "dim");
  shapeHandle.ToElement()->QueryFloatAttribute("size", &shape.size);
  // set default kernel size to be 4x particle size
  shape.kernelSize = shape.size * 4.f;
  shapeHandle.ToElement()->QueryFloatAttribute("kernel-size", &shape.kernelSize);
  shapeHandle.ToElement()->QueryFloatAttribute("mass", &shape.mass);
  float density;
  if (!shapeHandle.ToElement()->QueryFloatAttribute("density", &density))
  {
    shape.mass = density * shape.dim[0] * shape.dim[1];
    if (shape.dim[2] > 0.f)
    {
      shape.mass *= shape.dim[2];
    }
  }

  return shape;
}

void ReaderScene::readEntities(MainSystem* system, XMLElement* entities)
{
  for (const XMLElement* entity = entities->FirstChildElement(); entity; entity = entity->NextSiblingElement())
  {
    PhysicsEntity* newEntity = NULL;
    XMLConstHandle shapeHandle = XMLConstHandle(entity).FirstChildElement("shape");
    ShapeData shape = readShape(XMLConstHandle(entity).FirstChildElement("shape"));

    if (strcmp(entity->Name(), "cloth") == 0)
    {
      Cloth* cloth = new Cloth();

      if (strcmp(shapeHandle.ToElement()->Attribute("type"), "rectangle") == 0)
      {
        cloth->initXY(&shape.dim[0], (uint*)&shape.subDivision[0], shape.mass);
        newEntity = cloth;
      }
    }
    else if (strcmp(entity->Name(), "rigidbody") == 0)
    {
      RigidBody* rigidBody = new RigidBody();

      if (strcmp(shapeHandle.ToElement()->Attribute("type"), "cuboid") == 0)
      {
        rigidBody->initCube(&shape.dim[0], shape.size, shape.mass);
        newEntity = rigidBody;
      }
    }
    else if (strcmp(entity->Name(), "fluid") == 0)
    {
      Fluid* fluid = new Fluid();

      if (strcmp(shapeHandle.ToElement()->Attribute("type"), "cuboid") == 0)
      {
        fluid->initFluid(&shape.dim[0], shape.size, shape.mass, shape.kernelSize);
        newEntity = fluid;
      }
    }

    string identity(entity->Attribute("id"));
    if (identity.size() == 0)
    {
      logComputeError("Entity attribute id required!");
    }

    if (registeredEntities.find(identity) != registeredEntities.end())
    {
      logComputeError("Duplicate id %s, not allowed!", identity.c_str());
    }

    registeredEntities[identity] = system->physicsSystem.registerEntity(newEntity);
  }
}

void ReaderScene::createInstances(MainSystem* system, XMLElement* instances)
{
  for (const XMLElement* instance = instances->FirstChildElement(); instance; instance = instance->NextSiblingElement())
  {
    string identity(instance->Name());

    vector<Matrix4> matrixTransforms;

    int count = 0;
    instance->QueryIntAttribute("count", &count);
    for (int i=0; i<count; i++)
    {
      Matrix matrix;

      QueryTransformElement(instance->FirstChildElement("transform"), matrix);
      matrixTransforms.push_back(matrix[TRANS]);
    }

    system->physicsSystem.addEntityInstance(registeredEntities[identity], count, &matrixTransforms[0]);
  }
}

bool ReaderScene::readFile(MainSystem* system, const char fileName[])
{
  tinyxml2::XMLDocument document;
  string xmlContents = IOInterface::readFile(fileName);
  document.Parse(xmlContents.c_str());

  if (document.Error())
  {
    logComputeError("Error reading scene file!\n%s\n", document.ErrorStr());
    return false;
  }

  XMLElement* scene = document.FirstChildElement("scene");
  if(!scene)
  {
    logComputeError("Scene XML tag not found");
    return false;
  }

  // read settings first
  readSettings(system, scene->FirstChildElement("settings"));

  // then read entities
  readEntities(system, scene->FirstChildElement("entities"));

  // then instanciate entities
  createInstances(system, scene->FirstChildElement("instances"));

  return true;

}
