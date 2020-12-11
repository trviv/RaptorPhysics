#include "Material.h"

Material::Material(MaterialTypes type)
{
  diffuse     = Half4(0.f, 0.f, 0.f, 0.f);
  specular    = Half4(0.f, 0.f, 0.f, 0.f);
  emissive    = Half4(0.f, 0.f, 0.f, 0.f);
  parameters  = Half4(0.f, 0.f, 0.f, 0.f);

  flags = 0;
  setMaterialType(*this, type);
}
