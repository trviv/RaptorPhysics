#include "Material.h"

Material::Material(MaterialTypes type)
{
  diffuse     = Half4(0.f, 0.f, 0.f, 0.f);
  specular    = Half4(0.f, 0.f, 0.f, 0.f);
  emissive    = Half4(0.f, 0.f, 0.f, 0.f);
  components  = Half4(0.f, 0.f, 0.f, 0.f);

  setMaterialType(*this, type);
}
