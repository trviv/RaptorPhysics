/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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
