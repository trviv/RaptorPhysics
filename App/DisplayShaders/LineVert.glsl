/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

layout (location = 0) in vec4 particlePos;
layout (location = 1) in vec4 particleCollData;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;

void main()
{
  int encodedTransformedGradient = floatBitsToInt(particleCollData.x);
  ivec3 magnitude = ivec3(encodedTransformedGradient, encodedTransformedGradient >> 10, encodedTransformedGradient >> 20) & 0x3FF;
  magnitude.x |= (magnitude.x & 0x200) > 0 ? 0xFFFFFC00 : 0;
  magnitude.y |= (magnitude.y & 0x200) > 0 ? 0xFFFFFC00 : 0;
  magnitude.z |= (magnitude.z & 0x200) > 0 ? 0xFFFFFC00 : 0;

  float scale = particleCollData.w;

  if (gl_VertexID>0)
  {
    scale += particleCollData.y;
  }

  col = vec4((magnitude.x != 0) ? 1.f : 0.f, (magnitude.y != 0) ? 1.f : 0.f, 0.f, 1.f);

  vec3 pos = particlePos.xyz + vec3(magnitude) * scale * (1.f / 511.f);

  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.f);
}
