/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 particleData;

uniform float fillShader;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;
out vec3 normal;

void main()
{
  int encodedTransformedGradient = floatBitsToInt(particleData.x);
  ivec3 magnitude = ivec3(encodedTransformedGradient, encodedTransformedGradient >> 10, encodedTransformedGradient >> 20) & 0x3FF;
  magnitude.x |= (magnitude.x & 0x200) > 0 ? 0xFFFFFC00 : 0;
  magnitude.y |= (magnitude.y & 0x200) > 0 ? 0xFFFFFC00 : 0;
  magnitude.z |= (magnitude.z & 0x200) > 0 ? 0xFFFFFC00 : 0;

  vec3 gradientVector = vec3(magnitude) * (1.f / 511.f);
  gradientVector *= particleData.y;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz + gradientVector.xyz, 1.f) + vec4(0.f, 0.f, fillShader, 0.f) * 0.0001f;
  normal = normalize(gradientVector.xyz);

  col = vec4(sign(particleData.w) < 0.0f ? 1.0f : 0.0f, 0.2f, 0.8f, 1.0f);
}
