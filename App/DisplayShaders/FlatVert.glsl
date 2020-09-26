layout (location = 0) in vec3 position;
layout (location = 1) in vec4 particlePos;
layout (location = 2) in vec4 particleCollData;
layout (location = 3) in float density;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform int screenAligned;
uniform float invRestDensity;

out vec4 col;
out vec3 normal;

void main()
{
  float radius = particleCollData.w;

  vec3 billpos = position.xyz * abs(radius);

  if (screenAligned > 0)
  {
    vec3 cameraRight = vec3(modelViewMatrix[0].x, modelViewMatrix[1].x, modelViewMatrix[2].x);
    vec3 cameraUp = vec3(modelViewMatrix[0].y, modelViewMatrix[1].y, modelViewMatrix[2].y);
    billpos = cameraRight * billpos.x + cameraUp * billpos.y;
  }
  else
  {
    int encodedTransformedGradient = floatBitsToInt(particleCollData.x);
    ivec3 magnitude = ivec3(encodedTransformedGradient, encodedTransformedGradient >> 10, encodedTransformedGradient >> 20) & 0x3FF;
    magnitude.x |= (magnitude.x & 0x200) > 0 ? 0xFFFFFC00 : 0;
    magnitude.y |= (magnitude.y & 0x200) > 0 ? 0xFFFFFC00 : 0;
    magnitude.z |= (magnitude.z & 0x200) > 0 ? 0xFFFFFC00 : 0;

    vec3 direction = vec3(magnitude) * (1.f / 511.f);
    vec3 crossDir = normalize(cross(direction, vec3(0.f, 1.f, 0.f)));
    direction = normalize(cross(crossDir, direction));
    billpos = direction * billpos.x + crossDir * billpos.y;
  }

  gl_Position = projectionMatrix * modelViewMatrix * vec4(billpos + particlePos.xyz, 1.0f);
  normal = vec3(0.f, 1.f, 0.f);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.0f, 1.f, 0.125f * (density * invRestDensity));
}
