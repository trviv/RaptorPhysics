layout (location = 0) in vec3 position;
layout (location = 1) in vec4 particlePos;
layout (location = 2) in vec4 particleCollData;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;
out vec3 normal;

void main()
{
  float radius = particleCollData.w;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz * abs(radius) + particlePos.xyz, 1.0f);
  normal = normalize(position.xyz);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.2f, 0.8f, 1.0f);
}
