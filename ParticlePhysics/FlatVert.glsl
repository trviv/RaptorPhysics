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

  vec3 billpos = position.xyz * abs(radius);

  vec3 cameraRight = vec3(modelViewMatrix[0].x, modelViewMatrix[1].x, modelViewMatrix[2].x);
  vec3 cameraUp = vec3(modelViewMatrix[0].y, modelViewMatrix[1].y, modelViewMatrix[2].y);

  billpos = cameraRight * billpos.x + cameraUp.xyz * billpos.y;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(billpos + particlePos.xyz, 1.0f);
  normal = vec3(0.f, 1.f, 0.f);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.0f, 1.f, 0.5f);
}
