in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleCollData;

out vec4 col;
out vec3 normal;

#define texureWidth     128
#define texureWidthExp  7

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & (texureWidth - 1), gl_InstanceID >> texureWidthExp), 0);
  float radius = texelFetch(particleCollData, ivec2(gl_InstanceID & (texureWidth - 1), gl_InstanceID >> texureWidthExp), 0).w;

  pos.w = abs(radius);

  vec3 billpos = position.xyz * pos.w;

  vec3 cameraRight = vec3(modelViewMatrix[0].x, modelViewMatrix[1].x, modelViewMatrix[2].x);
  vec3 cameraUp = vec3(modelViewMatrix[0].y, modelViewMatrix[1].y, modelViewMatrix[2].y);

  billpos = cameraRight * billpos.x + cameraUp.xyz * billpos.y;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(billpos + pos.xyz, 1.0f);
  normal = vec3(0.f, 1.f, 0.f);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.0f, 1.f, 0.5f);
}
