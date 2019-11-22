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

  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz * pos.w + pos.xyz, 1.0f);
  normal = normalize(position.xyz);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.2f, 0.8f, 1.0f);
}
