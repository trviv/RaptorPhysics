in vec3 position;
in float radius;
in vec4 transformedGradient;

uniform int fill;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;
out vec3 normal;

void main()
{
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz + transformedGradient.xyz, 1.f) + vec4(0.f, 0.f, fill, 0.f) * 0.0001f;
  normal = normalize(transformedGradient.xyz);

  col = vec4(sign(radius) < 0.0f ? 1.0f : 0.0f, 0.2f, 0.8f, 1.0f);
}
