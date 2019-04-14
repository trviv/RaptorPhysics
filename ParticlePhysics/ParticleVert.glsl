#version 330

layout(location=0) in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleCol;
uniform sampler2D particleSDFGrad;

in int gl_InstanceID;
out vec4 col;
out vec3 normal;

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & 0xF, gl_InstanceID >> 4), 0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz * pos.w + pos.xyz, 1.0f);
  normal = normalize(position.xyz);

  vec4 gradient = texelFetch(particleSDFGrad, ivec2(gl_InstanceID & 0xF, gl_InstanceID >> 4), 0);
  col = vec4(sign(gradient.w) < 0.0f ? 1.0f : 0.0f, 0.2f, 0.8f, 1.0f);
}