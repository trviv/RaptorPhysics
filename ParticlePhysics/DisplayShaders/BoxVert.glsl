layout (location = 0) in vec3 position;
layout (location = 1) in vec4 minMax[2];

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;

void main()
{
  vec4 pos = projectionMatrix * modelViewMatrix * vec4(((minMax[1].xyz + minMax[0].xyz) + position * (minMax[1].xyz - minMax[0].xyz)) * 0.5f, 1.f);

  gl_Position = pos;
  col = vec4(0.0f, 0.f, 0.f, 0.50f);
}
