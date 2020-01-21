in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec4 systemMin;
uniform float maxRadius;
uniform int gridSize;
uniform float totalParticles;

uniform highp isampler2D gridCellParticleCount;

out vec4 col;

void main()
{
  ivec4 count = texelFetch(gridCellParticleCount, ivec2((gl_InstanceID >> 2) & 0x7F, (gl_InstanceID >> 2) >> 7), 0);
  ivec4 countPrev;
  if (gl_InstanceID > 0)
  {
    countPrev = texelFetch(gridCellParticleCount, ivec2(((gl_InstanceID-1) >> 2) & 0x7F, ((gl_InstanceID-1) >> 2) >> 7), 0);
  }
  else
  {
    countPrev = ivec4(0, 0, 0, 0);
  }
  count -= ivec4(countPrev.w, count.xyz);

  vec3 index = vec3(ivec3(gl_InstanceID, gl_InstanceID / gridSize, gl_InstanceID / (gridSize * gridSize)) & (gridSize - 1));
  vec3 vertexPos = ((position + 1.f) * 0.5f + index) * maxRadius;
  vec4 pos = projectionMatrix * modelViewMatrix * vec4(vertexPos + systemMin.xyz, 1.f);

  col = vec4(0.f);

  if (count[gl_InstanceID & 3] > 0)
  {
    col = vec4(1.f , 0.f, 0.f, .02f * float(count[gl_InstanceID & 3]));
  }
  else
  {
    pos /= 0.f;
  }

  gl_Position = pos;
}
