uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleCollData;

out vec4 col;

#define texureWidth     128
#define texureWidthExp  7

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & (texureWidth - 1), gl_InstanceID >> texureWidthExp), 0);
  col = vec4(1.f, 0.f, 0.f, 1.f);

  if (gl_VertexID>0)
  {
    vec4 particleData = texelFetch(particleCollData, ivec2(gl_InstanceID & (texureWidth - 1), gl_InstanceID >> texureWidthExp), 0);
    int encodedTransformedGradient = floatBitsToInt(particleData.x);
    ivec3 magnitude = ivec3(encodedTransformedGradient, encodedTransformedGradient >> 10, encodedTransformedGradient >> 20) & 0x3FF;
    magnitude.x |= (magnitude.x & 0x200) > 0 ? 0xFFFFFC00 : 0;
    magnitude.y |= (magnitude.y & 0x200) > 0 ? 0xFFFFFC00 : 0;
    magnitude.z |= (magnitude.z & 0x200) > 0 ? 0xFFFFFC00 : 0;

    vec3 gradientVector = vec3(magnitude) * (1.f / 511.f);
    gradientVector *= 2.f * particleData.y;

    pos += vec4(gradientVector.xyz, 1.0f);

    col = vec4((magnitude.x != 0) ? 1.f : 0.f, (magnitude.y != 0) ? 1.f : 0.f, 0.f, 1.f);
  }

  pos = projectionMatrix * modelViewMatrix * vec4(pos.xyz, 1.f);

  gl_Position = pos;
}
