in vec4 pos;
in vec2 inTexCoord;
out vec2 texCoord;

void main()
{
  texCoord = vec2(inTexCoord.x + 1.0f, 1.f - inTexCoord.y) * 0.5f;
  gl_Position = pos;
}
