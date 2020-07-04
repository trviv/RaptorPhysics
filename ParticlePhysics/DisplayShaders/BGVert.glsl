layout (location = 0) in vec4 pos;
layout (location = 1) in vec2 inTexCoord;

uniform vec4 frameDimensions;
uniform int flipY;
uniform int fillScreen;

out vec2 texCoord;

void main()
{
  vec2 textureUV = vec2(1.0f);

  float displayAspect = frameDimensions.x/frameDimensions.y;
  float textureAspect = frameDimensions.z/frameDimensions.w;

  if ((displayAspect > textureAspect && fillScreen > 0) || (displayAspect <= textureAspect && fillScreen <= 0))
  {
    textureUV.y *= (textureAspect / displayAspect);
  }
  else
  {
    textureUV.x *= (displayAspect / textureAspect);
  }
  
  textureUV *= inTexCoord;
  if (flipY > 0)
  {
    texCoord = vec2(textureUV.x + 1.0f, 1.0f - textureUV.y) * 0.5f;
  }
  else
  {
    texCoord = vec2(textureUV.x + 1.0f, textureUV.y + 1.f) * 0.5f;
  }
  gl_Position = pos;
}
