in highp vec2 texCoord;
out lowp vec4 col;

uniform sampler2D backgroundTexture;

void main()
{
  col = texture(backgroundTexture, texCoord);
}
