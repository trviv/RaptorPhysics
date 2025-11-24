/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

in highp vec2 texCoord;
out lowp vec4 col;

uniform sampler2D backgroundTexture;

void main()
{
  col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  if (texCoord.x >= 0.0f && texCoord.x < 1.0f && texCoord.y >= 0.0f && texCoord.y < 1.0f)
  {
    col = texture(backgroundTexture, texCoord);
  }
}
