/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "UIList.h"

/*!
@class Class representing user interface button.
*/
class UIElement : public UIObject
{
public:
  enum UIElementType
  {
    Bool,
    String,
    Slider
  };

  bool autoWidth;

private:
  UIElementType type;
  Texture texture;
  void*   font;

  explicit UIElement(const string& identifier, const char* iconFont, ushort iconId, const char* font);

public:
  static float ButtonWidth;
  static float ButtonHeight;

  static string getIconAsString(const char* iconFont = NULL, ushort iconId = 0);

  string  identifier;
  bool    boolValue;
  int     range[2];
  int     rangeValue;

  UIElement(const string& identifier, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const string& text, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const int min, const int max, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  ~UIElement();

  bool render();
};

#endif
