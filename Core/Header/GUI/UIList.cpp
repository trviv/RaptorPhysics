#include "UIList.h"

UIList::~UIList()
{
  for (int i=0; i<size(); i++)
  {
    delete at(i);
  }
}

void UIList::alignByAxis(UIObject* object, int axis)
{
  UIObject::UIObjectAlign alignFloat  = (axis==0) ? UIObject::FloatX : UIObject::FloatY;
  UIObject::UIObjectAlign alignDir    = (axis==0) ? UIObject::Right : UIObject::Bottom;

  object->pos[axis] = (object->alignment & alignDir) ? ImGui::GetIO().DisplaySize[axis] - object->size[axis] : 0.f;

  // don't proceed if float not specified
  if ((object->alignment & alignFloat) == 0)
  {
    return;
  }

  if (object->alignment & alignDir)
  {
    for (auto i : *this)
    {
      if ((i->alignment & alignDir) == 0)
        continue;

      if (i == object)
        break;

      float j = i->pos[axis] - object->size[axis];
      if (j < object->pos[axis])
      {
        object->pos[axis] = j;
      }
    }
  }
  else
  {
    for (auto i : *this)
    {
      if ((i->alignment & alignDir) != 0)
        continue;

      if (i == object)
        break;

      float j = i->pos[axis] + i->size[axis];
      if (j > object->pos[axis])
      {
        object->pos[axis] = j;
      }
    }
  }
}

void UIList::render()
{
  // loop to render all objects in the list
  for (const auto object : *this)
  {
    // if aligment option specified
    if (object->alignment)
    {
      alignByAxis(object, 0);
      alignByAxis(object, 1);
    }
    object->render();
  }
}
