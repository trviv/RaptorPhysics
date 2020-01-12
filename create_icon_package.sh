#!/bin/bash

if [ ! -f Shared/Resources/AppIcon.svg ]; then
  echo "ERROR: Required input file Shared/Resources/AppIcon.svg does not exist!"
  exit 1;
fi

mkdir -p Shared/Resources/AppIcon.xcassets/AppIcon.appiconset
cp Shared/Resources/Contents.json Shared/Resources/AppIcon.xcassets/AppIcon.appiconset

for i in 16 20 29 32 40 48 50 57 58 60 64 72 76 80 87 88 100 114 120 128 144 152 167 172 180 196 216 256 512 1024; do
  size=$i
  sizeName=$size
  /Applications/Inkscape.app/Contents/MacOS/inkscape Shared/Resources/AppIcon.svg --export-file=Shared/Resources/AppIcon.xcassets/AppIcon.appiconset/$sizeName.png --export-type=png --export-width="$size" --export-height="$size" --without-gui
done
