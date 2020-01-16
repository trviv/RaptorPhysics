#!/bin/bash

if [ ! -f Shared/Resources/AppIcon.svg ]; then
  echo "ERROR: Required input file Shared/Resources/AppIcon.svg does not exist!"
  exit 1;
fi

mkdir -p Shared/Resources/AppIcon.xcassets/AppIcon.appiconset
cp Shared/Resources/Contents_icon.json Shared/Resources/AppIcon.xcassets/AppIcon.appiconset/Contents.json

for i in 16 20 29 32 40 48 50 57 58 60 64 72 76 80 87 88 100 114 120 128 144 152 167 172 180 196 216 256 512 1024; do
  size=$i
  sizeName=$size
  /Applications/Inkscape.app/Contents/MacOS/inkscape Shared/Resources/AppIcon-1.svg --export-file=Shared/Resources/AppIcon.xcassets/AppIcon.appiconset/$sizeName.png --export-type=png --export-width="$size" --export-height="$size" --without-gui
done

/Applications/Inkscape.app/Contents/MacOS/inkscape Shared/Resources/LaunchScreen.svg --export-file=Shared/Resources/LaunchScreen.png --export-type=png --export-width=2048 --export-height=2048 --without-gui

# mkdir -p Shared/Resources/AppIcon.xcassets/LaunchImage.launchimage
# cp Shared/Resources/Contents_launch.json Shared/Resources/AppIcon.xcassets/LaunchImage.launchimage/Contents.json
#
# width=(  2688 1242 2436 1125 2208 1242 1792 828  750 )
# height=( 1242 2688 1125 2436 1242 2208 828  1792 1334)
# len=${#width[@]}
# for (( i=0; i<$len; i++ )); do
#   /Applications/Inkscape.app/Contents/MacOS/inkscape Shared/Resources/LaunchScreen.svg --export-file=Shared/Resources/AppIcon.xcassets/LaunchImage.launchimage/"${width[$i]}"x"${height[$i]}".png --export-type=png --export-width="${width[$i]}" --export-height="${height[$i]}" --without-gui
# done
