#!/bin/bash
echo "Renaming images to sequential numbers"
cd ../data/glove/rain
a=1
for i in *.png; do
  new=$(printf "%04d.png" "$a")
  mv -i -- "$i" "$new"
  let a=a+1
done

# cd ../style
# a=1
# for j in *.png; do
#   new=$(printf "%04d.png" "$a")
#   mv -i -- "$j" "$new"
#   let a=a+1
# done
