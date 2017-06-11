for file in ./scene*.png; do
  convert "$file" -rotate 180 "${file%.png}".png
done
