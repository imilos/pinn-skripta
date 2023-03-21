#
# Preslovljavanje iz cirilice u latinicu
# Koristi se https://pypi.org/project/srtools/ paket.
# Milos Ivanovic 2023. 
#

#!/bin/bash

LATINDIR=./lat/

for FAJL in $(find . -name "*.rst")
do 
    echo "Preslovljavam $FAJL..."
    mkdir -p "$LATINDIR/$(dirname $FAJL)/"
    cat $FAJL | srts --cl > "$LATINDIR/$FAJL"
done

# Preslovi conf.py i promeni jezik
cat conf.py | srts --cl > "$LATINDIR/conf.py"
sed -i -e "s/language = 'sr/language = 'hr/" "$LATINDIR/conf.py"

# Preslov latex-custom.sh i sredi dozvolu
cat latex-custom.sh | srts --cl > "$LATINDIR/latex-custom.sh"
chmod +x "$LATINDIR/latex-custom.sh"

# Kopiraj zajednicke fajlove
cp -r LICENSE refs.bib docutils.conf by-sa.svg _static/ $LATINDIR/

# Kopiraj slike
for SLIKA in $(find . -type f \( -name '*.jpg' -or -name '*.png' \))
do
    echo "Kopiram $SLIKA..."
    mkdir -p "$LATINDIR/$(dirname $SLIKA)/"
    cp $SLIKA "$LATINDIR/$SLIKA"
done

