#!/bin/bash

sed -i -e 's/Listing/Листинг/g' tex/sphinxlatexliterals.sty
sed -i -e 's/chapter{Предговор/chapter\*\{Предговор/g' tex/pinn-skripta.tex 

echo "Done."

