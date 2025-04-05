path=$(conda info | grep 'envs directories'| sed -E 's/(envs directories)| |://g' | sed 's/envs/etc\/profile.d\/conda.sh/g')
source $path
L=$(conda env list | grep "^solar "|wc -c)
if [ $L -ne 0 ]
then 
    conda env update --file environment.yml
else
    conda env create -f environment.yml
fi 

conda activate solar
echo -n "ipywidgets"
pip install -ve "./required_modules/ipywidgets[test]"

echo -n "jupyter_leaflet"
pip install -ve ./required_modules/jupyter_leaflet

echo -n "osmnx"
pip install -ve ./required_modules/osmnx

jupyter labextension develop ./required_modules/jupyter_leaflet --overwrite


echo -n "ipyleaflet"
pip install -ve ./required_modules/ipyleaflet
