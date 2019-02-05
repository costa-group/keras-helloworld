How to Install
==============

Go to repo `costa-group/iRankFinder`

run: 
```
./iRankFinder/installer/install_externals.sh
./iRankFinder/installer/install_modules.sh -m dev -b unstable -p PATH_TO_INSTALL```

then clone this repo and run using the command:

```
python3 main.py -f FILEPATH -v VERBOSITY_LEVEL [--no-split]```

if you use --no-split parameter it will use one phi per node instead of one phi per transition.


