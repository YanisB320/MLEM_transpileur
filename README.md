# MLEM_transpileur

## HOW TO USE

You can run the following command at the root to launch the project:
```shell
./launch.sh
```

This script does the following stuff:
- create virtual env and install ```requirements.txt```
- launch linear regression training
- generate a C file for prediction
- compile C file
- run C file to display a prediction on hard coded to data
