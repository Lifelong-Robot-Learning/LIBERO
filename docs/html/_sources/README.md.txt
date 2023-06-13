# Documentation Compilation

To compile the documentation, first of all make sure the installation of requirements:

```shell
    pip install -r requirements-docs.txt
```

Then, make sure you are in the folder `docs`. Do the compilation:

```shell
   make html
```

If you make changes to the file structures, you would need to delete the `html` folder and re-compiile the documentation again.

```shell
   rm -rf html && make html
```

## Instructions on modifying command generation

The command generation is supported by the js files located under `_static/js` folder.


