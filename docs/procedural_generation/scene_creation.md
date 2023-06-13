# Scene Creation

We have created multiple scenes for benchmarking purpose. In this section, we will walk you through the process to create a new scene to benefit your own purpose.

## Prepare mesh files

While the creation of mesh files are not the main focus of this repo, this process is critical to unlock task diversity. Here is a list of resources that you might find helpful:
- [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)

You can also take a look files located under `libero/libero/assets`.

## Create a problem class

Problem class defines a domain. A domain is something you will define initial state distributions and task goals on top of it. All the problem definitions in the LIBERO is located in `libero/libero/envs/problems`. We also provide a [template](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/templates/problem_class_template.py) that make it easy for users to create their own problem class.