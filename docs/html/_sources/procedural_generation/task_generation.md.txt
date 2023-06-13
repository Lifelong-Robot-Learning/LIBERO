# Task Generation


In this section, we describe how to generate a task in LIBERO. The procedure consists of three steps: 1) Specify a scene from the <a href="../procedural_generation/scene_creation.html">Scene Creation</a> step; 2) Specify initial state distributions; 3) Specify goals and language instructions.

## An example
We present an example of creating a task in LIBERO for intuitive understanding. 

The creation of a task given a pre-defined scene can be easily created using a function. A complete example of generating a task can be found in the provided [notebook](notebooks/procedural_creation_walkthrough.ipynb)

### Layout and Initial State Distributions

Here is an example t odefine the layout, objects, and the initial state dsistributions. 

```python
@register_mu(scene_type="kitchen")
class KitchenSceneExample(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }
        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )
    def define_regions(self):
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, -0.30], 
                                 region_name="wooden_cabinet_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.01,
                                 yaw_rotation=(np.pi, np.pi))
        )
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0., 0.0], 
                                 region_name="akita_black_bowl_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.025)
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, 0.25], 
                                 region_name="plate_init_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.025)
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region")]
        return states
```

### Goal and Language

Once we define the initial state distributions, we can define the task goal and the language specifications. Here is an example how to define them. Notice that `objects_of_interest` is an optional field, in case you want to modify the LIBERO problem classes to keep track of the internal simulation states of some certrain objects.

```python
# kitchen_scene_example
scene_name = "kitchen_scene_example"
language = "open the top drawer of the cabinet and put the bowl in it"
register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
                    goal_states=[("Open", "wooden_cabinet_1_top_region"), ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region")]
)
```