# Optional object densities

## Ready-to-run reference profile

From the LIBERO repository root, opt in to a reference profile covering all 39 registered non-articulated object categories:

```bash
export LIBERO_OBJECT_DENSITY_CONFIG="$PWD/docs/reference_object_densities.json"
```

Then run LIBERO normally in the same shell. The environment variable is read when objects are constructed; unset it to use the released benchmark parameters. This profile is an **unmeasured sensitivity-study scenario**, not a calibrated correction or a new benchmark default. It changes the dynamics, so scores obtained with it must be reported separately from official LIBERO scores.

The [mass audit](reference_object_density_audit.csv) lists the original and reference compiled collision-body masses, object dimensions, and the scenario assumed for each category. For the 36 categories with assets in this repository, each collision geom specifies `density="100"` kg/m³ (0.1 g/cm³). This is a *density*, not a 0.1 g object mass: each object's mass depends on its collision-geom volume. The reference densities were chosen to give nominal masses appropriate to the small model dimensions; they are **effective densities for the simplified collision geometry**, not measurements of real material density. For example, the 8 cm `white_bowl` needs 2877 kg/m³ to reach about 60 g because its collision geoms occupy little volume.

The registry also contains `cherries`, `corn`, and `mayo`, but their XML assets are absent from this repository revision. The profile includes provisional 1000 kg/m³ entries for key coverage; no mass can be calculated or tested for those three until their assets are supplied. Articulated fixtures have separate inertial definitions and are outside this profile. The 36 available objects compiled successfully in an isolated MuJoCo collision-geom model with both the original and reference densities; full benchmark rollout validation is still pending.

## Custom profile

Set `LIBERO_OBJECT_DENSITY_CONFIG` to a JSON file mapping a movable object category or instance to a density in **kg/m³**:

```json
{
  "porcelain_mug": 800,
  "porcelain_mug_1": 950
}
```

```bash
export LIBERO_OBJECT_DENSITY_CONFIG=/absolute/path/object_densities.json
```

An instance entry takes priority over its category entry. Objects absent from the file retain their existing XML. The override changes only collision geoms (group 0) before the task XML is assembled and MuJoCo compiles it. MuJoCo derives mass and inertia from those geoms when no explicit mass or inertial element overrides them; the loader rejects an override on an object with either explicit setting. The option does not change visual meshes, contact geometry, friction, or fixtures.

The file is read once per path in each process. Record its contents and the effective compiled body masses with any results. Changing density changes task dynamics and can invalidate comparisons with previously reported benchmark scores. The values above are format examples, **not measured or recommended physical densities**.
