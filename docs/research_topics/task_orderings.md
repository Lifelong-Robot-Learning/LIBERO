# Task Orderings
We enable 21 randomly sampled task orderings for all 10-task task suites. The task ordering can be specified by the ```data.task_order_index``` command in the following:
```
export CUDA_VISIBLE_DEVICES=GPU_ID && export MUJOCO_EGL_DEVICE_ID=GPU_ID && python lifelong/main.py seed=SEED benchmark_name=BENCHMARK policy=POLICY lifelong=ALGO data.task_order_index=ORDER_ID
```

<div class="admonition note">
<p class="admonition-title">Assitive Note - Specific command generation</p>

<div>
    <form id="commandForm">
        <label for="dropdown-policy">Policy: </label>
        <select id="dropdown-policy" class="command-option"  required>
            <option value="">Select a policy</option>
            <option value="bc_rnn_policy">bc_rnn_policy(ResNet-RNN)</option>
            <option value="bc_transformer_policy">bc_transformer_policy(ResNet-T)</option>
            <option value="bc_vilt_policy">bc_vilt_policy(ViT-T)</option>
        </select>
        <br>
        <label for="dropdown-algorithm">Algorithm:</label>
        <select id="dropdown-algorithm" class="command-option" required>
            <option value="">Select an algorithm</option>
            <option value="base">base (Sequential)</option>
            <option value="er">er (Experience Replay)</option>
            <option value="ewc">ewc (Elastic Weight Consolidation)</option>
            <option value="packnet">packnet (PackNet)</option>
            <option value="single_task">single_task (Single Task)</option>
            <option value="multitask">multitask (Multitask)</option>
        </select>
        <br>
        <label for="dropdown-benchmark">Benchmrak:</label>
        <select id="dropdown-benchmark" class="command-option" required>
            <option value="">Select a benchmark</option>
            <option value="libero_spatial">libero_spatial (LIBERO-Spatial)</option>
            <option value="libero_object">libero_object (LIBERO-Object)</option>
            <option value="libero_goal">libero_goal (LIBERO-Goal)</option>
            <option value="libero_90">libero_90 (90 short-horizon tasks in LIBERO-100)</option>
            <option value="libero_10">libero_10 (LIBERO-Long, 10 long-horizon tasks in LIBERO-100)</option>
        </select>
        <br>
        <label for="gpu-id">GPU Id:</label>
        <input id="gpu-id" class="command-blank" type="number" min="0" step="1" required>
        <br>
        <label for="random-seed">Random Seed</label>
        <input id="random-seed" class="command-blank" type="number" min="0" step="1" required>
        <br>
        <label for="task-order">Task Ordering Index</label>
        <input id="task-order" class="command-blank" type="number" min="0" step="1" required>
    </form>
    <script src="../_static/js/command_task_order_script.js"></script>
</div>
<div id="command">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
</div>
