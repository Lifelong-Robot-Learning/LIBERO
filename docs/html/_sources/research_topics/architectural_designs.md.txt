# Architectural Designs
Please follow the following commands to reproduce the study results on vision-language policy architectures.

## Reproducing Results
Fixing the ```ALGO``` to be one of ```[base, er, agem, ewc, packnet]```. Select a policy from
```[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]``` and run:
```
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lifelong/main.py seed=SEED \
                        benchmark_name=BENCHMARK 
                        policy=POLICY \
                        lifelong=ALGO
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
        <!-- <label for="dropdown-algorithm">Algorithm:</label>
        <select id="dropdown-algorithm" class="command-option" required>
            <option value="">Select an algorithm</option>
            <option value="base">base (Sequential)</option>
            <option value="er">er (Experience Replay)</option>
            <option value="ewc">ewc (Elastic Weight Consolidation)</option>
            <option value="packnet">packnet (PackNet)</option>
            <option value="multitask">multitask (Multitask)</option>
        </select>
        <br> -->
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
    </form>
    <script src="../_static/js/command_architecture_script.js"></script>
</div>

<br>
<br>

<p>Sequential (base)</p>
<div id="command-base">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p>Multitask (multitask)</p>
<div id="command-multitask">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p>Single Task (single_task)</p>
<div id="command-single-task">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p>Experience Replay (er)</p>
<div id="command-er">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p>Elastic Weight Consolidation (ewc)</p>
<div id="command-ewc">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p>PackNet (packnet)</p>
<div id="command-packnet">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

<p> Agem (agem)</p>
<div id="command-agem">
<div class="highlight-default notranslate"><div class="highlight"><pre></pre>
</div></div>
</div>
<br>

</div>
