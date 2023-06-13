function updateCommand() {
    const dropdown_algo = document.getElementById("dropdown-algorithm").value;
    const dropdown_policy = document.getElementById("dropdown-policy").value;
    const dropdown_benchmark = document.getElementById("dropdown-benchmark").value;
    const gpu_id_element = document.getElementById("gpu-id");
    const gpu_id = parseInt(gpu_id_element.value);

    const random_seed_element = document.getElementById("random-seed");
    const random_seed = parseInt(random_seed_element.value);

    const task_order_element = document.getElementById("task-order");
    const task_order = parseInt(task_order_element.value);

    let command = '';
    if (isNaN(gpu_id) || gpu_id < 0) {
        gpu_id_element.setCustomValidity("Input must be a nonnegative integer");
    }
    if (isNaN(random_seed) || random_seed < 0) {
        random_seed_element.setCustomValidity("Input must be a nonnegative integer");
    }
    if (isNaN(task_order) || task_order < 0) {
        task_order_element.setCustomValidity("Input must be a nonnegative integer");
    }
    if (dropdown_policy && dropdown_algo && gpu_id >= 0 && random_seed >= 0 && task_order >= 0) {
        // Update command conditions based on your requirements
        // command = `Run command for ${dropdown1}-${dropdown2}`;
        command = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">${dropdown_algo}</span> <span class="n">task_order</span><span class="o">=</span><span class="n">${task_order}</span> </pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
    }

    document.getElementById("command").innerHTML = command;
}

document.getElementById("dropdown-policy").addEventListener("change", function() {
    updateCommand();
});
document.getElementById("dropdown-algorithm").addEventListener("change", updateCommand);    
document.getElementById("dropdown-benchmark").addEventListener("change", updateCommand);    

document.getElementById("gpu-id").addEventListener("change", updateCommand);
document.getElementById("random-seed").addEventListener("change", updateCommand);
document.getElementById("task-order").addEventListener("change", updateCommand);
