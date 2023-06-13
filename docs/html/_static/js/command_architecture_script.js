function updateCommand() {
    // const dropdown_algo = document.getElementById("dropdown-algorithm").value;
    const dropdown_policy = document.getElementById("dropdown-policy").value;
    const dropdown_benchmark = document.getElementById("dropdown-benchmark").value;
    const gpu_id_element = document.getElementById("gpu-id");
    const gpu_id = parseInt(gpu_id_element.value);

    const random_seed_element = document.getElementById("random-seed");
    const random_seed = parseInt(random_seed_element.value);

    let command = '';
    if (isNaN(gpu_id) || gpu_id < 0) {
        gpu_id_element.setCustomValidity("Input must be a nonnegative integer");
    }
    if (isNaN(random_seed) || random_seed < 0) {
        random_seed_element.setCustomValidity("Input must be a nonnegative integer");
    }
    if (dropdown_policy && gpu_id >= 0 && random_seed >= 0) {
        // Update command conditions based on your requirements
        // command = `Run command for ${dropdown1}-${dropdown2}`;
        command_base = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">base</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_er = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">er</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_ewc = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">ewc</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_packnet = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">packnet</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_single_task = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">single_task</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_multitask = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">multitask</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
        command_agem = `<div class="highlight-default notranslate"><div class="highlight"><pre id="codecell1"><span></span><span class="n"> export</span> <span class="n">CUDA_VISIBLE_DEVICES</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">export</span> <span class="n">MUJOCO_EGL_DEVICE_ID</span><span class="o">=</span><span class="n">${gpu_id}</span> <span class="o">&amp;&amp;</span> <span class="n">python</span> <span class="n">lifelong</span><span class="o">/</span><span class="n">main</span><span class="o">.</span><span class="n">py</span> <span class="n">seed</span><span class="o">=</span><span class="n">${random_seed}</span> <span class="n">benchmark_name</span><span class="o">=</span><span class="n">${dropdown_benchmark}</span> <span class="n">policy</span><span class="o">=</span><span class="n">${dropdown_policy}</span> <span class="n">lifelong</span><span class="o">=</span><span class="n">agem</span></pre><button class="copybtn o-tooltip--left" data-tooltip="Copy" data-clipboard-target="#codecell1"><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <title>Copy to clipboard</title>
        <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
        <rect x="8" y="8" width="12" height="12" rx="2"></rect>
        <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
        </svg>
        </button>
        </div></div>
        `;
    }

    document.getElementById("command-base").innerHTML = command_base;
    document.getElementById("command-er").innerHTML = command_er;
    document.getElementById("command-ewc").innerHTML = command_ewc;
    document.getElementById("command-packnet").innerHTML = command_packnet;
    document.getElementById("command-single-task").innerHTML = command_single_task;
    document.getElementById("command-multitask").innerHTML = command_multitask;
    document.getElementById("command-agem").innerHTML = command_agem;
}

document.getElementById("dropdown-policy").addEventListener("change", function() {
    updateCommand();
});
document.getElementById("dropdown-benchmark").addEventListener("change", updateCommand);    

document.getElementById("gpu-id").addEventListener("change", updateCommand);
document.getElementById("random-seed").addEventListener("change", updateCommand);
