// CS336 Lecture 5 — GPUs
// 20 questions covering: GPU anatomy (SMs, warps, threads/blocks), the memory
// hierarchy (registers, shared memory, L2, global memory), the roofline model
// and memory-bound vs compute-bound work, tensor cores and low precision, the
// performance tricks (divergence, fusion, recomputation, coalescing, tiling),
// wave quantization, and FlashAttention.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture5"] = {
  title: "Lecture 5 — GPUs",
  questions: [
    // ---------- GPU anatomy ----------
    {
      type: "single",
      question: "CPUs and GPUs embody opposite design philosophies. What does a GPU primarily optimize for?",
      options: [
        "Minimizing the latency of a single thread of execution",
        "Maximizing single-core clock speed with deep branch prediction",
        "Maximizing total throughput across many parallel threads",
        "Minimizing power draw for mobile and embedded workloads"
      ],
      correct: [2],
      explanation: "A CPU spends its silicon on large caches, branch prediction, and control logic to make a few threads finish quickly — it optimizes latency. A GPU instead packs in many tiny ALUs with much less support for branching, optimizing throughput: total data processed across many, many threads. ML workloads, which are giant regular array computations, are exactly what this trade favors."
    },
    {
      type: "single",
      question: "What is a warp in NVIDIA's execution model?",
      options: [
        "32 consecutively numbered threads that execute in lockstep",
        "A group of 32 SMs sharing one partition of the L2 cache",
        "A 32 KB region of shared memory owned by one thread block",
        "A queue of 32 pending kernel launches awaiting scheduling"
      ],
      correct: [0],
      explanation: "Threads always execute in warps of 32 consecutively numbered threads, and all threads in a warp execute the same instruction at the same time (SIMT). Their memory accesses also happen together, which is why coalescing is defined at the warp level and why control-flow divergence within a warp is costly."
    },
    {
      type: "multi",
      question: "Which statements about the CUDA execution and memory model are correct? (Select all that apply)",
      options: [
        "Each thread block runs on a single SM with its own shared memory",
        "All threads execute the same instructions, each on different inputs (SIMT)",
        "A thread block always contains exactly one warp of 32 threads",
        "Threads in the same block can cooperate through shared memory",
        "Threads in different blocks share registers with each other"
      ],
      correct: [0, 1, 3],
      explanation: "The three players are threads, blocks, and warps: threads do the work under the SIMT model, blocks are groups of threads that each run on one streaming multiprocessor with its own shared memory, and warps are the 32-thread execution groups. Registers are private to each thread, blocks contain many warps, and information crossing blocks must go through slow global memory."
    },
    {
      type: "single",
      question: "Ordered from fastest/smallest to slowest/largest, what is the GPU memory hierarchy?",
      options: [
        "Shared memory, registers, L2 cache, global memory (HBM)",
        "Registers, shared memory, L2 cache, global memory (HBM)",
        "L2 cache, registers, shared memory, global memory (HBM)",
        "Registers, L2 cache, global memory (HBM), shared memory"
      ],
      correct: [1],
      explanation: "The closer the memory is to the SM, the faster it is: L1 and shared memory sit inside the SM, L2 is on die, and global memory is the chips next to the GPU. The latency gap is large — roughly 20-30 cycles for shared/L1 versus about 200 for L2 and about 290 for global memory — which is why respecting the hierarchy is the core of GPU optimization."
    },
    {
      type: "single",
      question: "What is shared memory on a GPU?",
      options: [
        "Host RAM that the GPU accesses remotely over the PCIe bus",
        "The portion of HBM reserved for inter-kernel communication",
        "A cache managed automatically by hardware, invisible to the programmer",
        "Programmer-managed SRAM on the SM, visible to one block's threads"
      ],
      correct: [3],
      explanation: "Shared memory is a small, programmer-controlled scratchpad of SRAM physically located on each SM, accessible by all threads of the block running there. Unlike a hardware cache, you decide what to stage in it — which is exactly what tiled matmul kernels exploit to reuse operands many times per global-memory load."
    },
    {
      type: "multi",
      question: "Which statements about global memory (DRAM) versus on-chip SRAM are accurate? (Select all that apply)",
      options: [
        "Global memory offers far more capacity — tens of GB versus tens of MB on chip",
        "SRAM is roughly 8x faster to access than DRAM",
        "SRAM costs on the order of 100x more per byte than DRAM",
        "Global memory is faster to access than the registers inside an SM"
      ],
      correct: [0, 1, 2],
      explanation: "SRAM (shared and cache memory) is about 8x faster than DRAM but roughly 100x more expensive, which is why chips carry only tens of megabytes of it next to tens of gigabytes of global memory. Registers remain the fastest storage of all — nothing in the hierarchy beats them, least of all off-chip DRAM."
    },
    // ---------- Compute vs memory bound ----------
    {
      type: "single",
      question: "What is operational (arithmetic) intensity, the key quantity in roofline analysis?",
      options: [
        "Peak FLOP/s divided by achieved FLOP/s",
        "FLOPs performed per byte of memory traffic",
        "Bytes transferred per second from global memory",
        "The fraction of SMs active during a kernel"
      ],
      correct: [1],
      explanation: "Operational intensity = total FLOPs / total bytes moved. It tells you whether a kernel can possibly saturate the compute units: if intensity is too low, the memory system becomes the bottleneck no matter how fast the ALUs are. The lecture's guiding question for performance is precisely how to avoid being memory bound."
    },
    {
      type: "single",
      question: "In the roofline model, what determines a kernel's attainable performance?",
      options: [
        "The minimum of peak FLOP/s and bandwidth times intensity",
        "Always the peak FLOP/s of the tensor cores on the device",
        "The maximum of compute throughput and memory throughput",
        "The number of thread blocks launched relative to SM count"
      ],
      correct: [0],
      explanation: "The roofline plots attainable FLOP/s against operational intensity: attainable FLOP/s $= \\min(\\text{peak compute},\\ \\text{bandwidth} \\times \\text{intensity})$. Low-intensity kernels sit on the slanted bandwidth roof (memory-bound); high-intensity kernels like dense matmul reach the flat compute roof (compute-bound). It is a one-line model that explains most GPU performance puzzles."
    },
    {
      type: "multi",
      question: "Which of these operations are typically memory-bound on a GPU? (Select all that apply)",
      options: [
        "Elementwise ops like ReLU, bias add, or scaling",
        "A large square matrix multiplication",
        "Layer normalization over activations",
        "Softmax over attention scores"
      ],
      correct: [0, 2, 3],
      explanation: "Pointwise ops, normalizations, and softmax do only a handful of FLOPs per element loaded — the lecture's ReLU example does 1 FLOP while moving 8 bytes in FP32 — so bandwidth caps their speed. Large dense matmuls are the exception: they perform $O(n^3)$ FLOPs on $O(n^2)$ data, so intensity grows with size and they can saturate the compute units."
    },
    {
      type: "single",
      question: "A kernel's operational intensity puts it on the slanted part of the roofline, left of where the two roofs meet. What does this imply?",
      options: [
        "The kernel is compute-bound and should reduce its FLOP count",
        "The kernel has too many threads and should shrink its grid",
        "The kernel is memory-bound; faster ALUs alone would not speed it up",
        "The kernel is bound by kernel-launch overhead on the host"
      ],
      correct: [2],
      explanation: "On the slanted roof, performance scales with bandwidth times intensity, so the wins come from moving fewer bytes — fusion, tiling, coalescing, lower precision — not from more compute. This matters more every year because compute (especially matmul) has scaled much faster than memory, making it hard to keep the compute units fed."
    },
    // ---------- Tensor cores & precision ----------
    {
      type: "single",
      question: "What are tensor cores?",
      options: [
        "Extra SMs enabled only when using the cuDNN library",
        "Specialized matrix multiplication circuits inside the GPU",
        "Cores dedicated to moving tensors between HBM and shared memory",
        "General-purpose FP64 units optimized for scientific computing"
      ],
      correct: [1],
      explanation: "Tensor cores, introduced in the V and T series GPUs, are specialized matrix multiplication circuits that multiply low-precision (e.g., 16-bit) inputs and accumulate the products in FP32. They make matmuls more than 10x faster than other floating-point ops, which is why deep learning performance hinges on whether your operation maps onto them."
    },
    {
      type: "multi",
      question: "Why does dropping from FP32 to lower precision (FP16/BF16) speed up GPU workloads? (Select all that apply)",
      options: [
        "Tensor cores have substantially higher peak FLOP/s at lower precision",
        "Smaller elements halve the bytes moved, improving arithmetic intensity",
        "Lower precision makes gradient descent converge in fewer steps",
        "More values fit in registers, shared memory, and caches"
      ],
      correct: [0, 1, 3],
      explanation: "Fewer bits means fewer bits to move: the lecture's ReLU example drops from 8 bytes per FLOP in FP32 to 4 in FP16, and everything from registers to caches holds twice as much. Tensor cores also run fastest on low-precision inputs. It does nothing for optimization itself — indeed reductions, large sums, and functions like exp still need FP32 or BF16 to stay accurate."
    },
    // ---------- Performance techniques ----------
    {
      type: "single",
      question: "What is memory coalescing?",
      options: [
        "Merging several small kernels into one to cut launch overhead",
        "Compressing tensors in HBM before transferring them to the SM",
        "Defragmenting the CUDA memory allocator between iterations",
        "Threads of a warp accessing addresses within one DRAM burst"
      ],
      correct: [3],
      explanation: "DRAM is read in burst mode — each read returns many consecutive bytes because of the slow per-row copy to the sense amplifier. Accesses are coalesced when all 32 threads of a warp fall within the same burst, so one transaction serves everyone. In a row-major matmul, threads striding along rows break this and waste most of the available bandwidth."
    },
    {
      type: "single",
      question: "Why do fast matmul kernels tile the computation into shared memory?",
      options: [
        "Loaded tiles get reused many times, cutting global memory traffic",
        "Shared memory is the only place tensor cores can write results",
        "Tiling reduces the total number of FLOPs the matmul must perform",
        "It lets different thread blocks exchange partial results directly"
      ],
      correct: [0],
      explanation: "By staging tiles of the inputs in shared memory and computing the matmul in phases, repeated reads hit fast shared memory instead of global memory, and the loads can be coalesced. With tile size $T$, each input is read $N/T$ times from global memory instead of $N$ — a factor-$T$ reduction in global memory access."
    },
    {
      type: "single",
      question: "In the recomputation trick, why can throwing away activations and recomputing them in the backward pass be faster?",
      options: [
        "It reduces the total FLOP count of the backward pass",
        "It trades cheap extra FLOPs for fewer total memory accesses",
        "It lets gradients skip layers whose activations were discarded",
        "Discarded activations are automatically stored at lower precision"
      ],
      correct: [1],
      explanation: "Storing and reloading every intermediate activation is bandwidth-heavy work with very low arithmetic intensity. In the lecture's stacked-sigmoid example, recomputing the activations during the backward pass needs only $5/8$ of the naive version's memory accesses. Since compute has scaled much faster than memory, trading FLOPs for memory traffic is often optimal."
    },
    {
      type: "multi",
      question: "Which statements about control (warp) divergence are true? (Select all that apply)",
      options: [
        "It occurs when threads within one warp take different branch paths",
        "The hardware executes each divergent path serially, masking inactive threads",
        "Divergence between different warps carries the same serialization penalty",
        "Branching on values that vary per-thread within a warp risks divergence"
      ],
      correct: [0, 1, 3],
      explanation: "Under SIMT, every thread in a warp executes the same instruction, so an if/else that splits a warp forces the SM to run both paths one after the other with non-participating threads masked off, wasting lanes. Conditionals themselves are fine — the overhead comes from the execution model. Different warps are scheduled independently, so divergence across warps costs nothing."
    },
    // ---------- Wave quantization & FlashAttention ----------
    {
      type: "single",
      question: "Why do practitioners pick matrix dimensions that are multiples of 8 or 16 (padding vocabularies, hidden sizes, etc.)?",
      options: [
        "Odd dimensions cause numerical overflow in FP16 accumulation",
        "CUDA cannot launch kernels whose grid sizes are not powers of two",
        "It reduces the asymptotic FLOP count of the underlying matmul",
        "They align with tile sizes and DRAM bursts, avoiding partial tiles"
      ],
      correct: [3],
      explanation: "Tile sizes that do not divide the matrix dimensions leave partially filled tiles and low utilization, and loading tiles is only fast when the matrix rows align with DRAM bursts — misaligned dimensions force padding or uncoalesced access. Hence the folk rule of padding sizes up to a friendly multiple so the matmul maps cleanly onto full, aligned tiles."
    },
    {
      type: "single",
      question: "Increasing a square matmul's size from 1792 to 1793 sharply lowers throughput on an A100. Why?",
      options: [
        "The 1793-sized matrix spills out of the A100's 40 MB L2 cache",
        "Odd dimensions disable the tensor cores, falling back to CUDA cores",
        "With $256 \\times 128$ tiles, the count jumps from 98 to 120, past the 108 SMs",
        "cuBLAS pads every matrix dimension up to the next power of two"
      ],
      correct: [2],
      explanation: "This is wave quantization: at size 1792 the kernel makes $7 \\times 14 = 98$ tiles, which the A100's 108 SMs can run in a single wave. At 1793 it needs $8 \\times 15 = 120$ tiles, so a second wave launches with most SMs idle, and utilization drops sharply until the size grows enough to fill the wave again."
    },
    {
      type: "single",
      question: "Softmax normally needs a whole row of attention scores. How does FlashAttention still compute attention tile-by-tile?",
      options: [
        "It approximates the softmax with a sparse low-rank factorization",
        "It first materializes the full score matrix in global memory",
        "It skips normalization during training and rescales outputs afterward",
        "An online softmax keeps a running max and a telescoping normalizer per tile"
      ],
      correct: [3],
      explanation: "FlashAttention combines tiling of the K, Q, V matmuls with the online softmax trick: as each tile is processed it updates the running maximum and rescales the partial sums via a telescoping sum, fusing the exponential along the way. The result is the exact softmax computed in fast on-chip memory — and the backward pass recomputes tile-by-tile rather than storing the score matrix."
    },
    {
      type: "multi",
      question: "A kernel sits well below the roofline's bandwidth roof. Which changes could raise its arithmetic intensity or effective bandwidth? (Select all that apply)",
      options: [
        "Fuse adjacent elementwise ops so intermediates never round-trip through global memory",
        "Restructure loads so each warp accesses contiguous, coalesced addresses",
        "Launch the same kernel with a higher thread count but identical memory traffic",
        "Store activations in lower precision to move fewer bytes"
      ],
      correct: [0, 1, 3],
      explanation: "These are the lecture's core tricks: fusion removes whole rounds of global-memory writes and reads, coalescing makes the bytes you do move arrive in full bursts, and lower precision shrinks the bytes per element — all either raise FLOPs-per-byte or utilization of the memory system. Merely adding threads without changing the traffic pattern leaves the same bytes as the bottleneck."
    }
  ]
};
