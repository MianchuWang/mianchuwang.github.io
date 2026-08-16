// CS336 Lecture 6 — Kernels and Triton
// 20 questions covering: correct GPU benchmarking (synchronization, warmup),
// profiling with the PyTorch profiler, the GPU execution model (waves, arithmetic
// intensity), kernel fusion and torch.compile, the Triton block-level programming
// model and PTX, the fused softmax kernel, and tiled matrix multiplication.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture6"] = {
  title: "Lecture 6 — Kernels & Triton",
  questions: [
    // ---------- Benchmarking ----------
    {
      type: "single",
      question: "You time a PyTorch GPU operation with Python's time.time() and get an absurdly small number. What is the most likely cause?",
      options: [
        "The GPU clock runs at a different frequency than the CPU clock, skewing the result",
        "Kernel launches are asynchronous, so the timer stops before the GPU finishes",
        "PyTorch caches the result of the first call and skips the computation entirely",
        "time.time() only has millisecond resolution, far too coarse for GPU kernels"
      ],
      correct: [1],
      explanation: "When Python calls a CUDA operation, the kernel is queued on the GPU and control returns to the CPU immediately. Without calling torch.cuda.synchronize() before reading the clock, you measure only the launch cost, not the actual execution. The lecture's benchmark() helper synchronizes before starting and before stopping the timer for exactly this reason."
    },
    {
      type: "single",
      question: "Why does the lecture's benchmark() function run warmup iterations before starting the timer?",
      options: [
        "To let the Python interpreter JIT-compile the benchmark loop before timing begins",
        "To fill the operating system's disk cache with the dataset before measurement",
        "The first calls pay one-time compilation and caching costs; steady state is what matters",
        "To raise the GPU clock to its boost frequency, which then persists for the run"
      ],
      correct: [2],
      explanation: "The first invocations of an operation can trigger CUDA context setup, kernel compilation (especially with torch.compile or Triton), and allocator growth. Since kernels are run many times in practice, the steady-state timing is what matters, so warmup iterations amortize those one-time costs away before measurement begins."
    },
    {
      type: "multi",
      question: "Which practices belong in a trustworthy wall-clock benchmark of a GPU operation? (Select all that apply)",
      options: [
        "Call torch.cuda.synchronize() before stopping the timer",
        "Run untimed warmup iterations first",
        "Average over multiple timed trials rather than trusting one",
        "Time only the very first call, since it is the most representative",
        "Disable the GPU and time the CPU fallback instead"
      ],
      correct: [0, 1, 2],
      explanation: "Synchronization ensures the queued kernels have actually finished when the timer stops; warmup removes one-time compilation and caching costs; and averaging several trials captures variance — the lecture's benchmark() does all three. The first call is the least representative, and timing a CPU fallback measures a different device entirely."
    },
    // ---------- Profiling ----------
    {
      type: "single",
      question: "Wall-clock timing says an operation is slow. What extra information does a profiler add that timing alone cannot?",
      options: [
        "A breakdown of which kernels ran, for how long, and time in memory operations",
        "The theoretical peak FLOP/s and memory bandwidth of the installed GPU model",
        "A guarantee that the computation produced numerically correct results",
        "An estimate of the power drawn and electricity consumed during the run"
      ],
      correct: [0],
      explanation: "A wall-clock number is a single opaque total. The PyTorch profiler decomposes it: which CUDA kernels were actually launched, their individual durations, and how time splits between compute and memory operations. The lecture stresses the deeper benefit — profiling reveals what is being called under the hood, such as cutlass matmul kernels whose names even encode the tile size (e.g., 256x128)."
    },
    {
      type: "multi",
      question: "Which observations in a profile suggest a kernel is memory-bound rather than compute-bound? (Select all that apply)",
      options: [
        "Achieved FLOP/s is far below the hardware's peak",
        "Runtime scales with bytes moved, not with arithmetic performed",
        "The kernel does few arithmetic operations per byte loaded from DRAM",
        "The kernel keeps the tensor cores busy near 100% utilization"
      ],
      correct: [0, 1, 2],
      explanation: "Memory-bound kernels are limited by memory bandwidth: arithmetic units sit idle waiting for data, so achieved FLOP/s is low, runtime tracks data volume, and arithmetic intensity (FLOPs per byte) is small — the lecture's definition of a low-intensity, memory-bound operation. Near-peak tensor-core utilization is the signature of the opposite, compute-bound, regime."
    },
    // ---------- GPU execution model ----------
    {
      type: "single",
      question: "Why does the lecture recommend launching at least $4\\times$ as many thread blocks as there are SMs?",
      options: [
        "It guarantees each thread block gets its own dedicated SM",
        "Shared memory is only enabled when blocks outnumber SMs",
        "CUDA refuses to launch grids smaller than the SM count",
        "Otherwise a partially-filled final wave leaves many SMs idle"
      ],
      correct: [3],
      explanation: "Thread blocks are scheduled onto SMs in waves, and the last wave usually has fewer blocks than SMs, leaving some idle (low occupancy). With at least $4\\times$ as many blocks as SMs, that ragged final wave is a small fraction of total runtime. The lecture also mentions wave quantization — sizing the grid so the block count divides evenly across SMs."
    },
    {
      type: "single",
      question: "Arithmetic intensity is defined as FLOPs per byte transferred. What general rule does the lecture give?",
      options: [
        "All GPU operations become compute-bound once the batch size is large enough",
        "Elementwise operations have the highest intensity since they parallelize well",
        "Matrix multiplication is compute-bound; nearly everything else is memory-bound",
        "Being memory-bound is desirable because memory bandwidth exceeds compute rates"
      ],
      correct: [2],
      explanation: "High arithmetic intensity means the compute units stay busy per byte loaded (good); low intensity means the operation is bottlenecked by memory (bad). Matmul reuses each loaded element many times, so it is compute-bound, while elementwise ops and reductions touch each byte only a few times and are limited by memory bandwidth."
    },
    // ---------- Kernel fusion ----------
    {
      type: "single",
      question: "Why is the lecture's hand-written GELU (a chain of separate multiplies, adds, and a tanh) so much slower than PyTorch's fused version?",
      options: [
        "Elementwise operations cannot be parallelized across more than one thread block",
        "Every operation reads its input from DRAM and writes its result back to DRAM",
        "PyTorch serializes elementwise operations onto a single streaming multiprocessor",
        "Each separate operation allocates and tears down a fresh CUDA context"
      ],
      correct: [1],
      explanation: "Profiling manual_gelu shows many separate kernels — one per multiply, add, and tanh — while pytorch_gelu launches just one. Each unfused kernel loads the tensor from global memory, does trivial arithmetic, and stores it back, so the data makes a full round trip per operation. The lecture's warehouse/factory analogy: shipping goods back to the warehouse between every assembly step."
    },
    {
      type: "single",
      question: "How does kernel fusion speed up a sequence of elementwise operations?",
      options: [
        "One kernel loads each element once, applies every step, and writes once",
        "It compresses the tensor in DRAM so far fewer bytes need to be moved",
        "It moves the elementwise chain onto a faster specialized CPU code path",
        "It skips any operation whose output is provably never used afterwards"
      ],
      correct: [0],
      explanation: "Fusion combines the chain into a single kernel: each element is read from DRAM once, transformed through every step while it stays on-chip, and written back once. Memory traffic drops from one round trip per operation to one total — organizing computation to minimize reads and writes is the lecture's key principle."
    },
    {
      type: "multi",
      question: "What does torch.compile do that can recover fused-kernel performance without hand-writing CUDA? (Select all that apply)",
      options: [
        "Captures the Python-level operation graph instead of executing eagerly op by op",
        "Automatically fuses chains of pointwise operations into single generated kernels",
        "Retrains the model with fewer parameters so each forward pass does less work",
        "Generates Triton kernels as its GPU code-generation backend"
      ],
      correct: [0, 1, 3],
      explanation: "The lecture's fifth way to write GELU is 'write it in Python and compile it into Triton': torch.compile traces the program into a graph and emits fused Triton kernels for chains of pointwise ops, eliminating the intermediate DRAM round trips of eager mode. It is a compiler, not a training procedure — the model and its parameters are unchanged."
    },
    // ---------- Triton programming model ----------
    {
      type: "single",
      question: "What is the unit of work you write when programming in Triton, in contrast to raw CUDA?",
      options: [
        "A program for one warp of 32 threads with explicit lane indexing",
        "A program written per individual scalar thread, exactly as in raw CUDA",
        "A whole-GPU program that manually schedules every streaming multiprocessor",
        "A program over a whole block (tile) of elements with vectorized operations"
      ],
      correct: [3],
      explanation: "In CUDA you write code from the perspective of a single thread and index with threadIdx. A Triton program instead operates on a block of values at once — you load a tile with tl.arange-style offsets and compute on it as a vector. Because the compiler controls the thread mapping, it can apply optimizations like thread coarsening and sometimes outperform PyTorch."
    },
    {
      type: "single",
      question: "In a Triton kernel, what is the purpose of the mask argument passed to tl.load and tl.store?",
      options: [
        "To encrypt sensitive tensor values while they reside in the GPU's shared memory",
        "To guard out-of-bounds accesses when the size is not a multiple of the block size",
        "To select which GPU device in a multi-GPU system the memory access is routed to",
        "To mark which tensor elements require gradients during the backward pass"
      ],
      correct: [1],
      explanation: "Each program instance computes pointer offsets for a full block, but the last block usually overhangs the end of the tensor. A mask like offsets < num_elements — exactly what the lecture's GELU kernel uses — disables the invalid lanes, so out-of-range loads return a default value and out-of-range stores are suppressed. Without it the kernel would read or write memory it does not own."
    },
    {
      type: "single",
      question: "What does tl.program_id(axis=0) return inside a Triton kernel?",
      options: [
        "The instance's index in the launch grid, used to locate its block of data",
        "The CUDA compute capability version of the currently active device",
        "A unique per-launch random seed for the kernel's number generator",
        "The number of elements each individual thread has been assigned to process"
      ],
      correct: [0],
      explanation: "The kernel is launched over a grid of program instances, one per block of data. program_id tells the running instance which block it is, and multiplying it by BLOCK_SIZE gives the starting offset of the elements it owns — the Triton analogue of blockIdx in CUDA, but at tile granularity."
    },
    {
      type: "multi",
      question: "Which low-level concerns does the Triton compiler handle for you, rather than the programmer? (Select all that apply)",
      options: [
        "Managing shared memory usage within a block",
        "Coalescing memory transfers from DRAM",
        "Scheduling work within a streaming multiprocessor",
        "Scheduling work across streaming multiprocessors"
      ],
      correct: [0, 1, 2],
      explanation: "The lecture's comparison table: memory coalescing, shared memory management, and scheduling within SMs are all manual in CUDA but automatic in Triton. Scheduling across SMs — the grid and how the problem is tiled into blocks — stays manual in both. Because the compiler does more work, Triton can actually outperform PyTorch implementations."
    },
    {
      type: "single",
      question: "What did the lecture observe when inspecting the PTX code generated for the Triton GELU kernel?",
      options: [
        "The compiler removed the tanh computation entirely",
        "Every element required its own separate global memory transaction",
        "The kernel spilled all registers to local memory",
        "One thread processes 8 elements at a time (thread coarsening)"
      ],
      correct: [3],
      explanation: "PTX is the assembly-like language for NVIDIA GPUs, and Triton lets you dump it for a compiled kernel. The lecture points out ld.global/st.global instructions for global memory, %ctaid.x/%tid.x for block and thread indices, and that one thread handles 8 elements simultaneously — thread coarsening, an optimization the Triton compiler applied automatically."
    },
    {
      type: "multi",
      question: "The lecture benchmarks several GELU implementations against each other. Which results does it report? (Select all that apply)",
      options: [
        "The unfused manual Python version was by far the slowest",
        "The naive hand-written CUDA kernel beat the Triton kernel",
        "PyTorch's built-in fused GELU was the fastest",
        "The Triton kernel was several times faster than PyTorch's"
      ],
      correct: [0, 1, 2],
      explanation: "The reported ordering: PyTorch's fused implementation is fastest, the naive CUDA kernel is next, the Triton kernel is almost as good as PyTorch but slower than CUDA, and everything is far faster than the unfused manual version. Triton's block-level model lets its compiler apply optimizations like thread coarsening, but it did not beat PyTorch here."
    },
    // ---------- Softmax in Triton ----------
    {
      type: "single",
      question: "Why does subtracting the maximum before exponentiating leave the softmax output unchanged?",
      options: [
        "The common factor $e^{-m}$ appears in numerator and denominator and cancels",
        "Because $e^{x-m}$ equals $e^x - e^m$, so the shift distributes over the terms",
        "Floating-point subtraction of the max is exact, so no information is lost",
        "Softmax depends only on the sign of each input logit, not on its magnitude"
      ],
      correct: [0],
      explanation: "Softmax is invariant to adding a constant to every logit: $\\frac{e^{x_i - m}}{\\sum_j e^{x_j - m}} = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$ because the common factor $e^{-m}$ cancels. Choosing $m$ as the row maximum makes every exponent nonpositive so the exponential never overflows — which is why both the lecture's manual and Triton softmax subtract the max first."
    },
    {
      type: "single",
      question: "The lecture counts the naive softmax's memory traffic on an $M{\\times}N$ matrix at $5MN + M$ reads and $3MN + 2M$ writes. What could an ideal fused kernel achieve?",
      options: [
        "Zero DRAM reads, since the data stays in registers between kernel launches",
        "About $MN$ reads and $MN$ writes — roughly a $4\\times$ reduction in traffic",
        "$M$ reads and $M$ writes, since only one value per row must be materialized",
        "Still $5MN$ reads, but only $M$ writes once the normalization is deferred"
      ],
      correct: [1],
      explanation: "Each step of manual_softmax — max, subtract, exponentiate, sum, normalize — makes its own pass over the matrix, totaling $5MN + M$ reads and $3MN + 2M$ writes. A fused kernel that keeps each row on-chip needs to read and write every element only once, $MN$ in and $MN$ out, so roughly a $4\\times$ speedup is available from memory traffic alone."
    },
    {
      type: "single",
      question: "How does the lecture's Triton softmax kernel divide the work across program instances?",
      options: [
        "One program per column, aggregating partial max and sum values across rows",
        "A single program instance that processes the whole matrix row by row",
        "One program per row, with a block spanning all columns padded to a power of 2",
        "One program per square tile, combining partial results through atomic adds"
      ],
      correct: [2],
      explanation: "triton_softmax launches a grid of $M$ program instances, one per row, with BLOCK_SIZE = triton.next_power_of_2(N) so a single block covers the entire row. The load is masked with other=-inf, so padding lanes contribute nothing to the row max or the exponential sum, and the whole max-exp-sum-normalize pipeline happens in one kernel."
    },
    // ---------- Tiled matrix multiplication ----------
    {
      type: "multi",
      question: "Which ideas does the lecture present for making matrix multiplication fast? (Select all that apply)",
      options: [
        "Load tiles of A and B into shared memory and reuse them for many outputs",
        "Process blocks in a grouped order that improves L2 cache reuse",
        "Skip multiplications wherever one of the matrix entries is close to zero",
        "Fuse a following operation like GELU into the matmul kernel"
      ],
      correct: [0, 1, 3],
      explanation: "Tiling loads each block of A and B into fast shared memory once and reuses it for a whole tile of outputs, cutting DRAM reads from the naive $MKN$ toward $MK + KN$. Grouped block ordering exploits the L2 cache — the lecture's example loads 54 blocks instead of 90. And the main reason to hand-write a matmul kernel at all is fusion, e.g., computing gelu(A @ B) in one pass."
    }
  ]
};
