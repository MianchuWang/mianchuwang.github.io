// CS336 Lecture 7 — Parallelism I: basics
// 20 questions covering: collective communication primitives, the intra/inter-node
// communication hierarchy, data parallelism and its memory problem, ZeRO stages 1-3
// (FSDP), pipeline parallelism, tensor parallelism, sequence parallelism, batch-size
// limits, and 3D-parallelism rules of thumb.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture7"] = {
  title: "Lecture 7 — Parallelism I",
  questions: [
    // ---------- Collectives ----------
    {
      type: "single",
      question: "After an all-reduce (sum) over 4 GPUs that each hold a gradient tensor, what does each GPU end up holding?",
      options: [
        "Its own local gradient tensor, entirely unchanged",
        "The elementwise sum of all four gradient tensors",
        "One quarter (its own shard) of the summed gradient tensor",
        "The gradient tensor copied over from rank 0"
      ],
      correct: [1],
      explanation: "All-reduce combines (here, sums) the tensors from every rank and delivers the identical full result to every rank. This 'everyone contributes, everyone receives' semantics is exactly what data-parallel gradient averaging needs, which is why all-reduce is the workhorse collective of DDP."
    },
    {
      type: "single",
      question: "What does a reduce-scatter over N ranks do?",
      options: [
        "Every rank ends up holding the full reduced tensor",
        "Rank 0 alone ends up holding the full reduced tensor",
        "Each rank ends with a distinct $1/N$ shard of the result",
        "Each rank sends its shard to all others without any reduction"
      ],
      correct: [2],
      explanation: "Reduce-scatter performs the reduction (e.g., sum) across ranks but scatters the result: rank $i$ keeps only the $i$-th shard of the reduced tensor. It is the 'reduce half' of all-reduce, and it is exactly the operation ZeRO uses so each rank owns just its own slice of the summed gradients."
    },
    {
      type: "multi",
      question: "Which collectives leave EVERY rank holding the same complete result? (Select all that apply)",
      options: [
        "Broadcast",
        "All-reduce",
        "Reduce",
        "All-gather",
        "Scatter"
      ],
      correct: [0, 1, 3],
      explanation: "Broadcast copies one rank's data to all; all-reduce gives everyone the combined tensor; all-gather gives everyone the concatenation of all shards. By contrast, reduce delivers the combined result to a single root rank, and scatter gives each rank a different chunk — the 'all-' prefix is the tell that every rank ends with the full result."
    },
    {
      type: "single",
      question: "All-reduce can be implemented as which sequence of two collectives?",
      options: [
        "Broadcast first, then reduce",
        "Gather first, then scatter",
        "All-gather, then reduce-scatter",
        "Reduce-scatter, then all-gather"
      ],
      correct: [3],
      explanation: "First reduce-scatter sums the tensor and leaves each rank with its own reduced shard; then all-gather circulates those shards so every rank holds the full reduced tensor. The lecture stresses that in the bandwidth-limited regime this two-step implementation is the best you can do — and ZeRO gets its memory savings by keeping the two halves apart."
    },
    // ---------- Communication hierarchy ----------
    {
      type: "single",
      question: "How does GPU communication differ within a node versus across nodes?",
      options: [
        "GPUs within a node enjoy far higher bandwidth than GPUs on different nodes",
        "Inter-node links are faster because each server bundles many network cards",
        "Bandwidth is identical inside and across nodes; only latency differs",
        "The network between nodes cannot transfer GPU tensors directly"
      ],
      correct: [0],
      explanation: "Parallelism spans two tiers: high-speed interconnects (e.g., NVLink) among the roughly 8 GPUs inside one server, and a much slower network between servers. This hierarchy drives strategy placement — bandwidth-hungry tensor parallelism stays within a node, while pipeline and data parallelism run over the slower inter-node links."
    },
    // ---------- Data parallelism ----------
    {
      type: "multi",
      question: "Which statements correctly describe naive (distributed) data parallelism? (Select all that apply)",
      options: [
        "Every GPU holds a full replica of the model parameters",
        "The global batch is split so each GPU computes on its own shard of examples",
        "Gradients are synchronized across replicas with an all-reduce every step",
        "Each GPU is responsible for training a different layer of the model",
        "Parameters are sharded so that no single GPU stores the whole model"
      ],
      correct: [0, 1, 2],
      explanation: "Data parallelism replicates the model everywhere, splits the $B$-sized batch across the $M$ machines, and synchronizes with an all-reduce over gradients (about $2\\times\\#\\text{params}$ of communication per batch) so every replica takes the identical step. Assigning different layers to different GPUs is pipeline parallelism, and sharding parameters is what ZeRO stage 3 / tensor parallelism do — naive DP gives no memory scaling at all."
    },
    {
      type: "single",
      question: "With Adam and mixed precision, why does naive data parallelism fail to solve the memory problem even though it scales compute?",
      options: [
        "Activations always dominate memory and cannot be reduced by any parallelism",
        "Gradients must be kept in full fp64 precision on every replica",
        "The per-GPU batch shards remain too large to fit in a single GPU's memory",
        "Optimizer state is memory-dominant yet fully replicated on every GPU"
      ],
      correct: [3],
      explanation: "Mixed-precision Adam needs about 16 bytes per parameter — effectively 5 copies of the weights: 2 bytes for bf16 parameters, 2 for bf16 gradients, 4 for fp32 master weights, and 4 + 4 for Adam's first and second moment estimates. Naive DP replicates all of it on every GPU, so adding GPUs adds compute but zero per-GPU memory headroom — the observation that motivates ZeRO."
    },
    // ---------- ZeRO ----------
    {
      type: "multi",
      question: "Which statements are true of ZeRO stage 1? (Select all that apply)",
      options: [
        "Optimizer state (first and second moments) is sharded across ranks",
        "Parameters are sharded during the forward and backward passes",
        "Each rank applies the optimizer update only to its own parameter slice",
        "Gradients and parameters remain fully replicated on every rank"
      ],
      correct: [0, 2, 3],
      explanation: "Stage 1 partitions only the optimizer state — the biggest memory consumer — so each rank stores $1/N$ of it and updates just its slice of the parameters. The step is: compute a full gradient, reduce-scatter the gradients, update your slice, then all-gather the updated parameters; forward and backward still run on fully replicated parameters and gradients."
    },
    {
      type: "single",
      question: "What does ZeRO stage 2 shard in addition to what stage 1 already shards?",
      options: [
        "Gradients",
        "Parameters",
        "Activations",
        "The input batch"
      ],
      correct: [0],
      explanation: "The progression is cumulative: stage 1 shards optimizer state, stage 2 additionally keeps gradients sharded, and stage 3 additionally shards the parameters themselves. The stage-2 subtlety is that each worker must still compute a full gradient (it is data parallel), so gradients are reduced to the responsible worker layer by layer during backward and freed immediately — a full gradient vector is never materialized."
    },
    {
      type: "single",
      question: "In ZeRO stage 3 (FSDP), parameters are sharded. How does a rank run a layer's forward pass?",
      options: [
        "It uses only its own parameter shard, producing partial outputs",
        "It requests the missing parameters from a central parameter server",
        "It all-gathers that layer's parameters on demand, then frees them",
        "It keeps a cached full copy of every layer's parameters at all times"
      ],
      correct: [2],
      explanation: "Stage 3 materializes full parameters only transiently: parameters are requested and sent on demand while stepping through the compute graph, used, and immediately freed. FSDP also overlaps these all-gathers with ongoing computation, masking much of the communication cost while peak memory scales as $1/N$."
    },
    {
      type: "multi",
      question: "Which statements about ZeRO communication costs are correct? (Select all that apply)",
      options: [
        "Stages 1 and 2 match naive DP's $2\\times\\#\\text{params}$ per-step communication cost",
        "Stage 3 costs $3\\times\\#\\text{params}$ — about 1.5x naive DP — due to parameter gathering",
        "Stage 1 doubles the per-step communication volume relative to naive data parallelism",
        "Stage 1 replaces the gradient all-reduce with a reduce-scatter plus an all-gather"
      ],
      correct: [0, 1, 3],
      explanation: "Stage 1 swaps DDP's gradient all-reduce for a reduce-scatter (gradients) plus an all-gather (updated parameters) — the same $2\\times\\#\\text{params}$, since all-reduce equals that very composition, so it is essentially free and 'you might as well always do it'. Stage 2 is also (almost) free ignoring overhead, while stage 3's on-demand parameter gathering brings the total to $3\\times\\#\\text{params}$ — a 1.5x premium."
    },
    {
      type: "multi",
      question: "Which collectives occur during one ZeRO stage 3 (FSDP) training step? (Select all that apply)",
      options: [
        "All-gather of parameter shards before each layer's forward computation",
        "All-gather of parameter shards again during the backward pass",
        "Reduce-scatter of gradients so each rank keeps only its own shard",
        "Broadcast of the updated optimizer states from rank 0 after each step"
      ],
      correct: [0, 1, 2],
      explanation: "Per the lecture, one stage-3 step costs 2 all-gathers plus 1 reduce-scatter, each of size $\\#\\text{params}$: parameters are gathered for forward, gathered again for backward (they were freed in between to save memory), and gradients are reduce-scattered so each rank keeps just its shard. Optimizer states are never communicated — each rank permanently owns and locally updates its slice."
    },
    // ---------- Model parallelism ----------
    {
      type: "single",
      question: "Model parallelism shards parameters across GPUs, just like ZeRO stage 3. What is the key difference in what gets communicated?",
      options: [
        "Model parallelism communicates activations, whereas ZeRO-3 sends the parameters themselves",
        "Model parallelism communicates optimizer state, whereas ZeRO-3 sends gradients",
        "ZeRO-3 communicates activations, whereas model parallelism sends full parameters",
        "Both communicate exactly the same tensors; only the scheduling differs"
      ],
      correct: [0],
      explanation: "The lecture defines model parallelism as splitting parameters across GPUs like ZeRO-3 does, but communicating activations between GPUs instead of shipping parameter shards on demand. This flips the communication profile — pipeline parallelism sends small point-to-point activation tensors and tensor parallelism all-reduces activations per layer — and it scales memory without needing a bigger batch."
    },
    // ---------- Pipeline parallelism ----------
    {
      type: "single",
      question: "In pipeline parallelism, why is the batch split into micro-batches?",
      options: [
        "So each GPU needs to hold fewer parameters in memory",
        "To keep multiple stages busy at once, shrinking the idle bubble",
        "To reduce the size of the gradients being communicated",
        "To avoid storing any activations between pipeline stages"
      ],
      correct: [1],
      explanation: "Naive layer-wise parallelism has terrible utilization: with $n$ GPUs, each is active only $1/n$ of the time, idling while it waits for other stages. Streaming micro-batches lets stages work concurrently, and the bubble-to-compute ratio becomes $\\frac{n_{\\text{stages}}-1}{n_{\\text{micro}}}$ — so a large batch (many micro-batches) is essential to hide the bubble."
    },
    {
      type: "single",
      question: "Why is pipeline parallelism typically placed on slower inter-node links rather than inside a node?",
      options: [
        "It requires each node to contain exactly one pipeline stage",
        "Inter-node links are faster than intra-node interconnects",
        "It communicates full parameter tensors for every micro-batch",
        "Its point-to-point activation transfers tolerate lower bandwidth"
      ],
      correct: [3],
      explanation: "A pipeline stage sends only its output activations ($b \\times s \\times h$) point-to-point to the next stage — far less traffic than tensor parallelism's roughly $8bsh$ per-layer all-reduces or FSDP's parameter gathering. The lecture's guidance: use pipelines on slower network links (inter-node) as a way to get linear memory scaling, and keep bandwidth-hungry tensor parallelism on fast intra-node interconnects."
    },
    // ---------- Tensor parallelism ----------
    {
      type: "single",
      question: "In tensor parallelism, why is the MLP's first matrix split by columns and the second by rows?",
      options: [
        "It halves the number of parameters each GPU must store for the MLP",
        "Column splits execute faster than row splits on GPU tensor cores",
        "The intermediate stays sharded, so one all-reduce at the output suffices",
        "It lets each GPU skip applying the nonlinearity to its own slice"
      ],
      correct: [2],
      explanation: "Splitting $A$ by columns gives each rank a slice of the intermediate activations; splitting $B$ by rows consumes exactly that slice and produces partial sums that one all-reduce combines. In the lecture's notation, $f$ is the identity and $g$ is an all-reduce in the forward pass, with the roles swapped in backward — so no communication is needed between the two matrix multiplies."
    },
    {
      type: "single",
      question: "Why is tensor parallelism usually confined to GPUs within a single node?",
      options: [
        "InfiniBand is physically unable to transmit activation tensors",
        "Tensor parallelism only supports splits of up to 8 ways by design",
        "Tensor parallelism requires identical GPUs, which only one node guarantees",
        "Its per-layer activation all-reduces need high-bandwidth interconnects"
      ],
      correct: [3],
      explanation: "Tensor parallelism communicates roughly $8bsh \\cdot \\frac{n-1}{n}$ per layer as all-reduces — much more than pipeline parallelism's $bsh$ point-to-point transfers — so it demands low-latency, high-bandwidth links. That is why it is used within a node (up to 8 GPUs) on the fast interconnect; in exchange it has no pipeline bubble, is simple to wrap around a model, and does not need large batch sizes."
    },
    // ---------- Sequence parallelism ----------
    {
      type: "single",
      question: "Which activations does sequence parallelism shard that tensor parallelism leaves replicated?",
      options: [
        "The quadratic attention score matrices in each head",
        "The learnable gain and bias parameters of each LayerNorm",
        "LayerNorm, dropout, and block-input activations",
        "The optimizer's first- and second-moment buffers"
      ],
      correct: [2],
      explanation: "Tensor parallelism shards the matrix multiplies in attention and the MLP, but about $10sbh$ of activations per layer — LayerNorm ($4sbh$), dropout ($2sbh$), and the inputs to attention and the MLP ($4sbh$) — stay replicated and keep growing with model size. Since these are pointwise operations over the sequence, sequence parallelism splits them along the sequence axis: in forward, $g$ becomes an all-gather and $\\bar{g}$ a reduce-scatter (reversed in backward), completing fully linear activation-memory scaling."
    },
    // ---------- Batch-size interplay ----------
    {
      type: "multi",
      question: "Global batch size = per-device batch x data-parallel degree. Which consequences follow? (Select all that apply)",
      options: [
        "Scaling out data parallelism grows the global batch unless the per-device batch shrinks",
        "The per-device batch size has no effect on each GPU's activation memory",
        "The number of data-parallel machines cannot usefully exceed the batch size",
        "Ever-larger batches hit diminishing returns, capping useful data-parallel scale"
      ],
      correct: [0, 2, 3],
      explanation: "Because data parallelism splits the batch across machines, #machines must stay below the batch size — and communication overhead is high even near that limit — while diminishing returns to batch size cap how far scaling out helps optimization. Per-device batch directly scales activation memory, so the second statement is false; when the batch is small, gradient accumulation can trade a higher effective batch for better communication efficiency."
    },
    // ---------- 3D parallelism ----------
    {
      type: "single",
      question: "According to the lecture's rules of thumb, how are the parallelism strategies combined for a large training run?",
      options: [
        "Data parallel first, then tensor parallel across nodes if the model still does not fit",
        "Tensor parallel within a node, pipeline (or ZeRO-3) across nodes until it fits, then data parallel",
        "Pipeline parallel within a node and tensor parallel across nodes, then data parallel everywhere",
        "One form of parallelism at a time, adding the next only after the previous is exhausted"
      ],
      correct: [1],
      explanation: "Until the model fits in memory: tensor parallel up to the GPUs per machine (about 8, where it caps out and an 8 x 8 configuration is often optimal), plus pipeline parallel — or ZeRO-3, depending on bandwidth — across machines. Then scale the rest of the way with data parallel; real runs (e.g., Narayanan 2021) follow exactly this pattern, with DP degree shrinking as models grow."
    }
  ]
};
