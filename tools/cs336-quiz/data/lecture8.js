// CS336 Lecture 8 — Parallelism (Percy): distributed basics in code.
// 20 questions covering: collective operations and terminology, hardware
// interconnects (NVLink/NVSwitch/PCIe/HBM), NCCL and torch.distributed,
// bandwidth benchmarking, and bare-bones data/tensor/pipeline parallelism
// implementations on deep MLPs.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture8"] = {
  title: "Lecture 8 — Parallelism II",
  questions: [
    // ---------- Big picture ----------
    {
      type: "single",
      question: "In pipeline parallelism, how is the model partitioned across devices?",
      options: [
        "Each weight matrix is sliced column-wise across all devices",
        "Consecutive layers are grouped into stages, one stage per device",
        "Every device keeps a full replica of the entire model's layers",
        "Attention heads go to one device while MLPs go to another"
      ],
      correct: [1],
      explanation: "Pipeline parallelism cuts the model along its depth: each rank gets a contiguous subset of layers and transfers all data/activations to the next rank. In the lecture's implementation each of the 2 ranks computes $\\text{num\\_layers} / \\text{world\\_size} = 2$ layers. This contrasts with tensor parallelism, which slices within each layer along the width."
    },
    {
      type: "single",
      question: "In the lecture's pipeline implementation, why is the batch split into micro-batches?",
      options: [
        "To keep gradients fresh and improve convergence",
        "To fit the optimizer states of each stage in memory",
        "To minimize the bubble of idle time on waiting stages",
        "To avoid re-sending activations for every layer"
      ],
      correct: [2],
      explanation: "If the whole batch moved through the stages at once, each stage would sit idle while the others compute — the pipeline bubble. Streaming smaller micro-batches lets stages work concurrently on different chunks. The lecture notes it does NOT handle the further step of overlapping communication with computation to eliminate bubbles."
    },
    {
      type: "multi",
      question: "Which statements about collective-operation terminology are correct? (Select all that apply)",
      options: [
        "Reduce performs an associative/commutative operation such as sum, min, or max",
        "Rank refers to the total number of devices participating",
        "The 'all' prefix means the result is delivered to all devices",
        "Broadcast and scatter are inverses of gather"
      ],
      correct: [0, 2, 3],
      explanation: "The lecture's mnemonic: reduce applies an associative/commutative op, broadcast/scatter invert gather, and 'all' means every device is a destination. World size is the number of devices (e.g., 4); a rank is one device (0, 1, 2, 3), not the count."
    },
    {
      type: "single",
      question: "According to the lecture, all-reduce is equivalent to which composition of collectives?",
      options: [
        "A broadcast followed by a local gather",
        "A scatter followed by a reduce on each rank",
        "A reduce-scatter followed by an all-gather",
        "An all-gather followed by a broadcast"
      ],
      correct: [2],
      explanation: "The lecture demonstrates this in code: it feeds each rank's reduce-scatter output into an all-gather and recovers exactly the all-reduce result — 'Indeed, all-reduce = reduce-scatter + all-gather!'. This decomposition also explains all-reduce's communication cost."
    },
    {
      type: "single",
      question: "Which communication primitive moves activations between stages in the lecture's pipeline-parallel code?",
      options: [
        "An all-gather of activations after every local layer",
        "Point-to-point dist.send / dist.recv between adjacent ranks",
        "An all-reduce of gradients at the end of every micro-batch",
        "A broadcast of each micro-batch from rank 0 to all stages"
      ],
      correct: [1],
      explanation: "Each stage calls dist.recv to get a micro-batch's activations from rank − 1, computes its local layers, then dist.send to rank + 1. Pipeline parallelism is the one strategy in the lecture built on point-to-point transfers rather than collectives."
    },
    {
      type: "multi",
      question: "Which statements correctly describe communication in the lecture's three implementations? (Select all that apply)",
      options: [
        "Tensor parallelism all-gathers activations after every layer",
        "Pipeline parallelism sends activations point-to-point to the next stage",
        "Data parallelism requires no communication between workers",
        "Data parallelism all-reduces gradients once per step after backward"
      ],
      correct: [0, 1, 3],
      explanation: "TP communicates inside every layer: local matmul + GeLU, then all-gather and concatenate. PP uses send/recv at stage boundaries. DP is not communication-free — it all-reduces (averages) every parameter's gradient after backward, which is its only departure from standard training."
    },
    // ---------- Summary trade-offs ----------
    {
      type: "single",
      question: "Which is NOT one of the lecture summary's three options for handling a value needed later?",
      options: [
        "Re-compute the value when it is needed",
        "Store it in the GPU's own memory",
        "Store it in another GPU's memory and communicate",
        "Keep a compressed copy in lower precision"
      ],
      correct: [3],
      explanation: "The summary lists exactly three choices: re-compute, store in memory, or store in another GPU's memory and communicate. Quantization is a real technique but is not part of this lecture's trade-off framing, which is about where computation sits relative to data."
    },
    {
      type: "single",
      question: "In the lecture's all-reduce benchmark, sent bytes are $\\text{size} \\times 2 \\times (\\text{world\\_size} - 1)$. Why the factor of 2?",
      options: [
        "Each rank sends its input and receives the reduced output",
        "The warmup pass gets counted together with the timed pass",
        "NCCL retransmits every packet to guarantee reliable delivery",
        "Both the gradients and the parameters must be transferred"
      ],
      correct: [0],
      explanation: "All-reduce moves data in and out of every rank: inputs must be sent for reduction and reduced outputs received back — mirroring its reduce-scatter + all-gather structure. Reduce-scatter alone drops the $2\\times$, since each rank only contributes input data."
    },
    {
      type: "multi",
      question: "Which statements about the lecture's bandwidth benchmarks are correct? (Select all that apply)",
      options: [
        "The measured bandwidth is the same thing as HBM memory bandwidth",
        "A warmup call runs before the timed collective",
        "torch.cuda.synchronize ensures CUDA kernels finish before timing stops",
        "Reduce-scatter's byte count omits the $2\\times$ factor used for all-reduce"
      ],
      correct: [1, 2, 3],
      explanation: "The benchmarks warm up first, then time a single collective, using torch.cuda.synchronize plus dist.barrier so all kernels and processes finish before the clock stops. Reduce-scatter counts $\\text{data\\_bytes} \\times (\\text{world\\_size} - 1)$ with no $2\\times$. What is measured is interconnect bandwidth between GPUs, not HBM bandwidth within one."
    },
    // ---------- Data parallelism ----------
    {
      type: "single",
      question: "In the lecture's data-parallel implementation, what is the only difference from standard single-process training?",
      options: [
        "Parameters are broadcast from rank 0 before every step",
        "The loss is all-reduced across ranks before backward",
        "Optimizer states are sharded evenly across the ranks",
        "Gradients are averaged across ranks after backward"
      ],
      correct: [3],
      explanation: "The code comment says it directly: syncing gradients with dist.all_reduce(op=AVG) is the only difference between standard training and DDP. Each rank holds all parameters and its own optimizer state, and works on its own slice of the batch."
    },
    {
      type: "single",
      question: "Why do parameters remain identical across ranks in data parallelism, even though losses differ?",
      options: [
        "All ranks start identical and apply the same averaged gradients",
        "Rank 0 re-broadcasts the parameters after each optimizer step",
        "Every rank actually trains on the same local slice of the data",
        "The optimizer state lives in memory shared across the ranks"
      ],
      correct: [0],
      explanation: "Losses differ because each rank computes on its local data slice, but the all-reduce makes every rank's gradients identical. Identical initial parameters plus identical gradient updates keep the replicas in sync without any explicit re-broadcasting — exactly what the lecture's printed output confirms."
    },
    // ---------- Tensor parallelism ----------
    {
      type: "single",
      question: "In the lecture's tensor-parallel MLP with world size 4, what parameters does each rank hold?",
      options: [
        "A full replica of all four of the layers' weight matrices",
        "One complete layer of the four, with its full weight matrix",
        "A quarter of the data batch plus a full copy of all parameters",
        "A $\\text{num\\_dim} \\times (\\text{num\\_dim} / 4)$ slice of every layer"
      ],
      correct: [3],
      explanation: "Tensor parallelism shards the width dimension: each rank owns a $\\text{num\\_dim} \\times \\text{local\\_num\\_dim}$ slice of every layer's weight matrix, where $\\text{local\\_num\\_dim} = \\text{num\\_dim} / \\text{world\\_size}$. All ranks process the full batch, unlike data parallelism, which shards the batch instead."
    },
    {
      type: "multi",
      question: "Which statements describe the lecture's tensor-parallel forward pass? (Select all that apply)",
      options: [
        "Each rank computes a local matmul and GeLU on its parameter slice",
        "Activations are all-gathered from all ranks after every layer",
        "Gathered chunks are concatenated to restore the full hidden width",
        "The backward pass is fully implemented in the lecture code"
      ],
      correct: [0, 1, 2],
      explanation: "Per layer, each rank produces a $\\text{batch\\_size} \\times \\text{local\\_num\\_dim}$ activation slice, all-gathers everyone's slices, and concatenates along the hidden dimension to rebuild the $\\text{batch\\_size} \\times \\text{num\\_dim}$ input for the next layer. The backward pass is left as a homework exercise (as is pipeline parallelism's)."
    },
    // ---------- Hardware ----------
    {
      type: "single",
      question: "Per the lecture's numbers, how does an H100's NVLink bandwidth compare with its HBM bandwidth?",
      options: [
        "They are roughly equal, at about 1 TB/s in each direction",
        "NVLink (~3.9 TB/s) is several times faster than HBM (~900 GB/s)",
        "HBM (~3.9 TB/s) is several times faster than NVLink (~900 GB/s)",
        "NVLink is faster because HBM must route through the PCIe bus"
      ],
      correct: [2],
      explanation: "An H100 has 18 NVLink 4.0 links totaling 900 GB/s, while its HBM delivers about 3.9 TB/s. Cross-GPU links are several times slower than local memory — one more level of the small/fast-to-big/slow hierarchy, and the reason to orchestrate computation to avoid data transfer."
    },
    {
      type: "single",
      question: "In the modern datacenter setup described in the lecture, what does NVSwitch provide?",
      options: [
        "A faster PCIe bus linking each GPU to the host CPU",
        "Direct GPU-to-GPU links across nodes, bypassing Ethernet",
        "Extra HBM capacity pooled among the GPUs within a node",
        "A dedicated low-latency link from GPUs to network storage"
      ],
      correct: [1],
      explanation: "The classic home setup routes intra-node traffic over PCIe and inter-node traffic over Ethernet (~200 MB/s). Modern datacenters replace both: NVLink connects GPUs directly within a node, bypassing the CPU, and NVSwitch connects GPUs directly across nodes, bypassing Ethernet."
    },
    // ---------- Software stack ----------
    {
      type: "single",
      question: "Which best describes NCCL's role in distributed training?",
      options: [
        "It compiles PyTorch graphs into fused, optimized GPU kernels",
        "It schedules Python worker processes across the cluster's nodes",
        "It provides the user-facing API such as all_gather_into_tensor",
        "It translates collectives into low-level packets sent between GPUs"
      ],
      correct: [3],
      explanation: "NCCL sits below the API: it detects the hardware topology (nodes, switches, NVLink/PCIe), optimizes the paths between GPUs, and launches CUDA kernels to send and receive data. The clean interface for collective operations comes from torch.distributed, which calls into NCCL."
    },
    {
      type: "single",
      question: "Which backend pairing does the lecture's setup() function use for torch.distributed?",
      options: [
        "gloo for GPU, nccl for CPU",
        "mpi for CPU, gloo for GPU",
        "gloo for CPU, nccl for GPU",
        "nccl for both CPU and GPU"
      ],
      correct: [2],
      explanation: "torch.distributed supports multiple backends for different hardware: the lecture initializes nccl when CUDA is available and falls back to gloo on CPU. Setup also sets MASTER_ADDR/MASTER_PORT so rank 0 can coordinate the processes, while the actual data flows through NCCL."
    },
    // ---------- Pipeline details ----------
    {
      type: "multi",
      question: "Which statements describe the lecture's pipeline-parallel implementation? (Select all that apply)",
      options: [
        "Rank 0 chunks the real data; later ranks allocate empty buffers for incoming activations",
        "It overlaps communication with computation so pipeline bubbles are fully hidden",
        "Each rank computes $\\text{num\\_layers} / \\text{world\\_size}$ consecutive layers",
        "Each stage receives from rank − 1 and sends to rank + 1 per micro-batch"
      ],
      correct: [0, 2, 3],
      explanation: "Only rank 0 holds the input data (chunked into micro-batches); downstream ranks pre-allocate empty tensors to receive activations. Each rank owns an equal consecutive slice of layers and relays micro-batches with recv/send. Overlapping communication and computation to eliminate bubbles is explicitly listed as NOT handled."
    },
    {
      type: "single",
      question: "Why does the lecture demonstrate all three parallelism strategies on deep MLPs?",
      options: [
        "Deep MLPs are the only architecture torch.distributed supports",
        "MLPs dominate Transformer compute, making them representative",
        "Attention layers cannot be split across multiple devices",
        "MLPs need no backward pass, which greatly simplifies the code"
      ],
      correct: [1],
      explanation: "The lecture uses bare-bones deep MLPs because MLPs dominate Transformer compute, making them a representative testbed. What's missing — attention, more communication/computation overlap — requires more bookkeeping, and frameworks like Jax can even derive the distributed execution from a declared sharding strategy."
    },
    {
      type: "multi",
      question: "Which parallelism axes does the lecture's summary pair with the correct dimension? (Select all that apply)",
      options: [
        "Sequence parallelism cuts along the batch dimension",
        "Data parallelism cuts along the batch dimension",
        "Tensor/expert parallelism cuts along the width dimension",
        "Pipeline parallelism cuts along the depth dimension"
      ],
      correct: [1, 2, 3],
      explanation: "The summary lists: data (batch), tensor/expert (width), pipeline (depth), and sequence (length). Sequence parallelism cuts along the sequence length, not the batch — it is a distinct axis from data parallelism even though both leave the model weights whole."
    }
  ]
};
