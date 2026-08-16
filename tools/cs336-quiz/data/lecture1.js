// CS336 Lecture 1 — Overview & Tokenization
// 20 questions covering: course philosophy and the efficiency framing,
// Unicode/byte-level and word-level tokenization problems, the BPE algorithm,
// tokenizer observations, the bitter lesson, scaling laws, and model openness.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture1"] = {
  title: "Lecture 1 — Overview & Tokenization",
  questions: [
    // ---------- Course philosophy & efficiency framing ----------
    {
      type: "single",
      question: "CS336 frames language modeling as fundamentally an 'efficiency' problem. Given fixed compute and data budgets, what is the goal?",
      options: [
        "Minimize wall-clock training time regardless of final loss",
        "Maximize the amount of raw data collected before training",
        "Train the best model achievable with the given resources",
        "Minimize the parameter count of the final model"
      ],
      correct: [2],
      explanation: "The course's framing: resources (compute, data) are the binding constraint, so every design decision — tokenizer, architecture, systems, data curation — is judged by how much model quality it buys per unit of resource. Minimizing wall-clock time or parameters is only instrumental; the objective is the best model the budget allows."
    },
    {
      type: "single",
      question: "Why does the course insist on building everything from scratch rather than relying on high-level libraries?",
      options: [
        "High-level libraries run too slowly for cutting-edge research use",
        "Abstractions are leaky, and fundamental research needs the full stack",
        "Industrial training codebases are closed-source and cannot be studied",
        "From-scratch implementations reliably outperform library versions"
      ],
      correct: [1],
      explanation: "The claim is not that libraries are bad, but that their abstractions leak: tokenization quirks, numerical precision, and memory behavior all surface in model quality, so you cannot reason about them without understanding what is underneath. Fast library code is fine to use once you understand what it hides."
    },
    {
      type: "multi",
      question: "Which claims match the course's stated philosophy? (Select all that apply)",
      options: [
        "Understanding comes from building the components, not just reading about them",
        "Frontier-scale training runs can be reproduced on a single academic GPU",
        "Mechanics and mindset transfer to frontier scale, though intuitions may not",
        "Tokenization details can be delegated to an off-the-shelf library"
      ],
      correct: [0, 2],
      explanation: "The course's motto is 'understanding via building', and it distinguishes three types of knowledge: mechanics (how things work) and mindset (squeezing the most out of hardware, taking scale seriously) transfer to frontier models, while intuitions about data and modeling decisions only partially transfer across scales. It never claims frontier runs are reproducible on academic hardware, and tokenization is treated as something to understand, not outsource."
    },

    // ---------- Unicode & byte-level tokenization ----------
    {
      type: "single",
      question: "What is the main problem with using raw Unicode codepoints directly as the token vocabulary?",
      options: [
        "Unicode cannot represent emoji or scripts beyond the Latin alphabet",
        "Codepoint sequences are far longer than the equivalent byte sequences",
        "Decoding codepoint sequences back into text is inherently lossy",
        "The vocabulary is huge (~150K) and many codepoints are extremely rare"
      ],
      correct: [3],
      explanation: "Unicode defines on the order of 150K codepoints, most of which appear rarely or never in a training corpus, so their embeddings would be badly undertrained. The length concern runs the other way: codepoint sequences are SHORTER than byte sequences, and decoding codepoints is lossless."
    },
    {
      type: "single",
      question: "Why is pure byte-level tokenization problematic for a vanilla transformer?",
      options: [
        "Sequences become very long, and attention cost is quadratic in length",
        "A 256-entry vocabulary is too small for the output softmax to train well",
        "Bytes cannot faithfully represent characters outside the ASCII range",
        "Byte sequences cannot be decoded unambiguously back into Unicode text"
      ],
      correct: [0],
      explanation: "UTF-8 bytes give a compression ratio of 1 — at least one token per character — so contexts blow up in length and quadratic ($O(n^2)$) attention makes training and inference expensive. The small vocabulary itself is not a training problem, and UTF-8 bytes represent all of Unicode losslessly — that is precisely their appeal."
    },
    {
      type: "multi",
      question: "Which are genuine advantages of a byte-level representation? (Select all that apply)",
      options: [
        "Sequences are shorter than with word-level tokenization",
        "Any text can be encoded — there are no out-of-vocabulary symbols",
        "Attention cost becomes independent of input length",
        "The vocabulary is tiny and fixed at 256 values"
      ],
      correct: [1, 3],
      explanation: "Bytes give complete coverage (every string is a byte sequence) with a fixed 256-entry vocabulary — no UNK, no rare-token problem. The price is the opposite of option one: byte sequences are much LONGER than word- or subword-level sequences, and attention cost still scales with length."
    },

    // ---------- Word-level tokenization ----------
    {
      type: "single",
      question: "A word-level tokenizer maps unseen words to a special UNK token. Why is this harmful for language modeling?",
      options: [
        "UNK makes sequences much longer than character-level tokenization would",
        "UNK tokens cannot be assigned a row in the embedding matrix",
        "Input content is erased, and perplexity computations become distorted",
        "UNK forces the vocabulary to keep growing throughout training"
      ],
      correct: [2],
      explanation: "Collapsing every novel word to UNK throws away the input's content, and probabilities assigned to UNK make loss and perplexity comparisons misleading. UNK is an ordinary vocabulary entry with an embedding — the problem is what it erases, not how it is stored."
    },
    {
      type: "multi",
      question: "Which problems afflict word-level tokenization? (Select all that apply)",
      options: [
        "The vocabulary becomes enormous on diverse corpora",
        "Rare words receive poorly trained embeddings",
        "Common words get split into many small pieces",
        "New or misspelled words fall outside the vocabulary"
      ],
      correct: [0, 1, 3],
      explanation: "Open-ended text has a heavy-tailed word distribution: the vocabulary explodes, tail words are seen too rarely to train well, and anything outside the list becomes OOV. Splitting common words into pieces is not a word-level failure — word-level tokenizers keep words whole; over-splitting is a subword/byte concern."
    },

    // ---------- The BPE algorithm ----------
    {
      type: "single",
      question: "During BPE training, what operation is repeated until the target vocabulary size is reached?",
      options: [
        "Split the least frequent token into two shorter tokens",
        "Merge the most frequent adjacent pair of tokens into a new token",
        "Delete the least frequent token from the vocabulary",
        "Merge the longest substring that appears anywhere in the corpus"
      ],
      correct: [1],
      explanation: "BPE greedily finds the adjacent token pair with the highest count in the corpus, creates a new token for it, and replaces all occurrences — repeating until the vocabulary budget is spent. It only ever merges bottom-up from the base units; it never splits or deletes, and frequency of the PAIR, not substring length, drives each step."
    },
    {
      type: "single",
      question: "How does a trained BPE tokenizer encode new text at inference time?",
      options: [
        "It looks up whole words, backing off to characters on a miss",
        "It searches for the segmentation that minimizes total token count",
        "It samples among valid segmentations to regularize the model",
        "It replays the recorded merges on the byte sequence in learned order"
      ],
      correct: [3],
      explanation: "Encoding replays training: start from base units (e.g., bytes) and apply each recorded merge rule in its learned order wherever it matches. This is deterministic and greedy — BPE does not search for a globally minimal segmentation, which is why its output can occasionally be a suboptimal split."
    },
    {
      type: "single",
      question: "In byte-level BPE, as used by GPT-2, what is the initial vocabulary before any merges are learned?",
      options: [
        "The 256 possible values of a single byte",
        "All words appearing at least twice in the corpus",
        "The full set of Unicode codepoints",
        "A hand-designed list of common subwords"
      ],
      correct: [0],
      explanation: "Byte-level BPE starts from the $2^8 = 256$ byte values, guaranteeing every string is representable from step zero, and merges build larger tokens on top. Starting from codepoints or word lists would reintroduce coverage problems (rare codepoints, OOV words) that the byte base exists to avoid."
    },
    {
      type: "multi",
      question: "Which statements about a trained BPE tokenizer are true? (Select all that apply)",
      options: [
        "Its merge rules are corpus-independent, fixed by the algorithm alone",
        "Frequent words typically end up represented as single tokens",
        "With a byte-level base, any input string can be tokenized without UNK",
        "Its tokens are guaranteed to align with linguistic morphemes"
      ],
      correct: [1, 2],
      explanation: "Merges are learned from pair frequencies, so common words quickly coalesce into single tokens, and the byte base guarantees full coverage. But the rules depend entirely on the training corpus — a code-heavy corpus yields different merges than English prose — and frequency-driven merges follow statistics, not morphology."
    },

    // ---------- Tokenizer properties in practice ----------
    {
      type: "single",
      question: "A tokenizer's compression ratio is often reported as bytes per token. What does a higher ratio indicate?",
      options: [
        "A smaller vocabulary was used to train the tokenizer",
        "The encoding discards more information per character",
        "Each token covers more text, so sequences come out shorter",
        "Encoding requires more compute for every input character"
      ],
      correct: [2],
      explanation: "The ratio $\\frac{\\text{num bytes}}{\\text{num tokens}}$ measures how much raw text each token absorbs — higher means better compression, so the same document costs fewer tokens and fits more content per context window. Higher ratios generally come from LARGER vocabularies, not smaller, and tokenization stays lossless either way."
    },
    {
      type: "single",
      question: "Where does the BPE algorithm originally come from?",
      options: [
        "It was designed by OpenAI specifically for GPT-2's tokenizer",
        "It originated in 1994 as a general data compression algorithm",
        "It emerged from classical research on linguistic morphology",
        "It was first proposed in the 2017 Transformer paper"
      ],
      correct: [1],
      explanation: "BPE was introduced by Philip Gage in 1994 for data compression, then adapted to NLP for neural machine translation (Sennrich et al., 2016) at a time when papers used word-based tokenization, and later used by GPT-2. It is a compression heuristic driven by corpus statistics, not a linguistic theory."
    },
    {
      type: "single",
      question: "What is the lecture's assessment of tokenizer-free approaches (e.g., ByT5, MegaByte, BLT) that use bytes directly?",
      options: [
        "They already power the majority of today's frontier models",
        "They fail outright because bytes cannot represent all of Unicode",
        "They were tried in GPT-2 and abandoned as too slow to train",
        "They are promising but not yet scaled up to frontier models"
      ],
      correct: [3],
      explanation: "Working with raw bytes is elegant and promising, but these approaches have not been scaled up to frontier models — byte sequences are compute-inefficient with today's architectures. Hence the lecture calls tokenization a 'necessary evil': maybe one day we'll just do it from bytes."
    },
    {
      type: "multi",
      question: "Which observations does the lecture make while exploring the GPT-2 tokenizer? (Select all that apply)",
      options: [
        "A word and its preceding space are part of the same token",
        "Whitespace runs are stripped out before tokens are assigned",
        "Numbers are tokenized into chunks of every few digits",
        "A word at the start versus the middle of text is tokenized differently",
        "Input text is lowercased before token ids are assigned"
      ],
      correct: [0, 2, 3],
      explanation: "Playing with the GPT-2 tokenizer shows that \" world\" folds the leading space into the token, that \"hello hello\" gives the same word different tokens by position, and that numbers split into every few digits. Whitespace and case are preserved — encode() and decode() must round-trip the exact string."
    },

    // ---------- Efficiency, scaling, and openness ----------
    {
      type: "multi",
      question: "Today's models are compute-constrained. Which design decisions does the lecture attribute to squeezing the most out of hardware? (Select all that apply)",
      options: [
        "Training for only a single epoch",
        "Tokenizing rather than modeling raw bytes, which is compute-inefficient",
        "Tuning hyperparameters on smaller models guided by scaling laws",
        "Repeating scarce data over many epochs to exploit it fully"
      ],
      correct: [0, 1, 2],
      explanation: "Efficiency drives design decisions in the compute-constrained regime: one epoch suffices, tokenization exists because raw bytes are elegant but compute-inefficient with today's architectures, and scaling laws let you tune hyperparameters cheaply at small scale. Multi-epoch training on scarce data belongs to the future: 'tomorrow, we will become data-constrained.'"
    },
    {
      type: "single",
      question: "According to the lecture, what is the right interpretation of the 'bitter lesson'?",
      options: [
        "Algorithms that scale are what matter, not scale by itself",
        "Scale is all that matters, and algorithmic ideas are irrelevant",
        "Advances in hardware alone explain the recent progress",
        "Results at small scale reliably predict large-scale behavior"
      ],
      correct: [0],
      explanation: "The lecture explicitly rejects 'scale is all that matters; algorithms don't matter' as the wrong reading — the right one is that algorithms that scale are what matter, summarized as $\\text{accuracy} = \\text{efficiency} \\times \\text{resources}$. Efficiency becomes MORE important at larger scale, where waste is unaffordable; one study found $44\\times$ algorithmic efficiency gains on ImageNet between 2012 and 2019."
    },
    {
      type: "single",
      question: "The compute-optimal scaling rule quoted in the lecture is $D^* = 20\\,N^*$. How many training tokens does it prescribe for a 1.4B-parameter model?",
      options: [
        "About 1.4B tokens",
        "About 7B tokens",
        "About 28B tokens",
        "About 280B tokens"
      ],
      correct: [2],
      explanation: "Chinchilla-style compute-optimal scaling says to train on about 20 tokens per parameter ($D^* = 20\\,N^*$), so a 1.4B model should see roughly $1.4\\text{B} \\times 20 = 28\\text{B}$ tokens. The lecture notes a caveat: this rule ignores inference costs, which push toward smaller models trained on more data."
    },
    {
      type: "single",
      question: "In the lecture's levels of openness, what characterizes open-weight models like DeepSeek?",
      options: [
        "API access is offered, but no weights are released",
        "Weights and architecture details, but no data details",
        "Weights, data, and most training details are all released",
        "The training data is released, but the weights are withheld"
      ],
      correct: [1],
      explanation: "Open-weight releases publish the weights and a paper with architecture and some training details, but no data details. Closed models (e.g., GPT-4o) offer API access only, while open-source models like OLMo release weights and data — though even they rarely share the rationale or failed experiments."
    }
  ]
};
