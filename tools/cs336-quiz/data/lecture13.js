// CS336 Lecture 13 — Data I
// 20 questions covering: data as the biggest (and least published) lever of model
// quality, Common Crawl and WARC vs WET, pipeline stages (language ID, quality
// filtering, deduplication), rule-based heuristics (C4, MassiveText/Gopher),
// model-based quality filters (CCNet, GPT-3, DCLM, Nemotron-CC), the corpus
// lineage from WebText to FineWeb/DCLM, copyright and fair use, and
// mid-training / post-training data (Alpaca, Vicuna, Llama 2 chat).
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture13"] = {
  title: "Lecture 13 — Data I",
  questions: [
    // ---------- Data as the biggest lever ----------
    {
      type: "single",
      question: "The lecture argues that among architecture, systems, and data, one ingredient is simultaneously the biggest lever on model quality and the least published. Which?",
      options: [
        "Architecture — attention variants are the main differentiator between frontier labs",
        "Systems — parallelism strategies are guarded as trade secrets",
        "Data — curation drives quality, yet frontier papers say almost nothing about it",
        "Optimizers — learning-rate schedules are the key undisclosed ingredient"
      ],
      correct: [2],
      explanation: "Open-weight models like Llama 3 are fully transparent about architecture and even training procedure, but disclose basically nothing about data. The lecture's hot take is that data is the most important thing to get right — and unlike architectures and systems, it is a long-tail problem that scales with human effort."
    },
    {
      type: "single",
      question: "Why do frontier labs reveal so little about their pretraining data, according to the lecture?",
      options: [
        "The pipelines change too quickly for documentation to stay accurate",
        "Competitive advantage, plus disclosure invites copyright liability",
        "Regulators require the recipes for training data to stay confidential",
        "The datasets are too large for any published details to be meaningful"
      ],
      correct: [1],
      explanation: "The lecture names exactly two forces behind the secrecy: competitive dynamics and copyright liability. Llama 3, for instance, documents its architecture and training in detail while reducing its data description to almost nothing."
    },

    // ---------- Common Crawl ----------
    {
      type: "single",
      question: "What is Common Crawl, the starting point for most open pretraining corpora?",
      options: [
        "A nonprofit crawling the web since 2007, releasing free periodic snapshots",
        "Google's internal search index, licensed to researchers on request",
        "A curated collection of manually approved high-quality websites",
        "A distributed protocol where volunteers donate their browsing histories"
      ],
      correct: [0],
      explanation: "Common Crawl is a nonprofit founded in 2007 that runs a web crawl roughly every month (about 100 crawls from 2008-2025) using Apache Nutch, starting from hundreds of millions of seed URLs. Nearly every open web corpus — C4, RefinedWeb, Dolma, FineWeb, DCLM — begins from these crawls."
    },
    {
      type: "multi",
      question: "Which statements about Common Crawl's WARC and WET formats are true? (Select all that apply)",
      options: [
        "WARC files store raw HTTP responses, including the full HTML of each page",
        "WET files preserve the original HTML markup of every crawled page",
        "WET files contain plain text already extracted by Common Crawl's own pipeline",
        "WARC files contain only URLs and fetch metadata, with no page content"
      ],
      correct: [0, 2],
      explanation: "WARC is the raw crawl: full HTTP responses with the original HTML. WET is a convenience derivative where Common Crawl has already stripped the HTML down to plain text via a lossy process. The markup lives only in WARC — which is why pipelines that care about extraction quality go back to it."
    },
    {
      type: "single",
      question: "RefinedWeb re-extracts text from WARC using trafilatura, and Pile-CC used jusText on WARC, instead of using WET. Why?",
      options: [
        "WET files are compressed in a format that is expensive to decode at scale",
        "WARC files are smaller than WET, so re-extraction cuts storage costs",
        "WET files omit all non-English pages, which these pipelines still need",
        "WET extraction is lossy; better HTML-to-text tools improve downstream accuracy"
      ],
      correct: [3],
      explanation: "Common Crawl's WET conversion is a lossy process. The DCLM paper shows that the choice of HTML-to-text conversion matters for downstream task accuracy — which is why RefinedWeb runs trafilatura on WARC and The Pile found jusText on WARC better than WET."
    },

    // ---------- The processing pipeline ----------
    {
      type: "single",
      question: "How did CCNet select high-quality documents from Common Crawl?",
      options: [
        "Kept pages linked from Reddit posts with at least 3 karma",
        "Trained a fastText quality classifier on instruction-style positive examples",
        "Kept documents that look like Wikipedia under a KenLM 5-gram model",
        "Dropped lines without terminal punctuation, following C4's manual rules"
      ],
      correct: [2],
      explanation: "CCNet's pipeline is deduplication of paragraphs, fastText language identification, then quality filtering that keeps documents resembling Wikipedia under a KenLM 5-gram model. It aimed to build large pretraining sets automatically — including for low-resource languages — and BERT trained on CCNet output outperformed BERT trained on Wikipedia."
    },
    {
      type: "multi",
      question: "Which of the following are standard stages in a pretraining data pipeline? (Select all that apply)",
      options: [
        "Gradient checkpointing to reduce activation memory",
        "Deduplication, both exact and fuzzy (e.g., MinHash)",
        "PII removal and toxicity filtering",
        "Reinforcement learning from human feedback",
        "Language identification with a fastText classifier"
      ],
      correct: [1, 2, 4],
      explanation: "The pipelines in the lecture combine language ID (fastText in CCNet, Dolma, FineWeb), quality filtering, deduplication (MinHash in RefinedWeb and The Stack, Bloom filters in Dolma), plus PII anonymization (FineWeb) and toxicity filtering (Dolma's Jigsaw classifier). Gradient checkpointing is a training-systems technique, and RLHF belongs to post-training."
    },
    {
      type: "single",
      question: "How is language identification typically performed at web scale?",
      options: [
        "A cheap fastText classifier scores each document against a confidence threshold",
        "A large language model reads each page and reports back its language",
        "The HTML lang attribute is trusted directly, since browsers require it",
        "Documents are routed by their top-level domain, such as .fr, .de, or .jp"
      ],
      correct: [0],
      explanation: "Pipelines run a fast, cheap classifier — canonically fastText's language ID model, as in CCNet, Dolma, and OpenWebText — over every document and keep those above a confidence threshold (FineWeb keeps English if p(en) > 0.65). HTML metadata and domains are unreliable, and LLM-based scoring is far too expensive at billions of pages."
    },

    // ---------- Rule-based quality heuristics ----------
    {
      type: "single",
      question: "Which set of rules is characteristic of C4's rule-based cleaning?",
      options: [
        "Keep pages that a neural classifier judges most similar to Wikipedia articles",
        "Keep only pages shared in Reddit posts that earned at least 3 karma",
        "Drop unpunctuated lines, short pages, and docs containing 'lorem ipsum' or '{'",
        "Keep only pages scoring low perplexity under a KenLM 5-gram language model"
      ],
      correct: [2],
      explanation: "C4 (the T5 corpus) is purely heuristic: keep lines ending in punctuation with at least 5 words, drop pages with fewer than three sentences, and remove documents containing bad words, 'lorem ipsum', or '{' (a proxy for code). One April 2019 snapshot of 1.4T tokens became 806 GB (156B tokens). Reddit karma is WebText's proxy, and 5-gram perplexity is CCNet's."
    },
    {
      type: "multi",
      question: "Which statements about MassiveText, Gopher's training dataset, are true? (Select all that apply)",
      options: [
        "Quality filtering used a fastText classifier trained on Wikipedia positives",
        "Quality filtering used manual rules like 80% of words containing a letter",
        "Toxicity filtering used Google SafeSearch rather than word lists",
        "Gopher trained on all 10.5 TB of text that the pipeline produced"
      ],
      correct: [1, 2],
      explanation: "MassiveText deliberately used manual rules rather than a classifier for quality — for example requiring 80% of words to contain at least one alphabetic character — and used Google SafeSearch instead of word lists for toxicity. The pipeline produced 10.5 TB, but Gopher trained on only 300B tokens, about 12% of it."
    },

    // ---------- Model-based quality filtering ----------
    {
      type: "single",
      question: "What is the core recipe for model-based quality filtering?",
      options: [
        "Ask a frontier LLM to rewrite every crawled page into cleaner prose",
        "Train a classifier on trusted seed text vs. web text, keep the top-scored pages",
        "Cluster pages by embedding similarity and keep only the largest clusters",
        "Filter with hand-written regular expressions tuned by expert annotators"
      ],
      correct: [1],
      explanation: "The pattern: pick a trusted 'positive' corpus, take generic web text as negatives, train a cheap classifier to distinguish them, and keep the highest-scoring documents. GPT-3 did this with {WebText, Wikipedia, Books} as positives; DCLM did it with a fastText classifier over instruction-style positives and RefinedWeb negatives."
    },
    {
      type: "single",
      question: "DCLM's winning quality filter is a fastText classifier. What did it use as positive examples?",
      options: [
        "Wikipedia articles together with out-of-copyright Gutenberg books",
        "Pages hand-labeled as high quality by paid expert annotators",
        "Common Crawl pages with the lowest KenLM perplexity scores",
        "OpenHermes-2.5 instruction data plus ELI5 subreddit answers"
      ],
      correct: [3],
      explanation: "DCLM trained fastText on 200K positives from OpenHermes-2.5 (mostly GPT-4-generated instruction data) and the ELI5 subreddit, against 200K negatives from RefinedWeb, then filtered the 240T-token DCLM-pool down to a 3.8T-token baseline. Strikingly, 'quality' here means resembling instruction data, not encyclopedias — and this classifier outperformed other filtering methods."
    },
    {
      type: "single",
      question: "Nemotron-CC took a different stance from FineWebEdu and DCLM. What motivated its design?",
      options: [
        "Shadow-library books offered far more training tokens than the open web",
        "Synthetic rephrased data had to be removed entirely from pretraining",
        "A smaller, purely rule-filtered corpus in the style of C4 was the goal",
        "Those filters remove ~90% of data; it targets more tokens without losing quality"
      ],
      correct: [3],
      explanation: "Nemotron-CC argues FineWebEdu and DCLM filter too aggressively when models like Llama 3 (15T tokens) and Qwen3 (36T) need volume. It chose jusText over trafilatura because it returns more tokens, ensembled quality classifiers, and used an LM to rephrase low-quality data and generate tasks from high-quality data, yielding 6.3T tokens (1.1T high-quality)."
    },

    // ---------- Corpus lineage ----------
    {
      type: "multi",
      question: "Which corpus-to-contribution pairings are correct? (Select all that apply)",
      options: [
        "The Pile assembled 22 diverse curated sources, including code and academic text",
        "C4 pioneered model-based quality classifiers for filtering web text at scale",
        "RefinedWeb showed filtered web data alone can rival curated mixtures",
        "Dolma aggregated many sources, including Pushshift Reddit and PeS2o papers"
      ],
      correct: [0, 2, 3],
      explanation: "The Pile (a grassroots open-source effort in reaction to GPT-3) curated 22 high-quality domains; RefinedWeb's point was 'web data is all you need'; Dolma (3T tokens) combined Reddit via Pushshift, PeS2o academic papers, C4, Gutenberg, and Wikipedia. C4 is the odd one out — its cleaning is entirely rule-based, not model-based."
    },
    {
      type: "single",
      question: "What quality proxy did WebText (GPT-2's corpus) use, later replicated openly as OpenWebText?",
      options: [
        "Pages ranked highly by Google search",
        "Pages whose authors were verified journalists",
        "Outbound Reddit links that received at least 3 karma",
        "Pages that passed an early BERT-based quality classifier"
      ],
      correct: [2],
      explanation: "WebText scraped the pages linked from Reddit posts with at least 3 karma — upvotes as a free surrogate for quality — yielding 8 million pages and 40 GB of text. OpenWebText replicated the recipe from the Reddit submissions dataset, filtering language with fastText and removing near-duplicates."
    },
    {
      type: "single",
      question: "According to the lecture, which statement about copyright protection is accurate?",
      options: [
        "Only works registered with the copyright office receive legal protection",
        "Registration is not needed for protection, only for suing over infringement",
        "Copyright protects ideas, such as algorithms, not just their expression",
        "Most content posted on the Internet falls into the public domain by default"
      ],
      correct: [1],
      explanation: "The threshold for copyright is extremely low and registration is not required (unlike patents) — so most things on the Internet are actually copyrighted. Registration ($65) is only needed before a creator can sue for infringement, and copyright covers expression, not ideas like quicksort."
    },

    // ---------- Domain value and fair use ----------
    {
      type: "single",
      question: "Why do pretraining mixtures include source code even for models not aimed at coding?",
      options: [
        "Code is believed to help reasoning, not just programming tasks",
        "Code is far cheaper to license than natural language text",
        "Code tokenizes more efficiently, reducing overall training cost",
        "Code prevents the model from memorizing natural-language text"
      ],
      correct: [0],
      explanation: "The lecture notes that code is helpful for programming tasks but also for reasoning — a belief it labels 'folklore', yet one that motivates including GitHub-derived data like The Stack (3.1 TB of permissively licensed code, deduplicated with MinHash) in general-purpose mixtures."
    },
    {
      type: "multi",
      question: "Which of these are among the four fair-use factors? (Select all that apply)",
      options: [
        "Whether the copyright holder registered the work before it was used",
        "The purpose and character of the use (transformative favored over reproductive)",
        "The nature of the copyrighted work (factual favored over creative)",
        "The effect of the use on the market for the original work"
      ],
      correct: [1, 2, 3],
      explanation: "The four factors are purpose and character of the use, nature of the copyrighted work, amount and substantiality of the portion used, and effect on the market. Training a model is arguably transformative, but the lecture notes language models can affect the market for writers and artists regardless of copyright."
    },

    // ---------- Mid-training and post-training data ----------
    {
      type: "single",
      question: "In the lecture's three stages of training, what characterizes mid-training?",
      options: [
        "Fine-tuning on instruction-following data to produce a chat model",
        "Further training on small amounts of high-quality data to boost capabilities",
        "Pausing training to re-tokenize the corpus with a larger vocabulary",
        "Reinforcement learning to align the model with human preferences"
      ],
      correct: [1],
      explanation: "The progression runs from large amounts of lower-quality data (pre-training) to small amounts of high-quality data (mid-training), then post-training on instruction data or RL. A 'base model' is the result of pre-training plus mid-training; an 'instruct/chat model' comes after post-training — though in practice the lines are blurry."
    },
    {
      type: "multi",
      question: "Which pairings of instruction/chat dataset and data source are correct? (Select all that apply)",
      options: [
        "Llama 2 chat: millions of examples scraped from open instruction datasets",
        "Alpaca: 52K examples generated from text-davinci-003 via self-instruct",
        "Baize: conversations hand-written by paid expert annotators",
        "Vicuna: 70K ChatGPT conversations users shared on ShareGPT"
      ],
      correct: [1, 3],
      explanation: "Alpaca fine-tuned LLaMA 7B on 52K self-instruct examples from text-davinci-003, and Vicuna fine-tuned LLaMA on 70K ShareGPT conversations. Baize generated its 111.5K examples via GPT-3.5 self-chat seeded with Quora and StackOverflow questions, while Llama 2 chat used just 27,540 high-quality vendor annotations — reportedly better than millions of open examples."
    }
  ]
};
