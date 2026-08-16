// CS336 Lecture 14 — Data II
// 20 questions covering: filtering algorithms (KenLM n-gram models, CCNet,
// fastText classifiers, DSIR importance resampling), filtering applications
// (language identification, quality filtering, toxicity filtering), and
// deduplication (exact dedup, hash functions, Bloom filters, Jaccard,
// MinHash, LSH banding).
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture14"] = {
  title: "Lecture 14 — Data II",
  questions: [
    // ---------- Filtering algorithms ----------
    {
      type: "single",
      question: "Why are Kneser-Ney n-gram models (via KenLM) commonly used for data filtering?",
      options: [
        "They capture long-range dependencies better than transformer scorers",
        "They require no training data beyond the raw corpus being filtered",
        "They are extremely fast — essentially just counting and normalizing",
        "They are the only models that output well-calibrated probabilities"
      ],
      correct: [2],
      explanation: "Filtering has to run over the entire raw corpus, which is huge, so the scorer must be extremely fast — and n-gram estimation is just count-and-normalize. Kneser-Ney smoothing handles the sparse-count problem by backing off to lower-order n-grams for unseen sequences."
    },
    {
      type: "single",
      question: "How does CCNet use a KenLM language model to filter CommonCrawl?",
      options: [
        "Sort paragraphs by perplexity and keep the lowest-perplexity third",
        "Rewrite high-perplexity paragraphs using the language model",
        "Drop any page containing a sentence with perplexity over 15000",
        "Resample paragraphs with probability proportional to perplexity"
      ],
      correct: [0],
      explanation: "CCNet scores paragraphs with a KenLM model, sorts them by increasing perplexity, and keeps the top 1/3; this pipeline was used to build LLaMA's training data. Perplexity $\\exp(-\\frac{\\log p(x)}{N})$ normalizes log-probability by the token count $N$ so short documents are not unfairly favored."
    },
    {
      type: "single",
      question: "How does the fastText classifier keep its parameter count small versus a naive bag-of-words model?",
      options: [
        "It prunes the vocabulary to only the few thousand most frequent words",
        "It quantizes every embedding weight down to a single bit per entry",
        "It averages low-dimensional word embeddings before a small linear head",
        "It shares a single weight vector across all of the output classes"
      ],
      correct: [2],
      explanation: "A naive bag-of-words classifier needs $V \\times K$ parameters, which explodes with vocabulary and class count; fastText embeds words into $H$ dimensions, averages them, and applies an $H \\times K$ head, needing only $H(V+K)$. Bag-of-bigram features stay bounded via the hashing trick, and for quality filtering $K = 2$, so it reduces to a linear classifier."
    },
    {
      type: "multi",
      question: "Which statements about DSIR (Data Selection via Importance Resampling) are true? (Select all that apply)",
      options: [
        "It resamples raw examples with weights proportional to $p_T(x) / p_R(x)$",
        "It fits full neural language models to the target and raw data",
        "It uses hashed n-gram models because the target dataset is too small",
        "It performed slightly better than fastText-style filtering on GLUE",
        "It costs vastly more computation than fastText filtering"
      ],
      correct: [0, 2, 3],
      explanation: "DSIR fits bag-of-hashed-n-gram distributions to the target and raw data, then does importance resampling with weights proportional to $p_T(x) / p_R(x)$. Because the target set is too small to fit a rich model, hashed n-grams keep estimation tractable; the result was slightly better than heuristic classification on GLUE at similar computational cost."
    },
    {
      type: "multi",
      question: "Which correctly pair a filtering method with its scoring function? (Select all that apply)",
      options: [
        "KenLM (generative model of target): $\\text{score}(x) = p_T(x)$",
        "fastText (discriminative classifier): $\\text{score}(x) = p(T \\mid x)$",
        "DSIR: $\\text{score}(x) = p_T(x) \\cdot p_R(x)$",
        "DSIR: $\\text{score}(x) = p_T(x) / p_R(x)$, used to resample",
        "KenLM: $\\text{score}(x) =$ edit distance to the target corpus"
      ],
      correct: [0, 1, 3],
      explanation: "All three methods instantiate one framework — score examples in the raw data $R$ by similarity to a target $T$, then keep or resample by score. KenLM scores with the target's generative probability $p_T(x)$, fastText with the classifier probability $p(T \\mid x)$, and DSIR with the importance ratio $p_T(x)/p_R(x)$ (not a product)."
    },

    // ---------- Filtering applications ----------
    {
      type: "single",
      question: "How does Dolma decide whether a web page is English?",
      options: [
        "Keep pages the fastText language identifier scores $p(\\text{English}) \\ge 0.5$",
        "Keep pages where at least half of the characters are ASCII English text",
        "Keep pages a Wikipedia-trained KenLM scores below a perplexity cutoff",
        "Keep pages from web domains that human annotators labeled as English"
      ],
      correct: [0],
      explanation: "Dolma runs the off-the-shelf fastText language identification model, which supports 176 languages and was trained on multilingual sites like Wikipedia, Tatoeba, and SETimes. Pages with $p(\\text{English}) \\ge 0.5$ are kept."
    },
    {
      type: "multi",
      question: "Which are real difficulties for fastText-style language identification? (Select all that apply)",
      options: [
        "Short sequences are difficult to classify reliably",
        "It supports only about a dozen high-resource languages",
        "Similar languages like Malay and Indonesian are hard to distinguish",
        "It cannot run at web-corpus scale without GPU acceleration",
        "Dialects of English can be accidentally filtered out"
      ],
      correct: [0, 2, 4],
      explanation: "Language ID struggles on short inputs, low-resource languages, and closely related language pairs, and aggressive English filtering risks discarding English dialects. Code-switched text (e.g., Spanish + English lyrics) is ill-defined for a single-label classifier. The model itself is fast, CPU-friendly, and covers 176 languages."
    },
    {
      type: "single",
      question: "How did GPT-3's quality classifier define its training data?",
      options: [
        "Positives from pages referenced by Wikipedia; negatives from Reddit",
        "Positives from The Stack; negatives from Books1 and Books2",
        "Positives from Wikipedia, WebText2, and Books; negatives from CommonCrawl",
        "Positives labeled by GPT-4 prompts; negatives sampled at random"
      ],
      correct: [2],
      explanation: "GPT-3 trained a linear classifier on word features with samples from Wikipedia, WebText2, Books1, and Books2 as positives and CommonCrawl as negatives, then kept documents stochastically using a Pareto-distributed threshold on the score. LLaMA instead used pages referenced by Wikipedia as positives; phi-1 used GPT-4 prompt labels."
    },
    {
      type: "single",
      question: "What was the philosophy behind phi-1's training data?",
      options: [
        "Scrape maximal raw web data and rely purely on the scale of tokens",
        "Feed a small model very high-quality 'textbook' data, synthetic plus filtered",
        "Train exclusively on human-written textbooks, with no synthetic data at all",
        "Replace data curation entirely with reinforcement learning from feedback"
      ],
      correct: [1],
      explanation: "phi-1 combined synthetic data from GPT-3.5 with code filtered by a classifier trained on GPT-4 educational-value labels over the Python subset of The Stack. A 1.3B model on the filtered subset hit 17.68% on HumanEval after 36K steps, beating 12.19% after 96K steps on the unfiltered subset."
    },
    {
      type: "single",
      question: "How does Dolma perform toxicity filtering?",
      options: [
        "A GPT-4 prompt scores every page for toxicity and hate speech",
        "Regular expressions match a fixed blocklist of offensive words and slurs",
        "A KenLM model flags any text whose perplexity is unusually high",
        "Two fastText classifiers (hate, NSFW) trained on Jigsaw Toxic Comments"
      ],
      correct: [3],
      explanation: "Dolma trained two fastText classifiers — one for hate and one for NSFW content — on the 2018 Jigsaw Toxic Comments dataset, which consists of Wikipedia talk-page comments annotated with labels like toxic, obscene, threat, and insult. This reuses the same fast classification machinery as quality and language filtering."
    },

    // ---------- Deduplication: motivation and exact dedup ----------
    {
      type: "multi",
      question: "Why deduplicate pretraining data? (Select all that apply)",
      options: [
        "Repeated data spends training compute on redundant tokens",
        "Deduplication guarantees no copyrighted text remains in the corpus",
        "Duplicated passages amplify verbatim memorization and regurgitation",
        "Removing duplicates has been shown to improve model quality",
        "Deduplication shrinks the tokenizer vocabulary"
      ],
      correct: [0, 2, 3],
      explanation: "Duplicates waste compute, amplify memorization (raising copyright and privacy concerns), and 'Deduplicating Training Data Makes Language Models Better' shows quality gains from removing them — one product description appeared 61,036 times in C4. Dedup does not remove unique copyrighted text, and it has nothing to do with vocabulary size."
    },
    {
      type: "single",
      question: "What is the basic recipe for exact deduplication of a corpus?",
      options: [
        "Compute MinHash signatures and compare all pairs of documents",
        "Hash each item and keep one representative per hash value",
        "Cluster documents by embedding similarity and keep cluster centroids",
        "Sort documents alphabetically and drop adjacent near-matches"
      ],
      correct: [1],
      explanation: "Exact dedup hashes each item (document, paragraph, or line) and keeps a single representative per hash — a MapReduce-style computation that parallelizes and scales easily. Its limitation is that it only catches byte-identical copies, so near-duplicates differing by a few tokens slip through."
    },
    {
      type: "single",
      question: "C4 deduplicated by exact-matching 3-sentence spans, removing all but one copy. What risk does this create?",
      options: [
        "Removing a span from mid-document can leave the remaining text incoherent",
        "The exact-match computation cannot be parallelized across machines",
        "Hash collisions silently delete large numbers of unique sentences",
        "It requires quadratic pairwise comparisons between all of the spans"
      ],
      correct: [0],
      explanation: "When the item is a 3-sentence span rather than a whole document, deleting a duplicated span from the middle of a page can break the flow of the surviving document. This illustrates the dedup design space: choosing the item, the matching rule, and the action all have consequences."
    },
    {
      type: "single",
      question: "Why do deduplication pipelines use hash functions like MurmurHash rather than SHA-256?",
      options: [
        "MurmurHash is cryptographically collision-resistant while SHA-256 is not",
        "Speed matters at corpus scale; collision resistance is unnecessary",
        "SHA-256 outputs are too small to index a hash table of documents",
        "MurmurHash produces identical hashes for near-duplicate documents"
      ],
      correct: [1],
      explanation: "There is a tradeoff between efficiency and collision resistance: cryptographic hashes like SHA-256 are collision-resistant but slow (suited to Bitcoin), while DJB2, MurmurHash, and CityHash are fast but not collision-resistant (suited to hash tables). Dedup over web-scale data needs the fast kind."
    },

    // ---------- Bloom filters ----------
    {
      type: "single",
      question: "When you query a Bloom filter for an item, what can it correctly report?",
      options: [
        "The exact count of how many times the item was inserted",
        "The item's position in the underlying storage array",
        "A guaranteed-correct yes or no answer for any item",
        "Either 'possibly in the set' or 'definitely not in the set'"
      ],
      correct: [3],
      explanation: "A Bloom filter sets $k$ hash-indexed bits per inserted item; a query returns 'possibly present' if all its bits are set and 'definitely absent' otherwise. It stores no items or counts — just the bit array — which is what makes it so memory-efficient."
    },
    {
      type: "multi",
      question: "Which statements about Bloom filters are true? (Select all that apply)",
      options: [
        "They can return false positives on membership queries",
        "They can return false negatives on membership queries",
        "The false-positive rate is tunable via the bin count and number of hash functions",
        "They store the inserted items so they can be retrieved later",
        "They use far less memory than storing the full set of items"
      ],
      correct: [0, 2, 4],
      explanation: "Bloom filters trade exactness for memory: unrelated items can collide on the same bits (false positives), but an inserted item always finds its bits set, so false negatives are impossible. Using more hash functions drives the false-positive rate down exponentially — with the optimal $k = \\ln 2 \\cdot m/n$ — and Dolma targets a rate of $10^{-15}$ over paragraphs."
    },

    // ---------- Near-duplicate detection ----------
    {
      type: "single",
      question: "How is the Jaccard similarity of two sets A and B defined?",
      options: [
        "The size of the larger set divided by the smaller set",
        "The number of items unique to A plus those unique to B",
        "The size of the intersection divided by the size of the union",
        "The cosine of the angle between the sets' count vectors"
      ],
      correct: [2],
      explanation: "$\\text{Jaccard}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$, ranging from 0 (disjoint) to 1 (identical sets). Two documents are declared near duplicates when their Jaccard similarity exceeds a threshold, and MinHash lets you estimate it without materializing the intersection."
    },
    {
      type: "single",
      question: "What is the key property of MinHash that makes it useful for estimating similarity?",
      options: [
        "It maps every document to a globally unique fixed-length fingerprint",
        "$P[h(A) = h(B)] = \\text{Jaccard}(A, B)$ for a random hash function $h$",
        "It gives a provable lower bound on the edit distance between documents",
        "It guarantees that dissimilar sets never produce equal hash values"
      ],
      correct: [1],
      explanation: "A random hash function induces a permutation over items, and each item of $A \\cup B$ is equally likely to be the minimum, so $P[\\text{minhash}(A) = \\text{minhash}(B)] = \\text{Jaccard}(A, B)$. Unlike ordinary hashing, you want collision probability to depend on similarity; averaging over many hash functions estimates Jaccard."
    },
    {
      type: "multi",
      question: "In LSH with banding, n MinHash functions are split into b bands of r each. Which statements are true? (Select all that apply)",
      options: [
        "Two documents become a candidate pair if they match on every hash of at least one band",
        "The probability a pair with Jaccard similarity $s$ collides is $1 - (1 - s^r)^b$",
        "Banding requires explicitly comparing every pair of documents in the corpus",
        "Increasing $r$ sharpens the threshold and moves it right (harder to match)"
      ],
      correct: [0, 1, 3],
      explanation: "A pair collides if some band has all $r$ hash values agree, giving collision probability $1 - (1 - s^r)^b$; this and-or structure sharpens the threshold. Only bucket collisions are checked — no all-pairs comparison — and increasing $r$ demands closer agreement while increasing $b$ moves the curve left (easier to match)."
    },
    {
      type: "single",
      question: "What is the 'S-curve' trade-off when choosing the banding parameters (b, r)?",
      options: [
        "More bands always reduce false positives and false negatives together",
        "The S-curve fixes the threshold at exactly $0.5$ regardless of $b$ and $r$",
        "They trade signature memory against exact recovery of Jaccard values",
        "They set where collision probability jumps from near 0 to near 1"
      ],
      correct: [3],
      explanation: "The candidate probability $1 - (1 - s^r)^b$ is an S-shaped function of similarity $s$, with the phase transition at roughly $(1/b)^{1/r}$. More bands (or fewer rows) slide the jump left, catching more true near-duplicates but admitting more false positives; the reverse slides it right — e.g., one paper used $n = 9000$ with $b = 20$, $r = 450$."
    }
  ]
};
