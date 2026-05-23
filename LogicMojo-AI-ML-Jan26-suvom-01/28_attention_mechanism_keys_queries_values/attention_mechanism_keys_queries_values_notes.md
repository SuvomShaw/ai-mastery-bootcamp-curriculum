# Attention Mechanism 

## Keys, Queries, and Values

Attention is one of the most important ideas behind modern NLP models and Transformers.

The core idea is simple:

```text
For this word, which other words are important?
```

Attention helps a model decide where to look.

It turns normal word embeddings into context-aware embeddings.

```text
Embeddings -> Attention -> Context-aware embeddings
```

---

## 1. Why Attention Comes After Embeddings

In NLP, raw text must first become numbers.

The usual pipeline is:

```text
raw text -> tokens -> token IDs -> embeddings -> model
```

Example:

```text
"I love NLP"
```

Tokens:

```text
["I", "love", "NLP"]
```

Token IDs:

```text
[2, 3, 4]
```

Embeddings:

```text
2 -> vector for "I"
3 -> vector for "love"
4 -> vector for "NLP"
```

Visual idea:

```text
Words:
I        love       NLP
|         |          |
v         v          v
Vectors:
x1        x2         x3
```

An embedding gives each token a vector.

But the meaning of a word depends on context.

Example 1:

```text
He sat by the river bank.
```

Here, `"bank"` means river side.

Example 2:

```text
He opened a bank account.
```

Here, `"bank"` means financial institution.

The word is the same, but the meaning is different.

So the model needs a way to update a word vector using nearby useful words.

That is what attention does.

```text
Embedding = vector for a word
Attention = update that vector using useful context
```

---

## 2. Simple Definition Of Attention

Attention is a mechanism that assigns weights to words in a sequence.

Higher weight means the word is more useful.

Lower weight means the word is less useful.

Simple definition:

```text
Attention learns which words to focus on while updating a word vector.
```

Example:

```text
The movie was not good.
```

If we only look at `"good"`, the sentiment looks positive.

But `"not"` changes the meaning.

So `"good"` should pay strong attention to `"not"`.

```text
The     movie     was     not     good
                         ^        |
                         |        |
                         +--------+
```

---

## 3. Why Attention Is Needed

Consider:

```text
The animal did not cross the road because it was tired.
```

The word `"it"` refers to:

```text
the animal
```

How do we know?

Because we look back at the useful word `"animal"`.

```text
The animal did not cross the road because it was tired
     ^                                           |
     |                                           |
     +------------------ important -------------+
```

Attention gives the model this ability.

```text
Attention = learn where to look
```

Another example:

```text
The camera is good but the battery is terrible.
```

If the question is:

```text
What is the sentiment about battery?
```

The useful words are:

```text
battery, terrible
```

The words `"camera"` and `"good"` are not the main focus for battery sentiment.

Attention helps the model focus on the useful words for the current task.

---

## 4. Attention Weights

Attention produces weights.

These weights usually add up to 1 for one query word.

Example:

```text
The battery life is amazing but the screen is dull.
```

If the focus is battery life, a possible attention row is:

```text
The      battery   life    is    amazing   but   screen   dull
0.02     0.30      0.25   0.03   0.35     0.01   0.02    0.02
```

Visual form:

```text
Word       Weight   Visual
The        0.02     .
battery    0.30     ######
life       0.25     #####
is         0.03     #
amazing    0.35     #######
but        0.01     .
screen     0.02     .
dull       0.02     .

Total      1.00
```

Important idea:

```text
One attention row = one word deciding how much to look at all words.
```

---

## 5. Query, Key, Value

Attention uses three vectors:

```text
Query
Key
Value
```

Simple meanings:

| Term | Simple meaning |
|---|---|
| Query | what I am looking for |
| Key | what I compare against |
| Value | information I will use |

Another way to remember:

```text
Query = request
Key   = match label
Value = useful content
```

Search analogy:

```text
Search query: "best phone camera"
```

The search engine compares your query with page keywords.

Then it returns useful pages.

Attention is similar:

```text
Query -> compare with Keys -> use Values
```

Most important separation:

```text
Keys are used for matching.
Values are used for information.
```

Flow:

```text
                 scores              weights
Query word  ----------------> Keys ------------+
                                               |
Values  ---------------------------------------+
                                               |
                                               v
                                   new vector for query word
```

Query and Key decide the attention weight.

Value carries the information that gets mixed into the output.

---

## 6. Q, K, V Example

Sentence:

```text
The animal did not cross the road because it was tired.
```

Focus word:

```text
it
```

The Query from `"it"` asks:

```text
Which word explains me?
```

Each Key answers:

```text
Do I match this query?
```

Each Value answers:

```text
If I am selected, what information do I contribute?
```

Table:

| Candidate word | Key role | Value role | Likely attention |
|---|---|---|---|
| animal | matches "who does it refer to?" | animal/entity information | high |
| road | weakly matches place/object | road/location information | low |
| tired | matches state/description | tired/state information | medium |

Visual:

```text
Query from "it": Which word explains me?

        compare with keys                 use values
        -----------------                 ----------
animal key  ---------------- high  ---->  animal value
road key    ---------------- low   ---->  little road value
tired key   ---------------- mid   ---->  some tired value
```

The word `"it"` becomes clearer after looking strongly at `"animal"`.

---

## 7. Where Q, K, and V Come From

Each word starts as an embedding vector.

The model learns three transformations from that embedding:

```text
word embedding
      |
      +--> Query
      +--> Key
      +--> Value
```

For one word:

```text
Embedding for one word
          |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
        Query              Key               Value
   "what I need"      "what I match"    "what I provide"
```

Example for the word `"good"`:

```text
Query: asks what context is needed
Key:   helps other words decide whether to look at "good"
Value: carries information from "good"
```

Same original embedding.

Three different roles.

---

## 8. Attention In Three Steps

Attention can be remembered as:

```text
match, weight, combine
```

Step 1: Match

```text
Compare the Query with all Keys.
```

Step 2: Weight

```text
Convert match scores into attention weights.
```

Step 3: Combine

```text
Use the weights to combine Values.
```

For the sentence:

```text
The movie was not good.
```

If the query word is `"good"`:

```text
1. MATCH
   q_good compares with every key

   q_good -> k_The     score
   q_good -> k_movie   score
   q_good -> k_was     score
   q_good -> k_not     score
   q_good -> k_good    score

2. WEIGHT
   scores -> softmax -> weights that add to 1

3. COMBINE
   weights mix the value vectors

   output_good =
       w_The   * v_The
     + w_movie * v_movie
     + w_was   * v_was
     + w_not   * v_not
     + w_good  * v_good
```

---

## 9. Attention Formula

The formula is:

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V
```

Meaning:

| Formula part | Meaning |
|---|---|
| `QK^T` | compare queries with keys |
| `/ sqrt(d_k)` | scale scores so they do not become too large |
| `softmax` | turn scores into weights |
| `V` | combine value information |

Read the formula as:

```text
compare -> scale -> softmax -> combine
```

One row of `QK^T`:

```text
one query word compared with all key words
```

One row after softmax:

```text
one attention distribution
```

---

## 10. Worked Math Example

Sentence:

```text
The movie was not good.
```

Focus word:

```text
good
```

Use only three important words to keep the math small:

```text
movie, not, good
```

Assume the model has created these vectors:

```text
Query for "good":
q_good = [1, 1]

Keys:
k_movie = [1, 0]
k_not   = [2, 2]
k_good  = [1, 2]
```

### Step 1: Dot Product Scores

```text
score(good -> movie) = q_good dot k_movie
                     = (1*1) + (1*0)
                     = 1

score(good -> not)   = q_good dot k_not
                     = (1*2) + (1*2)
                     = 4

score(good -> good)  = q_good dot k_good
                     = (1*1) + (1*2)
                     = 3
```

Highest raw score:

```text
good -> not
```

### Step 2: Scale Scores

Here:

```text
d_k = 2
sqrt(d_k) = sqrt(2) = 1.41
```

Scaled scores:

```text
movie: 1 / 1.41 = 0.71
not:   4 / 1.41 = 2.83
good:  3 / 1.41 = 2.12
```

### Step 3: Softmax Weights

Softmax converts scores into weights that add to 1.

```text
Word      Scaled score   Attention weight
movie     0.71           0.07
not       2.83           0.62
good      2.12           0.31

Total                    1.00
```

Visual:

```text
Query word: good

movie   0.07   #
not     0.62   ############
good    0.31   ######
```

Conclusion:

```text
"good" pays strong attention to "not".
```

These vectors are simple classroom numbers.

The real process is the same:

```text
dot product -> scale -> softmax
```

---

## 11. Weighted Sum Of Values

Attention weights tell the model how much of each Value to use.

Using the same example:

```text
movie = 0.07
not   = 0.62
good  = 0.31
```

Use simple value vectors:

```text
[movie-topic, negation-signal, positive-word]
```

Values:

```text
v_movie = [1, 0, 0]
v_not   = [0, 1, 0]
v_good  = [0, 0, 1]
```

Weighted sum:

```text
output_good =
    0.07 * [1, 0, 0]
  + 0.62 * [0, 1, 0]
  + 0.31 * [0, 0, 1]

output_good = [0.07, 0.62, 0.31]
```

Meaning:

```text
The updated vector for "good" contains:
- a little movie-topic information
- strong negation information from "not"
- some positive-word information from "good"
```

The new vector for `"good"` is context-aware.

It is not just `"good"` alone anymore.

It has absorbed useful information from `"not"`.

---

## 12. Self-Attention

Self-attention means:

```text
Words in the same sentence attend to each other.
```

Example:

```text
The movie was not good.
```

The word `"good"` should look at `"not"`.

```text
The     movie     was     not     good
                         ^        |
                         |        |
                         +--------+
```

Another example:

```text
He sat by the river bank.
```

Here `"bank"` should use:

```text
river
```

But in:

```text
He opened a bank account.
```

`"bank"` should use:

```text
account
```

Self-attention helps the same word get different meanings in different contexts.

```text
same word + different context = different updated vector
```

---

## 13. Attention Matrix

For a sentence with `T` words, every word can look at every word.

So the attention matrix has shape:

```text
T x T
```

Rows are query words.

Columns are words being looked at.

```text
Rows    = words asking "where should I look?"
Columns = words that can provide information
```

Example:

```text
The movie was not good
```

Attention matrix:

```text
                  Words being looked at
Query word      The   movie   was    not   good   row sum
The             0.70  0.10    0.10   0.05  0.05   1.00
movie           0.10  0.55    0.15   0.05  0.15   1.00
was             0.20  0.20    0.40   0.10  0.10   1.00
not             0.05  0.05    0.05   0.60  0.25   1.00
good            0.04  0.06    0.04   0.65  0.21   1.00
```

Focus on the last row:

```text
Query word = good

The     0.04   .
movie   0.06   #
was     0.04   .
not     0.65   #############
good    0.21   ####
```

Meaning:

```text
When updating "good", take most information from "not".
```

Every row adds up to 1.

Every row is one attention distribution.

---

## 14. Tensor Shapes

Let:

```text
B = batch size
T = number of tokens
D = embedding/vector size
```

Embeddings have shape:

```text
(B, T, D)
```

After attention, the final output also has one updated vector per token:

```text
(B, T, D)
```

Detailed shape flow:

```text
Embeddings:        (B, T, D)
Queries Q:         (B, T, d_k)
Keys K:            (B, T, d_k)
Values V:          (B, T, d_v)

Scores QK^T:       (B, T, T)
Attention weights: (B, T, T)
Output per head:   (B, T, d_v)
Final output:      (B, T, D)
```

Example:

```text
Input embeddings:  (2, 5, 16)
Attention output:  (2, 5, 16)
```

Meaning:

```text
2  = two sentences
5  = five tokens per sentence
16 = vector size
```

---

## 15. Self-Attention vs Cross-Attention

Self-attention:

```text
one sequence attends to itself
```

Example:

```text
words in a sentence look at other words in the same sentence
```

Cross-attention:

```text
one sequence attends to another sequence
```

Translation example:

```text
English: I love apples
French:  J'aime les pommes
```

When generating:

```text
pommes
```

the decoder should look at:

```text
apples
```

Visual:

```text
Decoder word: pommes
       |
       v
looks at English words:
I       love       apples
                    ^
                    |
              high attention
```

Summary:

```text
self-attention  = same sequence
cross-attention = different sequence
```

---

## 16. Masks In Attention

Masks tell attention what not to look at.

There are two important masks:

```text
Padding mask
Causal mask
```

### Padding Mask

Padding tokens are fake tokens added to make sequences the same length.

Example:

```text
I love NLP PAD PAD
```

The model should not attend to:

```text
PAD
```

Mask:

```text
I      love   NLP    PAD    PAD
keep   keep   keep   block  block
```

Padding mask blocks fake positions.

### Causal Mask

Causal mask is used in next-word prediction.

Rule:

```text
Look left, not right.
```

Example:

```text
I love deep learning
```

When predicting `"deep"`, the model can use:

```text
I love
```

It should not use:

```text
learning
```

Mask:

```text
          Can look at
          I   love  deep  learning
I         yes  no    no      no
love      yes  yes   no      no
deep      yes  yes   yes     no
learning  yes  yes   yes     yes
```

Summary:

```text
Padding mask = block fake tokens
Causal mask  = block future tokens
```

---

## 17. Multi-Head Attention

Single-head attention creates one attention pattern.

Multi-head attention creates several attention patterns in parallel.

Key idea:

```text
Single-head attention:
one set of Q, K, V -> one attention pattern -> one output

Multi-head attention:
many sets of Q, K, V -> many attention patterns -> many outputs -> combine
```

Multi-head attention is not just "look at many words."

Normal attention can also look at many words.

Multi-head attention means:

```text
The same sentence is viewed through multiple learned attention heads.
Each head has its own Q, K, and V transformations.
```

Example sentence:

```text
The movie was not good but the acting was excellent.
```

Separate heads:

```text
Input embeddings
      |
      +--> Head 1: make Q1, K1, V1 -> attention pattern 1 -> output 1
      |
      +--> Head 2: make Q2, K2, V2 -> attention pattern 2 -> output 2
      |
      +--> Head 3: make Q3, K3, V3 -> attention pattern 3 -> output 3
      |
      +--> Head 4: make Q4, K4, V4 -> attention pattern 4 -> output 4
```

For the same query word `"good"`, different heads may learn different attention rows.

```text
Query word: good

Head 1 may learn negation:
The   movie   was   not    good   but   acting   excellent
0.02  0.08    0.03  0.70   0.15   0.01  0.00     0.01

Head 2 may learn the aspect being described:
The   movie   was   not    good   but   acting   excellent
0.04  0.55    0.04  0.10   0.22   0.03  0.01     0.01

Head 3 may learn contrast or clause boundary:
The   movie   was   not    good   but   acting   excellent
0.02  0.10    0.03  0.12   0.28   0.35  0.04     0.06
```

The head outputs are combined:

```text
output_good_head_1
output_good_head_2
output_good_head_3
output_good_head_4
        |
        v
combine / concatenate
        |
        v
final updated vector for "good"
```

Simple memory line:

```text
Single head = one way of looking.
Multi-head = several learned ways of looking, combined into one final vector.
```

In real models, heads are not always easy for humans to interpret.

But multiple heads give the model several ways to capture useful relationships.

---

## 18. Complete Flow Summary

Sentence:

```text
The movie was not good
```

Step 1: Start with embeddings.

```text
The      movie     was      not      good
 x1       x2       x3       x4       x5
```

Step 2: Create Q, K, and V for each word.

```text
        q1,k1,v1  q2,k2,v2  q3,k3,v3  q4,k4,v4  q5,k5,v5
```

Step 3: For one query word, compare with all keys.

```text
Query: q_good

            k_The   k_movie   k_was   k_not   k_good
scores        .       .        .       high     medium
```

Step 4: Softmax turns scores into weights.

```text
            The     movie     was      not      good
weights     0.04    0.06      0.04     0.65     0.21
```

Step 5: Weights mix values.

```text
output_good =
  0.04*v_The + 0.06*v_movie + 0.04*v_was + 0.65*v_not + 0.21*v_good
```

Step 6: Repeat this for every word.

```text
output_The, output_movie, output_was, output_not, output_good
```

Final result:

```text
context-aware word vectors
```

---
