# Long Short-Term Memory (LSTM): Intuition, Mathematics, and Practice

A from-first-principles guide to the LSTM: what it is, why it was invented, where it shines, where it breaks, the full mathematics with derivations, and a question-and-answer section covering the conceptual sticking points that come up when you first learn it.

---

## Table of Contents

1. [Part 1: The Intuition](#part-1-the-intuition)
   - [What an LSTM Is](#what-an-lstm-is)
   - [Where ANNs, CNNs, and RNNs Fall Short](#where-anns-cnns-and-rnns-fall-short)
   - [What the LSTM Solved](#what-the-lstm-solved)
   - [The Role of LSTMs in NLP](#the-role-of-lstms-in-nlp)
   - [Weak Points of the LSTM](#weak-points-of-the-lstm)
2. [Part 2: The Mathematics in Depth](#part-2-the-mathematics-in-depth)
   - [Notation](#notation)
   - [The Forward Pass, Gate by Gate](#the-forward-pass-gate-by-gate)
   - [Why the Cell State Defeats Vanishing Gradients](#why-the-cell-state-defeats-vanishing-gradients)
   - [The Backward Pass (Backpropagation Through Time)](#the-backward-pass-backpropagation-through-time)
3. [Part 3: Questions and Answers](#part-3-questions-and-answers)

---

# Part 1: The Intuition

## What an LSTM Is

An LSTM is a type of recurrent neural network built to remember information over long stretches of a sequence. It processes a sequence one step at a time, and at each step it maintains two pieces of internal memory that it carries forward: a **cell state**, which acts as a long-term memory that travels down the sequence with only minor edits, and a **hidden state**, which is a short-term working output exposed at each step.

The defining feature of an LSTM is that it uses small learned components called **gates** to decide, at every step, what to erase from memory, what new information to write in, and what to expose as output. Because these gates are learned, the network discovers on its own what is worth remembering and for how long.

The core mental image is a conveyor belt. Information rides along the cell state from one timestep to the next. At each step, the gates make small additive edits to what is on the belt rather than rewriting it wholesale. This is what allows the signal to survive across many steps.

## Where ANNs, CNNs, and RNNs Fall Short

To understand what the LSTM adds, it helps to see the limitation of each architecture that came before it.

**A plain feedforward network (ANN)** takes a fixed input and produces a fixed output. It has no notion of order or sequence. If you feed it a sentence, it has no way to know that the third word came after the second. It cannot carry any memory of what it saw earlier.

**A convolutional network (CNN)** exploits spatial locality: nearby pixels or features are related, and the same filter is slid across the input to detect patterns wherever they occur. This is powerful for images, but a CNN uses fixed-size windows and treats the input as a static grid. Order across a long sequence is not its native language. It can catch local patterns but does not maintain an evolving memory of the whole sequence.

**A recurrent network (RNN)** was the first to carry memory forward. It processes a sequence step by step, maintaining a hidden state that summarizes everything seen so far, and it reuses the same weights at every step. In principle this lets it handle sequences of any length. In practice it fails on long-range dependencies. The reason is the **vanishing gradient problem**: during training, the error signal has to travel backward through many timesteps, and at each step it is multiplied by the same weight matrix. Repeated multiplication either shrinks the signal toward zero or blows it up. Shrinking is the common and damaging case: the network literally cannot feel the influence of inputs from many steps ago, so it cannot learn long-range structure. Its memory decays exponentially.

## What the LSTM Solved

The LSTM (Hochreiter and Schmidhuber, 1997) was designed to fix precisely the vanishing-gradient failure of the plain RNN.

The key idea is to add a separate memory channel, the cell state, that is edited by **addition** rather than by repeated matrix multiplication. In a plain RNN, the memory at each step is essentially the previous memory pushed through a weight matrix and a nonlinearity, over and over. In an LSTM, the cell state is updated by scaling the old memory (via a forget gate) and adding new information (via an input gate). Because the update is additive, gradients can flow backward across many steps almost unchanged, as long as the forget gate stays open. This is the mechanism that keeps the memory highway open and defeats the vanishing gradient.

The empirical signature of this is dramatic. On the classic "adding problem," where a network must remember two specific values marked in a long noisy sequence and output their sum, a vanilla RNN collapses to near-chance accuracy once the sequence length passes roughly thirty steps, while an LSTM maintains near-perfect accuracy across sequences of one hundred fifty steps or more. Same task, same budget; the only difference is the gated additive memory.

## The Role of LSTMs in NLP

LSTMs were the workhorse of natural language processing from roughly 2014 to 2017, before the Transformer took over. Language is inherently sequential and full of long-range dependencies: the sentiment of a sentence can hinge on a word many tokens back ("I expected to hate it, but..."), and subject-verb agreement can span a long clause. The LSTM's ability to carry information across a sequence made it the natural fit.

Concretely, LSTMs powered language modeling (predicting the next word), machine translation (the original sequence-to-sequence encoder-decoder was LSTM-based, encoding a source sentence into a state and decoding the target from it), and a wide range of tagging and classification tasks such as named-entity recognition and sentiment analysis, often using bidirectional LSTMs that read the sequence both forward and backward to build context-aware representations. They were also central to speech recognition and handwriting recognition, anywhere sequential context mattered.

## Weak Points of the LSTM

The LSTM is not without serious limitations, and these are what eventually led to its replacement by the Transformer for most large-scale NLP.

**It is sequential and cannot be parallelized over time.** Computing the state at step *t* requires the state at step *t minus one*, so timesteps must be processed in order. This prevents the training from exploiting the massive parallelism of modern hardware across the sequence dimension, which makes LSTMs slow to train on very large datasets. The Transformer processes all positions at once, and this is the single biggest reason it won.

**It still struggles with very long dependencies.** The LSTM greatly mitigates the vanishing gradient but does not fully eliminate it. Information hundreds or thousands of steps back still degrades, because everything must squeeze through the sequential bottleneck; there is no direct connection between two distant positions.

**The sequence-to-sequence bottleneck.** In the original encoder-decoder design, the entire input is compressed into a single fixed-size vector, which loses detail on long inputs. The attention mechanism was invented to patch exactly this problem, and attention eventually made the LSTM itself unnecessary.

**It is relatively heavy.** Three gates plus a cell state means many parameters and computations per step relative to what the architecture ultimately delivers.

A useful way to hold the whole history: RNN gave way to LSTM, LSTM was augmented with attention, and then attention alone (the Transformer) replaced the LSTM. Each step solved the previous one's core limitation. The LSTM solved the RNN's memory decay; the Transformer solved the LSTM's sequential bottleneck and its remaining weakness on long-range structure.

---

# Part 2: The Mathematics in Depth

This section builds the LSTM equations, explains what each one does and why it takes the form it does, and then derives the backward pass to show mathematically why the cell state protects the gradient.

## Notation

At timestep $t$ we have the following. Let $x_t$ be the input vector at step $t$, let $h_{t-1}$ be the hidden state from the previous step, and let $C_{t-1}$ be the previous cell state. The symbol $\sigma$ denotes the logistic sigmoid, $\sigma(z) = \frac{1}{1 + e^{-z}}$, which squashes any real number into the range $(0, 1)$. The symbol $\odot$ denotes elementwise (Hadamard) multiplication. Each gate has its own weight matrices $W$ and $U$ and bias $b$, where $W$ multiplies the input and $U$ multiplies the previous hidden state.

## The Forward Pass, Gate by Gate

An LSTM cell computes four quantities from the same two inputs ($x_t$ and $h_{t-1}$), then combines them. We take each in turn.

### The forget gate

$$
f_t = \sigma\left(W_f x_t + U_f h_{t-1} + b_f\right)
$$

The forget gate decides which parts of the old cell state to keep and which to erase. It looks at the current input and the previous hidden state, passes them through a linear layer, and squashes the result with a sigmoid so that every element lands between 0 and 1. Think of the output as a vector of dials, one per memory slot: a value near 1 means "keep this memory fully," a value near 0 means "erase this memory." The forget gate on its own can only remove or preserve information; it cannot add anything.

### The input gate and the candidate

$$
i_t = \sigma\left(W_i x_t + U_i h_{t-1} + b_i\right)
$$

$$
\tilde{C}_t = \tanh\left(W_C x_t + U_C h_{t-1} + b_C\right)
$$

Two separate things happen here, and keeping them distinct is important. The **candidate** $\tilde{C}_t$ is the proposed new information to write into memory. It uses $\tanh$ rather than sigmoid because it represents actual content, which can be positive or negative, so its range is $(-1, 1)$. The **input gate** $i_t$ is a sigmoid dial in $(0, 1)$ that decides how much of that candidate is actually allowed into the cell state, slot by slot. The candidate carries the content; the input gate regulates the flow.

### The cell state update

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

This is the heart of the LSTM. Read it as two halves. The first term, $f_t \odot C_{t-1}$, takes the old memory and scales it by the forget gate: if the forget gate is near 1, the old memory passes through untouched. The second term, $i_t \odot \tilde{C}_t$, takes the new candidate scaled by the input gate and **adds** it on.

The single most important word in the whole architecture is that plus sign. In a plain RNN the memory is transformed by matrix multiplication at every step, which is what causes the exponential decay. Here the cell state is updated by a running sum. As we will see in the backward pass, addition lets gradients flow backward almost unchanged, which is exactly what keeps long-range memory alive.

### The output gate and the hidden state

$$
o_t = \sigma\left(W_o x_t + U_o h_{t-1} + b_o\right)
$$

$$
h_t = o_t \odot \tanh\left(C_t\right)
$$

The cell state is the full internal memory, and it is kept private. What the rest of the network sees is the hidden state $h_t$, a filtered view of that memory. The output gate $o_t$ is a sigmoid dial deciding which parts of the current memory to expose right now. The cell state is first squashed through $\tanh$ to keep the exposed values bounded, and then the output gate selects which parts pass through. This design is powerful because the network can hold a piece of information in the cell state for many steps with the output gate closed on it, and only reveal it later when it becomes relevant.

### Summary of the forward pass

$$
\begin{aligned}
f_t &= \sigma\left(W_f x_t + U_f h_{t-1} + b_f\right) \\
i_t &= \sigma\left(W_i x_t + U_i h_{t-1} + b_i\right) \\
\tilde{C}_t &= \tanh\left(W_C x_t + U_C h_{t-1} + b_C\right) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma\left(W_o x_t + U_o h_{t-1} + b_o\right) \\
h_t &= o_t \odot \tanh\left(C_t\right)
\end{aligned}
$$

## Why the Cell State Defeats Vanishing Gradients

To see the point of the additive update concretely, compare how a number is treated across time in a plain RNN versus an LSTM cell state.

In a plain RNN, the hidden state is repeatedly multiplied by a weight matrix. If you strip it down to a single dimension, the influence of an early input on a later state scales like $w^{k}$ after $k$ steps. If $w = 0.9$, then after one hundred steps the factor is $0.9^{100} \approx 0.00003$, which is effectively zero. That is the vanishing gradient: the early input has no measurable effect on the later output, so the network cannot learn the dependency.

In an LSTM, the cell state update is $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. Along the cell-state path, the previous cell state is not multiplied by a learned weight matrix; it is multiplied by the forget gate $f_t$ and then added to the new content. When the forget gate stays near 1, the cell state passes essentially unchanged, so the chain of dependence does not decay. This is the "constant error carousel" the original paper described: an information channel where the gradient can ride backward without being repeatedly shrunk.

## The Backward Pass (Backpropagation Through Time)

Now we derive why the gradient survives. Training minimizes a loss $L$, and we need the gradient of the loss with respect to every parameter. Because parameters are shared across timesteps, we sum contributions over all steps. The crucial quantity is how the gradient with respect to the cell state propagates backward from step $t$ to step $t-1$.

Consider the gradient of the loss with respect to the cell state, written $\delta C_t = \frac{\partial L}{\partial C_t}$. The cell state $C_t$ influences the loss through two routes: directly through the hidden state $h_t$ at the current step, and indirectly through the next cell state $C_{t+1}$, because $C_{t+1}$ depends on $C_t$.

Start from the hidden state. Since $h_t = o_t \odot \tanh(C_t)$, the contribution of the current step to $\delta C_t$ is

$$
\frac{\partial h_t}{\partial C_t} = o_t \odot \left(1 - \tanh^2(C_t)\right)
$$

using the fact that the derivative of $\tanh(z)$ is $1 - \tanh^2(z)$.

Now the recurrent route. The cell-state update is $C_{t+1} = f_{t+1} \odot C_t + i_{t+1} \odot \tilde{C}_{t+1}$. Differentiating $C_{t+1}$ with respect to $C_t$, and treating the gates as the dominant path, the leading term is

$$
\frac{\partial C_{t+1}}{\partial C_t} = f_{t+1}
$$

This is the key result. The derivative of the next cell state with respect to the current cell state is, to leading order, just the forget gate. Putting the two routes together, the recursive relation for the cell-state gradient is

$$
\delta C_t = \frac{\partial L}{\partial h_t} \odot o_t \odot \left(1 - \tanh^2(C_t)\right) + \delta C_{t+1} \odot f_{t+1}
$$

Read the second term carefully. As the gradient travels backward from step $t+1$ to step $t$, it is multiplied by $f_{t+1}$, the forget gate, and nothing else. There is no repeated multiplication by a weight matrix and no forced squashing through a nonlinearity along this path. If the forget gates across a span of timesteps are near 1, then the product of the forget gates across that span is near 1, and the gradient arrives at the early step essentially intact.

Contrast this with the plain RNN, where the analogous backward step multiplies by the recurrent weight matrix (and a nonlinearity derivative) at every step, producing the exponential shrink. The LSTM replaces that toxic repeated multiplication with a multiplication by a learned gate that the network can drive toward 1 whenever it needs to preserve information. That single structural change is the whole reason LSTMs can learn long-range dependencies.

The gradients then flow from the cell state and hidden state into each gate's parameters in the usual way. For example, since $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ and it enters the update through $f_t \odot C_{t-1}$, the gradient with respect to the forget-gate pre-activation is $\delta C_t \odot C_{t-1} \odot f_t \odot (1 - f_t)$, using the sigmoid derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. The gradients for the input gate, candidate, and output gate follow the same pattern, each combining the upstream gradient with the local derivative of its own activation. In practice a framework computes all of this automatically, but the important conceptual takeaway is the forget-gate term in the cell-state recursion, because that is where the vanishing gradient is defeated.

---

# Part 3: Questions and Answers

This section collects the conceptual questions that naturally arise when learning the LSTM, with direct answers.

### Q: What is an LSTM and how is it different from an ANN, CNN, or RNN?

An LSTM is a recurrent network that carries a protected long-term memory (the cell state) edited by learned gates. An ANN has no memory and no notion of sequence order. A CNN detects local patterns with sliding filters but treats the input as a static grid rather than an ordered sequence. A plain RNN does carry a memory forward, but it rewrites that memory by matrix multiplication at every step, which causes it to forget across long spans. The LSTM keeps a separate additive memory channel and uses gates to control it, which is what lets it remember over long sequences where the RNN cannot.

### Q: What gap did the LSTM address?

The vanishing gradient problem in plain RNNs. When an RNN is trained on long sequences, the error signal is multiplied by the same weight matrix at every backward step, so it shrinks exponentially and the network cannot learn dependencies between distant positions. The LSTM introduces an additively updated cell state, so the gradient can travel backward across many steps scaled only by the forget gates rather than by repeated matrix multiplication. This keeps the long-range signal alive.

### Q: What role does the LSTM play in NLP?

It was the dominant sequence model for NLP from about 2014 to 2017. It powered language modeling, machine translation (the original encoder-decoder sequence-to-sequence models), and tagging and classification tasks like named-entity recognition and sentiment analysis, frequently as a bidirectional LSTM that reads context from both directions. It was the first architecture to make context-aware sequence understanding practical at scale, and it held that role until attention and the Transformer replaced it.

### Q: What are the problems with the LSTM?

Four main ones. It is sequential and cannot be parallelized across the time dimension, which makes it slow to train at scale. It still degrades on very long dependencies because everything passes through a sequential bottleneck with no direct link between distant positions. The classic encoder-decoder design compresses the whole input into one fixed-size vector, which loses information on long inputs. And it is relatively heavy, with three gates and a cell state per step. The Transformer addressed the parallelism and long-range weaknesses, which is why it took over.

### Q: Does the cell state just keep adding at every step, and when do we choose to forget, input, or output?

It is not pure accumulation, and you do not choose the gates; the network computes them fresh at every step. The update is $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, which first scales down the old memory with the forget gate and then adds the new content scaled by the input gate. So each step both removes and adds. The forget, input, and output gates are all recomputed at every timestep from the current input and the previous hidden state; they are learned functions, not manual switches. At one step the forget gate might be near 1 (hold everything), and at the next step near 0 (dump most of it), depending entirely on what the input at that step calls for.

### Q: What is each gate doing on its own?

Each gate is a small learned layer that outputs a vector of values between 0 and 1, one per memory slot, acting as a panel of dials. The **forget gate** multiplies the old cell state and decides which existing memories to keep or erase; on its own it can only remove, never add. The **input gate** decides how much of the newly proposed candidate to write in; it regulates flow but does not create content. The **output gate** decides which parts of the current memory to expose as the hidden state; it filters the view without changing what is stored. None of the gates carry information themselves. They are all pure traffic control in the range 0 to 1. The only component that carries actual content is the candidate, which uses $\tanh$ and ranges from -1 to 1. The gates are not fully independent, however: all three read the same input and previous hidden state and are trained together toward one objective, so they learn coordinated behavior.

### Q: Why did the recurrent models get stuck at fifty percent accuracy on padded text while the CNN worked?

Because of an interaction between right-padding and reading the last timestep. Reviews were padded on the right to a fixed length, so most sequences ended in a long run of padding tokens. Taking the hidden state at the very last position meant reading the state after processing hundreds of padding steps, by which point the real signal had washed out. All three recurrent models failed identically because they all read the last timestep the same way, which is the signature of a shared upstream problem rather than an architecture flaw. The CNN was unaffected because it max-pools over all positions and simply grabs the sentiment features wherever they occur, ignoring the padding tail. The fix is to pack the padded sequence so the recurrent net stops at each review's true final word and reads the genuine last hidden state.

### Q: Why did training accuracy reach nearly one while validation accuracy stalled?

That is overfitting. A model with far more parameters than the small dataset can support memorizes the training examples instead of learning generalizable patterns. The tell is the growing gap between train and validation accuracy, together with validation loss climbing even as training loss falls. More epochs do not help past that point; they widen the gap. The remedies are regularization (dropout, weight decay), pretrained embeddings so there is less to fit from scratch, more data, and early stopping to keep the checkpoint from the best validation epoch rather than the overtrained final one.

### Q: Why is training an LSTM to get good results this hard?

Because most of the difficulty is not in the LSTM itself. The model was working the whole time; the struggles were about the surrounding pipeline: how the data is padded, how the output is read, how much data there is, and when to stop training. This is the general truth of applied deep learning, where the model is often the easy part and the data plumbing is the hard part. The debugging habits that come out of it transfer to every model: if several architectures fail identically, the problem is upstream; if one model works when others do not, the contrast tells you where to look; if train accuracy climbs while validation stalls, it is overfitting rather than undertraining; and overfitting a single batch first confirms the code can learn at all before you blame the task.

### Q: Why did the long-dependency experiment fail to separate the RNN from the LSTM, and how was it fixed?

The first version used a single strong cue token followed by repetitive filler, and asked the model to recall the cue's class. That task is too easy in the wrong way: both models solve it trivially at short lengths and then both collapse together once the sequence gets long enough that the gradient to the cue dies for both at once. There is no regime where the RNN is stressed but the LSTM copes, so the curves never separate. The fix was to switch to the adding problem from the original 1997 LSTM paper, where the model must find two marked real-valued numbers anywhere in a long noisy sequence and output their sum. That task requires precise long-range memory with no shortcut, which is exactly the regime where the LSTM's gated cell state beats the RNN. With that task the separation is clean: the RNN collapses to near-chance by around thirty steps while the LSTM holds near-perfect accuracy through one hundred fifty steps.

### Q: Did the regularization actually work, and how do you know it was not just smaller numbers?

Yes, once the gap was measured at the best validation epoch rather than the final epoch. Without regularization the model reached perfect training accuracy with validation around sixty-three percent, a gap of about thirty-seven points. With dropout and weight decay the training accuracy at the best epoch was much lower while validation stayed about the same, cutting the gap to roughly nine points, a fourfold reduction. The reason this is a genuine result and not a numerical trick is the mechanism: validation accuracy was essentially unchanged between the two runs, so regularization did not improve what the model achieved; it closed the gap by pulling training accuracy down toward validation, which is exactly what preventing memorization looks like. On a small dataset regularization stops the model from wasting capacity memorizing noise rather than raising the ceiling.
