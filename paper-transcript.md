# MDGAN Reference code

Simple implementation of the research paper "MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets".

The focus of this codebase is to provide implementation as close to the original as possible, as there are no public repositories with full implementation of this paper. The idea is to use just minimal dependencies with a focus on performance. The ML library of choice is PyTorch together with the scalability of Ray library.

## Background on Generative Adversarial Networks

The particularity of GANs as initially presented in [1] is that their training phase is unsupervised, i.e., no description labels are required to learn from the data. A classic GAN is composed of two elements: a generator $\mathcal{G}$ and a discriminator $\mathcal{D}$. Both are deep neural networks. The generator takes as input a noise signal (e.g., random vectors of size $k$ where each entry follows a normal distribution $\mathcal{N}(0, 1)$) and generates data with the same format as training dataset data (e.g., a picture of 128x128 pixels and 3 color channels). The discriminator receives as input either some data from two sources: from the generator or from the training dataset. The goal of the discriminator is to guess from which source the data is coming from. At the beginning of the learning phase, the generator generates data from a probability distribution and the discriminator quickly learns how to differentiate that generated data from the training data. After some iterations, the generator learns to generate data which are closer to the dataset distribution. If it eventually turns out that the discriminator is not able to differentiate both, this means that the generator has learned the distribution of the data in the training dataset (and thus has learned an unlabeled dataset in an unsupervised way).

Formally, let a given training dataset be included in the data space $X$, where $x$ in that dataset follows a distribution probability $P_{\text{data}}$. A GAN, composed of generator $\mathcal{G}$ and discriminator $\mathcal{D}$, tries to learn this distribution. As proposed in the original GAN paper [1], we model the generator by the function $G_w : \mathbb{R}^\ell \to X$, where $w$ contains the parameters of its DNN $G_w$ and $\ell$ is fixed. Similarly, we model the discriminator by the function $\mathcal{D}\theta : X \to [0, 1]$ where $\mathcal{D}\theta(x)$ is the probability that $x$ is a data from the training dataset, and $\theta$ contains the parameters of the discriminator $\mathcal{D}_\theta$. Writing $\log$ for the logarithm to the base 2, the learning consists in finding the parameters $w^*$ for the generator:

$$ w^* = \arg\min_w \max_\theta (A_\theta + B_{\theta, w}), $$

with

$$ A_\theta = \mathbb{E}_{x \sim P{\text{data}}} [\log \mathcal{D}_\theta(x)] $$

and

$$ B_{\theta, w} = \mathbb{E}_{z \sim \mathcal{N}\ell} [\log (1 - \mathcal{D}_\theta(G_w(z)))]. $$

where $z \sim \mathcal{N}\ell$ means that each entry of the $\ell$-dimensional random vector $z$ follows a normal distribution with fixed parameters. In this equation, $\mathcal{D}$ adjusts its parameters $\theta$ to maximize $A\theta$, i.e., the expected good classification on real data, and $B_{\theta, w}$, the expected good classification on generated data. $\mathcal{G}$ adjusts its parameters $w$ to minimize $B_{\theta, w}$ ($w$ does not have impact on $A$), which means that it tries to minimize the expected good classification of $\mathcal{D}$ on generated data. The learning is performed by iterating two steps, named the discriminator learning step and the generator learning step, as described in the following.

### 1. Discriminator learning:

The first step consists in learning $\theta$ given a fixed $G_w$. The goal is to approximate the parameters $\theta$ which maximize $A_\theta + B_{\theta, w}$ with the actual $w$. This step is performed by a gradient descent (generally using the Adam optimizer [16]) of the following discriminator error function $J_{disc}$ on parameters $\theta$:

$$ J_{disc}(X_r, X_g) = \tilde{A}(X_r) + \tilde{B}(X_g), $$

with

$$ \tilde{A}(X_r) = \frac{1}{b} \sum_{x \in X_r} \log(\mathcal{D}\theta(x)); \quad \tilde{B}(X_g) = \frac{1}{b} \sum_{x \in X_g} \log(1 - \mathcal{D}_\theta(x)), $$

where $X_r$ is a batch of $b$ real data drawn randomly from the training dataset and $X_g$ a batch of $b$ generated data from $\mathcal{G}$. In the original paper [1], the authors propose to perform few gradient descent iterations to find a good $\theta$ against the fixed $G_w$.

### 2. Generator learning:

The second step consists in adapting $\mathbf{w}$ to the new parameters $\boldsymbol{\theta}$. As done for step 1), it is performed by a gradient descent of the following error function $J_{\text{gen}}$ on generator parameters $\mathbf{w}$:

$$ J_{\text{gen}}(Z_g) = \tilde{B}\left({\mathcal{G}_{\mathbf{w}}(z) \mid z \in Z_g}\right) = \frac{1}{b} \sum_{x \in {\{\mathcal{G}_{\mathbf{w}}(z) \mid z \in Z_g\}}} \log(1 - \mathcal{D}_{\boldsymbol{\theta}}(x)) = \frac{1}{b} \sum_{z \in Z_g} \log(1 - \mathcal{D}_{\boldsymbol{\theta}}(\mathcal{G}_{\mathbf{w}}(z))) $$

where $Z_g$ is a sample of $b$ $\ell$–dimensional random vectors generated from $\mathcal{N}_\ell$. Contrary to discriminator learning step, this step is performed only once per iteration.

By iterating those two steps a significant amount of times with different batches (see e.g., [1] for convergence related questions), the GAN ends up with a $\mathbf{w}$ which approximates $\mathbf{w}^*$ well. Such as for standard deep learning, guarantees of convergence are weak [17]. Despite this very recent breakthrough, there are lots of alternative proposals to learn a GAN (e.g., more details can be found in [18], [19], and [20]).

## Distributed computation setup for GANs
Before we present MD-GAN in the next Section, we introduce the distributed computation setup considered in this paper, and an adaptation of federated learning to GANs.

### a) Learning over a spread dataset:
We consider the following setup. $N$ workers (possibly from several datacenters [8]) are each equipped with a local dataset composed of $m$ samples (each of size $d$) from the same probability distribution $P_{\text{data}}$ (e.g., requests to a voice assistant, holiday pictures). Those local datasets will remain in place (i.e., will not be sent over the network). We denote by $\mathcal{B} = \bigcup_{n=1}^N \mathcal{B}_n$ the entire dataset, with $\mathcal{B}_n$ the dataset local to worker $n$. We assume in the remaining of the paper that the local datasets are i.i.d. on workers, that is to say that there are no bias in the distribution of the data on one particular worker node.

The assumption on the fixed location of data shares is complemented by the use of the parameter server framework we are now presenting.

### b) The parameter server framework:
Despite the general progress of distributed computing towards serverless operation even in datacenters (e.g., use of the gossip paradigm as in Dynamo [21] back in 2007), the case of deep learning systems is specific. Indeed, the amounts of data required to train a deep learning model, and the very iterative nature of the learning tasks (learning on batches of data, followed by operations of back-propagations) makes it necessary to operate in a parallel setup, with the use of a central server. Introduced by Google in 2012 [22], the parameter server framework uses workers for parallel processing, while one or a few central servers are managing shared states modified by those workers (for simplicity, in the remaining of the paper, we will assume the presence of a single central server). The method aims at training the same model on all workers using their given data share, and to synchronize their learning results with the server at each iteration, so that this server can update the model parameters.

Note that more distributed approaches for deep learning, such as gossip-based computation [23], [24], have not yet proven to work efficiently on the data scale required for modern applications; we thus leverage a variant of parameter server framework as our computation setup.

### c) FL-GAN: adaptation of federated learning to GANs

By the design of GANs, a generator and a discriminator are two separate elements that are yet tightly coupled; this fact makes it nevertheless possible to consider adapting a known computation method, that is generally used for training a single deep neural network.  
Federated learning [27] proposes to train a machine learning model, and in particular a deep neural network, on a set of workers. It follows the parameter server framework, with the particularity that workers perform numerous local iterations between each communication to the server (i.e., a round), instead of sending small updates. All workers are not necessarily active at each round; to reduce conflicting updates, all active workers synchronize their model with the server at the beginning of each round.

In order to compare MD-GAN to a federated learning type of setup, we propose an adapted version of federated learning to GANs. This adaptation considers the discriminator $\mathcal{D}$ and generator $\mathcal{G}$ on each worker as one computational object to be treated atomically. Workers perform iterations locally on their data and every $E$ epochs (i.e., each worker passes $E$ times the data in their GAN) they send the resulting parameters to the server. The server in turn averages the $\mathcal{G}$ and $\mathcal{D}$ parameters of all workers, in order to send updates to those workers at the next iteration. We name this adapted version FL-GAN; it is depicted by Figure 1 b).

We now detail MD-GAN, our proposal for the learning of GANs over workers and their local datasets.

## THE MD-GAN ALGORITHM

### A. Design rationale

To diminish computation on the workers, we propose to operate with a single $\mathcal{G}$, hosted on the server. That server holds parameters $\mathbf{w}$ for $\mathcal{G}$; data shares are split over workers. To remove part of the burden from the server, discriminators are solely hosted by workers, and move in a peer-to-peer fashion between them. Each worker $n$ starts with its own discriminator $\mathcal{D}_n$ with parameters $\theta_n$. Note that the architecture and initial parameters of $\mathcal{D}_n$ could be different on every worker $n$; for simplicity, we assume that they are the same. This architecture is presented on Figure 1 a).

The goal for GANs is to train generator $\mathcal{G}$ using $\mathcal{B}$. In MD-GAN, the $\mathcal{G}$ on the server is trained using the workers and their local shares. It is a 1-versus-N game where $\mathcal{G}$ faces all $\mathcal{D}_n$, *i.e.*, $\mathcal{G}$ tries to generate data considered as real by all workers. Workers use their local datasets $\mathcal{B}_n$ to differentiate generated data from real data. Training a generator is an iterative process; in MD-GAN, a *global learning iteration* is composed of four steps:

- The server generates a set $K$ of $k$ batches $K = \{X^{(1)}, \ldots, X^{(k)}\}$, with $k \leq N$. Each $X^{(i)}$ is composed of $b$ data generated by $\mathcal{G}$. The server then selects, for each worker $n$, two distinct batches, say $X^{(i)}$ and $X^{(j)}$, which are sent to worker $n$ and locally renamed as $X^{(g)}_n$ and $X^{(d)}_n$. The way in which the two distinct batches are selected is discussed in Section IV-B1.
- Each worker $n$ performs $L$ learning iterations on its discriminator $\mathcal{D}_n$ (see Section II-1) using $X^{(d)}_n$ and $X^{(r)}_n$, where $X^{(r)}_n$ is a batch of real data extracted locally from $\mathcal{B}_n$.
- Each worker $n$ computes an error feedback $F_n$ on $X^{(g)}_n$ by using $\mathcal{D}_n$ and sends this error to the server. We detail in Section IV-B2 the computation of $F_n$.
- The server computes the gradient of $J_{\text{gen}}$ for its parameters $\mathbf{w}$ using all the $F_n$ feedbacks. It then updates its parameters with the chosen optimizer algorithm (e.g., Adam [16]).

### Table I: Table of notations

| Notation         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| $\mathcal{G}$    | Generator                                                                   |
| $\mathcal{D}$    | Discriminator                                                               |
| $N$              | Number of workers                                                           |
| $C$              | Central server                                                              |
| $W_n$            | Worker $n$                                                                  |
| $P_{\text{data}}$| Data distribution                                                           |
| $P_{\mathcal{G}}$| Distribution of generator $\mathcal{G}$                                     |
| $\mathbf{w}$ (resp. $\boldsymbol{\theta}$) | Parameters of $\mathcal{G}$ (resp. $\mathcal{D}$) |
| $w_i$ (resp. $\theta_i$) | $i$-th parameter of $\mathcal{G}$ (resp. $\mathcal{D}$)             |
| $\mathcal{B}$    | Distributed training dataset                                                |
| $\mathcal{B}_n$  | Local training dataset on worker $n$                                        |
| $m$              | Number of objects in a local dataset $\mathcal{B}_n$                        |
| $d$              | Object size (e.g., image in Mb)                                             |
| $b$              | Batch size                                                                  |
| $I$              | Number of training iterations                                               |
| $K$              | The set of all batches $X^{(1)}, \ldots, X^{(k)}$ generated by $\mathcal{G}$ during one iteration |
| $F_n$            | The error feedback computed by worker $n$                                   |
| $E$              | Number of local epochs before swapping                                      |

Moreover, every $E$ epochs, workers start a peer-to-peer swapping process for their discriminators, using function $\text{Swap}()$. The pseudo-code of MD-GAN, including those steps, is presented in Algorithm 1.

Note that extra workers can enter the learning task if they enter with a pre-trained discriminator (e.g., a copy of another worker discriminator); we discuss worker failures in Section V.

### B. The generator learning procedure (server-side)

The server hosts generator $\mathcal{G}$ with its associated parameters $\mathbf{w}$. Without loss of generality, this paper exposes the training of GANs for image generation; the server generates new images to train all discriminators and updates $\mathbf{w}$ using error feedbacks.

#### 1) Distribution of generated batches:

Every global iteration, $\mathcal{G}$ generates a set of $k$ batches $K = \{X^{(1)}, \ldots, X^{(k)}\}$

of size $b$. Each participating worker $n$ is sent two batches among $K$, $X^{(g)}_n$ and $X^{(d)}_n$. This two-batch generation design is required, for the computation of the gradients for both $\mathcal{D}$ and $\mathcal{G}$ on separate data (such as for the original GAN design [1]). A possible way to distribute the $X^{(i)}$ among the $N$ workers could be to set $X^{(g)}_n = X_{((n \mod k)+1)}$ and $X^{(d)}_n = X_{(((n+1) \mod k)+1)}$ for $n = 1, \ldots, N$.

#### 2) Update of generator parameters:

Every global iteration, the server receives the error feedback $F_n$ from every worker $n$, corresponding to the error made by $\mathcal{G}$ on $X^{(g)}_n$. More formally, $F_n$ is composed of $b$ vectors $\{e_{n_1}, \ldots, e_{n_b}\}$, where $e_{n_i}$ is given by

$$
e_{n_i} = \frac{\partial \tilde{B}(X^{(g)}_n)}{\partial x_i},
$$

with $x_i$ the $i$-th data of batch $X^{(g)}_n$. The gradient $\Delta \mathbf{w} = \frac{\partial \tilde{B}\left(\bigcup_{n=1}^N X^{(g)}_n\right)}{\partial \mathbf{w}}$ is deduced from all $F_n$ as

$$
\Delta w_j = \frac{1}{N b} \sum_{n=1}^N \sum_{x_i \in X^{(g)}_n} e_{n_i} \frac{\partial x_i}{\partial w_j},
$$

with $\Delta w_j$ the $j$-th element of $\Delta \mathbf{w}$. The term $\partial x_i / \partial w_j$ is computed on the server. Note that $\bigcup_{n=1}^N X^{(g)}_n = \{G_w(z) \mid z \in Z_g\}$. Minimizing $\tilde{B}\left(\bigcup_{n=1}^N X^{(g)}_n\right)$ is thus equivalent to minimize $J_{\text{gen}}(Z_g)$. Once the gradients are computed, the server is able to update its parameters $\mathbf{w}$. We thus choose to merge the feedback updates through an averaging operation, as it is the most common way to aggregate updates processed in parallel [28], [22], [29], [30]. Using the Adam optimizer [16], parameter $w_i \in \mathbf{w}$ at iteration $t$, denoted by $w_i(t)$ here, is computed as follows:

$$
w_j(t) = w_j(t-1) + \text{Adam}(\Delta w_j),
$$

---

1. **Worker Procedure**

    1. Initialize $\theta_n$ for $\mathcal{D}_n$
    2. For $i \leftarrow 1$ to $I$ do:
        1. $X_n^{(r)} \leftarrow \text{Samples}(\mathcal{B}_n, b)$
        2. $X_n^{(g)}, X_n^{(d)} \leftarrow \text{ReceiveBatches}(C)$
        3. For $l \leftarrow 0$ to $L$ do:
            1. $\mathcal{D}_n \leftarrow \text{DiscLearningStep}(J_{\text{disc}}, \mathcal{D}_n)$
        4. $F_n \leftarrow \left\{ \frac{\partial \tilde{B}(X_n^{(g)})}{\partial x_i} \mid x_i \in X_n^{(g)} \right\}$
        5. $\text{Send}(C, F_n)$  // Send $F_n$ to server
        6. If $i \bmod (\frac{mE}{b}) = 0$ then:
            1. $\mathcal{D}_n \leftarrow \text{Swap}(\mathcal{D}_n)$
    3. End for

2. **Swap Procedure**

    1. $W_l \leftarrow \text{GetRandomWorker}()$
    2. $\text{Send}(W_l, \mathcal{D}_n)$  // Send $\mathcal{D}_n$ to worker $W_l$
    3. $\mathcal{D}_n \leftarrow \text{ReceiveD}()$  // Receive a new discriminator from another worker
    4. Return $\mathcal{D}_n$

3. **Server Procedure**

    1. Initialize $w$ for $\mathcal{G}$
    2. For $i \leftarrow 1$ to $I$ do:
        1. For $j \leftarrow 0$ to $k$ do:
            1. $Z_j \leftarrow \text{GaussianNoise}(b)$
            2. $X^{(j)} \leftarrow \{ \mathcal{G}_w(z) \mid z \in Z_j \}$
        2. $X_n^{(g)}, X_n^{(d)} \leftarrow \text{Split}(X^{(1)}, \ldots, X^{(k)})$ for $n = 1, \ldots, N$
        3. For $n \leftarrow 1$ to $N$ do:
            1. $\text{Send}(W_n, (X_n^{(d)}, X_n^{(g)}))$
        4. $F_1, \ldots, F_N \leftarrow \text{GetFeedbackFromWorkers}()$
        5. Compute $\Delta w$ according to $F_1, \ldots, F_N$
        6. For $w_i \in w$ do:
            1. $w_i \leftarrow w_i + \text{Adam}(\Delta w_i)$
    3. End for

#### 3) Workload at the server:

Placing the generator on the server increases its workload. It generates $k$ batches of $b$ data using $\mathcal{G}$ during the first step of a global iteration, and then receives $N$ error feedbacks of size $bd$ in the third step. The batch generation requires $kbG_{op}$ floating point operations (where $G_{op}$ is the number of floating operations to generate one data object with $\mathcal{G}$) and a memory of $kbG_a$ (with $G_a$ the number of neurons in $\mathcal{G}$). For simplicity, we assume that $G_{op} = O(|w|)$ and that $G_a = O(|w|)$. Consequently the batch generation complexity is $O(kb|w|)$. The merge operation of all feedbacks $F_n$ and the gradient computations imply a memory and computational complexity of $O(b(dN + k|w|))$.

#### 4) The complexity vs. data diversity trade-off.

At each global iteration, the server generates $k$ batches, with $k < N$. If $k = 1$, all workers receive and compute their feedback on the same training batch. This reduces the diversity of feedbacks received by the generator but also reduces the server workload. If $k = N$, each worker receives a different batch, thus no feedback has conflict on some concurrently processed data. In consequence, there is a trade-off regarding the generator workload: because $k = N$ seems cumbersome, we choose $k = 1$ or $k = \lceil \log(N) \rceil$ for the experiments, and assess the impact of those values on final model performances.

### C. The learning procedure of discriminators (worker-side)

Each worker $n$ hosts a discriminator $\mathcal{D}_n$ and a training dataset $\mathcal{B}_n$. It receives batches of generated images split in two parts: $X_n^{(d)}$ and $X_n^{(g)}$. The generated images $X_n^{(d)}$ are used for training $\mathcal{D}_n$ to discriminate those generated images from real images. The learning is performed as a classical deep learning operation on a standalone server [1]. A worker $n$ computes the gradient $\Delta \theta_n$ of the error function $J_{disc}$ applied to the batch of generated images $X_n^{(d)}$, and a batch or real image $X_n^{(r)}$ taken from $\mathcal{B}_n$. As indicated in Section II-1, this operation is iterated $L$ times. The second batch $X_n^{(g)}$ of generated images is used to compute the error term $F_n$ of generator $\mathcal{G}$. Once computed, $F_n$ is sent to the server for the computation of gradients $\Delta w$.

#### 1) The swapping of discriminators:

Each discriminator $n$ solely uses $\mathcal{B}_n$ to train its parameters $\theta_n$. If too many iterations are performed on the same local dataset, the discriminator tends to over specialize (which decreases its capacity of generalization). This effect, called *overfitting*, is avoided in MD-GAN by swapping the parameters of discriminators $\theta_n$ between workers after $E$ epochs. The swap is implemented in a gossip fashion, by choosing randomly for every worker another worker to send its parameters to.

#### 2) Workload at workers:

The goal of MD-GAN is to reduce the workload of workers without moving data shares out of their initial location. Compared to our proposed adapted federated learning method FL-GAN, the generator task is deported on the server. Workers only have to handle their discriminator parameters $\theta_n$ and to compute error feedbacks after $L$ local iterations. Every global iteration, a worker performs $2bD_{op}$ floating point operations (where $D_{op}$ is the number of floating point operations for a feed-forward step of $\mathcal{D}$ for one data object). The memory used at a worker is $O(|\theta|)$.

### D. The characteristic complexities of MD-GAN

#### 1) Communication complexity:

In the MD-GAN algorithm there are three types of communications:

- **Server to worker communication:** the server sends its $k$ batches of generated images to workers at the beginning of global iterations. The number of generated images is $kb$ (with $k \leq N$), but only two batches are sent per worker. The total communication from the server is thus $2bdN$ (i.e., $2bd$ per worker).

- **Worker to server communications:** after computing the generator errors on $X_n^{(g)}$, all workers send their error term $F_n$ to the server. The size of error term is $bd$ per worker, because solely one float is required for each feature of the data.

- **Worker to worker communications:** after $E$ local epochs, each discriminator parameters are swapped. Each worker sends a message of size $|\theta_n|$, and receives a message of the same size (as we assume for simplicity that discriminator models on workers have the same architecture).

Communication complexities are summarized in Table III, for both MD-GAN and FL-GAN. Table IV instantiates those complexities with the actual quantities of data measured for the experiment on the CIFAR10 dataset. The first observation is that MD-GAN requires server to workers communication at every iteration, while FL-GAN performs $mE/b$ iterations in between two communications. Note that the size of worker-server communications depends on the GAN parameters ($\theta$ and $w$) for FL-GAN, whereas it depends on the size of data objects and on the batch size in MD-GAN. It is particularly interesting to choose a small batch size, especially since it is shown by Gupta et al. [31] that in order to hope for good performances in the parallel learning of a model (as discriminators in MD-GAN), the batch size should be inversely proportional to the number of workers $N$. When the size of data is around the number of parameters of the GAN (such as in image applications), the MD-GAN communications may be expensive. For example, GoogLeNet [32] analyzes images of $224 \times 224$ pixels in RGB (150,528 values per data) with less than 6.8 millions of parameters.

We plotted on Figure 2 an analysis of the maximum ingress traffic ($x$-axis) of the FL-GAN and MD-GAN schemes, for a single iteration, and depending on chosen batch size ($y$-axis). This corresponds for FL-GAN to a worker-server communication, and for MD-GAN for both worker-server and worker-worker communications during an iteration. Plain lines depict the ingress traffic at workers, while dotted lines the traffic at the server; these quantities can help to dimension the network capabilities required for the learning process to take place. Note the log-scale on both axis.

As expected the FL-GAN traffic is constant, because the communications depends only on the model sizes that constitute the GAN; it indicates a target upper bound for the efficiency of MD-GAN. MD-GAN lines crossing FL-GAN is indicating more incurring traffic with increasing batch sizes. A global observation is that MD-GAN is competitive for smaller batch sizes, yet in the order of hundreds of images (here of less than around $b = 550$ for MNIST and $b = 400$ for CIFAR10).

#### 2) Computation complexity:

The goal of MD-GAN is to remove the generator tasks from workers by having a single one on the server. During the training of MD-GAN, the traffic between workers and the server is reasonable (Table III). The complexity gain on workers in term of memory and computation depends on the architecture of $\mathcal{D}$; it is generally half of the total complexity because $\mathcal{G}$ and $\mathcal{D}$ are often similar. The consequence of this single generator-based algorithm is more frequent interactions between workers and the server, and the creation of a worker-to-worker traffic. The overall operation complexities are summarized and compared in Table II, for both MD-GAN and FL-GAN; the Table indicates a workload of half the one of FL-GAN on workers.

## V. Experimental Evaluation

We now analyze empirically the convergence of MD-GAN and of competing approaches.

### A. Experimental setup

Our experiments are using the Keras framework with the Tensorflow backend. We emulated workers and the server on GPU-based servers equipped of two Intel Xeon Gold 6132 processors, 260 GB of RAM and four NVIDIA Tesla M60 GPUs or four NVIDIA Tesla P100 GPUs. This setup allows for a training of GANs that is identical to a real distributed deployment, as computation order of interactions for Algorithm IV are preserved. This choice for emulation is thus oriented towards a tighter control for the environment of competing approaches, to report more precise head to head result comparisons; raw timing performances of learning tasks are in this context inaccessible and are left to future work.

a) **Datasets:** We experiment competing approaches on two classic datasets for deep learning: MNIST [33] and CIFAR10 [34]. MNIST is composed of a training dataset of 60,000 grayscale images of 28 × 28 pixels representing handwritten digits and another test dataset of 10,000 images. These two datasets are composed respectively of 6,000 and 1,000 images for each digit. CIFAR10 is composed of a training set 50,000 RGB images of 32 × 32 pixels representing the following 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. CIFAR10 has a test dataset of 10,000 images.

b) **GAN architectures:** In the experiments, we train a classical type of GAN named ACGAN [19]. We experiment with three different architectures for $\mathcal{G}$ and $\mathcal{D}$: a multi-layer based architecture (MLP), a convolutional neural network based architecture (CNN) for MNIST and a CNN-based architecture for CIFAR10. Their characteristics are:

- In the MLP-based architecture for MNIST, $\mathcal{G}$ and $\mathcal{D}$ are composed of three fully-connected layers each. $\mathcal{G}$ layers contain respectively 512, 512 and 784 neurons, and $\mathcal{D}$ layers contain 512, 512 and 11 neurons. The total number of parameters is 716,560 for $\mathcal{G}$ and 670,219 for $\mathcal{D}$.
- In the CNN-based architecture for MNIST, $\mathcal{G}$ is composed of one full-connected layer of 6,272 neurons and two transposed convolutional layers of respectively 32 and 1 kernels of size $5 \times 5$. $\mathcal{D}$ is composed of six convolutional layers of respectively 16, 32, 64, 128, 256 and 512 kernels of size $3 \times 3$, a mini-batch discriminator layer [20] and one full-connected layer of 11 neurons. The total number of parameters is 628,058 for $\mathcal{G}$ and 286,048 for $\mathcal{D}$.
- In the CNN-based architecture for CIFAR10, $\mathcal{G}$ is composed of one full-connected layer of 6,144 neurons and three transposed convolutional layers of respectively 192, 96, and 3 kernels of size $5 \times 5$. $\mathcal{D}$ is composed of six convolutional layers of respectively 16, 32, 64, 128, 256 and 512 kernels of size $3 \times 3$, a mini-batch discriminator layer and one full-connected layer of 11 neurons. The total number of parameters is 628,110 for $\mathcal{G}$ and 100,203 for $\mathcal{D}$.

c) **Metrics:** Evaluating generative models such as GANs is a difficult task. Ideally, it requires human judgment to assess the quality of the generated data. Fortunately, in the domain of GANs, interesting methods are proposed to simulate this human judgment. The main one is named the Inception Score (we denote it by IS), and has been proposed by Salimans et al. [20], and shown to be correlated to human judgment. The IS consists to apply a pre-trained Inception classifier over the generated data. The Inception Score evaluates the confidence on the generated data classification (i.e., generated data are well recognized by the Inception network), and on the diversity of the output (i.e., generated data are not all the same). To evaluate the competitors on MNIST, we use the MNIST score (we name it MS), similar to the Inception score, but using a classifier adapted to the MNIST data instead of the Inception network. Heusel et al. propose a second metric named the Fréchet Inception Distance (FID) in [35]. The FID estimates a distance between the distribution of generated data $P_G$ and real data $P_{\text{data}}$. It applies the Inception network on a sample of generated data and another sample of real data and supposes that their outputs are Gaussian distributions. The FID computes the Fréchet Distance between the Gaussian distribution obtained using generated data and the Gaussian distribution obtained using real data. As for the Inception distance, we use a classifier more adapted to compute the FID on the MNIST dataset. We use the implementation of the MS and FID available in Tensorflow³.

d) **Configurations of MD-GAN and competing approaches:** To compare MD-GAN to classical GANs, we train the same GAN architecture on a standalone server (it thus has access to the whole dataset $\mathcal{B}$). We name this baseline *standalone-GAN* and parametrize it with two batch sizes $b = 10$ and $b = 100$.

We run FL-GAN with parameters $E = 1$ and $b = 10$ or $b = 100$; this parameter setting comes from the fact that $E = 1$ and $b = 10$ is one of the best configuration regarding computation complexity on MNIST, and because $b = 50$ is the best one for performance per iteration [15] (having $b = 100$ thus allows for a fair comparison for both FL-GAN and MD-GAN). MD-GAN is run with also $E = 1$; i.e., for FL-GAN and MD-GAN, respective actions are taken after the whole dataset has been processed once.

For MD-GAN and FL-GAN, the training dataset is split equally over workers (images are sampled *i.i.d.*). We run two configurations of MD-GAN, one with $k = 1$ and another with $k = \lceil \log(N) \rceil$, in order to evaluate the impact of the data diversity sent to workers. Finally, in FL-GAN, GANs over workers perform learning iterations (such as in the standalone case) during 1 epoch, *i.e.*, until $\mathcal{D}_n$ processes all local data $\mathcal{B}_n$.

We experimented with a number of workers $N \in \{1, 10, 25, 50\}$; geo-distributed approaches such as Gaia [8] or [9] also operate at this scale (where 8 nodes [9] and 22 nodes [8] at maximum are leveraged). All experiments are performed with $I = 50,000$, *i.e.*, the generator (or the $N$ generators in FL-GAN) are updated 50,000 times during a generator learning step. We compute the FID, MS and IS scores every 1,000 iterations using a sample of 500 generated data. The FID is computed using a batch of the same size from the test dataset. In FL-GAN, the scores are computed using the generator on the central server.
