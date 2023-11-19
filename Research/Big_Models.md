# LLM 背景
## Critical Question
> **为什么所有GPT-3复现都失败了？使用ChatGPT你应该知道这些?**：https://www.jiqizhixin.com/articles/2023-02-20-3  
* 闭源模型：GPT3，PALM；及其指令微调版本：Instruct-GPT、FLAN-PaLM
* 开源：OPT-175B，BLOOM-176B、GLM-130B；及其指令微调版本：OPT-IML、BLOOMZ
* 开源模型失败的一些猜想：
    * 语料库多样性、质量、语料库token数量
    * bfloat16 而不是 float16
    * 训练中途重启：包括改变截断梯度范数 (clip gradient norm)、学习率、改变优化器
    * 一些超参：词向量、layer normalization、激活函数、batch size 的逐渐增长

## 模型-数据 的 规模 
* 模型参数量
    | Model | Organization  | Date  | Size (# params)|
    | :---: | :-----------: | :---: | :-----:|
    | ELMo  | AI2 | Feb 2018 | 94M |
    | GPT   | OpenAI | Jun 2018 | 110M |
    | BERT  | Google | Oct 2018 | 340M |
    | XLM  | Facebook | Jan 2019 | 655M |
    | GPT-2  | OpenAI | Mar 2019 | 1500M |
    | RoBERTa  | Facebook | Jul 2019 | 355M |
    | Megatron-LM  | NVIDIA | Sep 2018 | 8.3B |
    |T5|	Google	|Oct 2019	|11B|
    |Turing-NLG	|Microsoft	|Feb 2020|	17B|
    |GPT-3	|OpenAI	|May 2020|	175B|
    |Megatron-Turing NLG|	Microsoft, NVIDIA|	Oct 2021|	530B|
    |Gopher	|DeepMind|	Dec 2021 |	280B|

* 千亿模型训练样本 token 数量：千亿到万亿
    * 见：Training Compute-Optimal Large Language Models

        <p align="left" >
        <img src="./pictures/train_token_LLM.png" width="600">
        </p>

    * GPT-3 用了 300B token 训练数据，但总收集数据量为 500B token

        <p align="left" >
        <img src="./pictures/train_token_GPT3.png" width="600">
        </p>

## 常见模型
* Transformer 模型结构及参数：https://zhuanlan.zhihu.com/p/107891957
* CLIP
    * Chinese CLIP：https://github.com/billjie1/Chinese-CLIP 
    * mCLIP：https://github.com/FreddeFrallan/Multilingual-CLIP 
* GPT Familty
    * GPT-1/2/3，[各代之间区别见：PDF文档](./pictures/GPTS.pdf)   
    * GPT-1 
        * 确认了单向 transformer 在 unsupervised pre-training + supervised finetuning 的范式下也可以 NLU
        * 在 9 out of 12 个下游任务上超过 SOTA
    * GPT-2 
        * 结构类似 GPT-1，但去掉了微调，而引入 `task conditioning: P(output|input, task)`
        * 相比一代，用了更大的网络（1.5B vs. 117M），更大数据（**40GB vs. 5GB**），规模大约是 10 倍
        * 在 zero-shot setting 下在 7 out of 8 数据集超过了 SOTA 
    * GPT-3
        * 见上图，175B 参数，其中 Common Crawl 有 45TB 原始数据，清洗后 **570GB**（400B BPE token），**所以千亿大模型大约 1-2 TB 高质量干净数据差不多够训练了**
    * GPT-3.5 / InstructGPT / ChatGPT
    
        > [1. **拆解追溯 GPT-3.5 各项能力的起源**](https://yaofu.notion.site/GPT-3-5-360081d91ec245f29029d37b54573756)  
        > [2. **Model index for researchers**](https://platform.openai.com/docs/model-index-for-researchers)   
    
        code-davinci-002，text-davinci-002/003，ChatGPT 都叫 GPT-3.5，都是 code-davinci-002  的微调版本 

        <p align="center" >
        <img src="./pictures/gpt_family.png" width="500">
        </p>

        * InstructGPT 三阶段：supervised fine-tuning on pre-trained GPT-3 --> Reward Model --> RL PPO；三阶段所用的标注的额外数据如下，总量并不大  

            <p align="left" >
            <img src="./pictures/train_token_chatgpt.png" width="600">
            </p>
    
        * 结论：
            * code-davinci-002 的基础模型可能不是 initial GPT-3 davinci 模型，而是可能经过如图所示技术路线改造过的 
            * code-davinci-002 推理能力很强（很强的基础模型），但与人的 alignment 不够  
            text-davinci-002 alignment 能力增强了，但在很多任务上跑分变低（上下文 ICL 能力变弱）  
            text-davinci-003 加上了 RLHF，普遍的生成通常比 text-davinci-002 长，然后上下文能力有所恢复     
            ChatGPT 进一步接近和人对话
            * 1.3B 的经过 RLHF 的 InstructGPT（模型来源于 GPT-3 XL，见 GPT-3 论文 Table E1），在 labeler 评测中，就可优于原始 175B GPT-3
            * **模型规模超过阈值后的涌现能力（大约100B）**：突破 scaling law
* 其他系列：PaLM，GLM，LLama, BLOOM，OPT
    * 其中 GLM 的训练方法特别一点：用的 Autoregressive Blank Infilling，目标是在 NLU NLG 都表现得好

## 大模型架构总结
趋势是：相对位置编码替代绝对位置编码，Pre-SwiGLU替代最早的GeLU，Causal-Decoder统治，AdamW + BF16混合精度，multi-query/group-query attention 

| 模型 | 模型架构 | 注意力机制 | 激活函数 | 位置编码 | Normalization | 其他 |
|-|-|-|-|-|-|-|  
| GPT-3 | Causal-decoder | Multi-Head Attention | GeLU | 可学习绝对位置编码 | Pre-LayerNorm | - |
| GPT-4 | Causal-decoder MoE | - | - | - | - | - |
| PaLM | Causal-decoder | Multi-Query Attention | SwiGLU | 相对位置编码 RoPE | Pre-LN | Attention 层和 FFN 层并行；无 Bias 层；Adafactor 优化器 |  
| Gopher 280B/Chinchilla 70B | Causal-decoder | MHA | GeLU | RoPE | Pre-RMSNorm | Gopher 用 Adam，Chinchilla 用 AdamW 优化器 |
| Bloom | Causal-decoder | MHA | GeLU | ALiBi | Pre-LN | BF16混合精度 |
| Falcon | Causal-decoder | MQA | GeLU | ALiBi | Pre-LN | - |
| GLM-130B | **Prefix-Decoder** | MHA | GeGLU | RoPE | **Post-LN with Deep-Norm** | AdamW 优化器,FP16混合精度 |
| ChatGLM2-6B | Causal-decoder | MQA | SwiGLU | RoPE | Pre-RMSNorm | - |
| Baichuan-2 | Causal-decoder | MHA | SwiGLU | 7B-RoPE, 13B-ALiBi | Pre-RMSNorm | AdamW 优化器,BF16混合精度 |  
| LaMMa1 | Causal-decoder | MHA | SwiGLU | RoPE | Pre-RMSNorm | AdamW 优化器,BF16混合精度 |
| LaMMa2 | Causal-decoder | Group-Query Attention | SwiGLU | RoPE | Pre-RMSNorm | AdamW 优化器,BF16混合精度 |


<br>
<br>



# LLM Tuning 方法

## Fine-tuning
* 针对 BERT：How to Fine-Tune BERT for Text Classification?  
    * 可包含 3 stages：extensive pre-training, in-domain pre-training, in-domain finetuning 
* 针对 GPT，可以参考 OpenAI API：https://beta.openai.com/docs/guides/fine-tuning 

<br>

## Prompt tuning：白盒和黑盒
### Whit Box: 按时间顺序，有以下文章 

* `Prefix Tuning：Optimizing continuous prompts for generation_ACL21` ***(关注的是用 GPT/BART 做 NLG 任务)***
    * 在每一层 transformer layer 前都加一个前缀

* `P-Tuning: GPT Understands, Too` ***(NLU 任务)***
    * 提出 GPT 这种 decoder only 也可以做 NLU 任务，但由于人工模板构造不好构造，之前工作没做到
    * 提出了 `P-Tuning`，用了 pseudo prompts 和 prompt encoder；encoder 可以生成 learnable continuous prompts，但怎样 interleaved 在 input 中，还是用了一些 human design  


* `The Power of Scale for Parameter-Efficient Prompt Tuning_EMNLP21`：第一篇文章正式 ***term prompt tuning (NLU 任务)***
    * 是 Prefix Tuning 的一个简化版本：只对 input embedding 加前缀
    * 相比 P-Tuning：prompt 不经过人工设计插在 input 中了，直接 prepend 即可；另外 P-Tuning 和 model tuning 一起用以达到效果

* `P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks_ACL22` ***(NLU 任务)***
    * 对 330M 大小的模型也有用（P-tuning v1 在 10B 大小模型和一些 tasks 上才能和 finetuning 可比）
    * 把 P-Tuning 的思想拓展到了每一层，把 tunable token 从 interleaving 变成了 prefix

        <p align="left" >
        <img src="./pictures/p-tuning.png" width="800">
        </p>

* `SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer_ACL22` ***(NLU 任务)***
    * 在 prompt tuning 开山之作的基础上，引入了从多个 source tasks 向 target task 进行 transfer 的设计
    * 预先训练很多个 source tasks 的 prompt，然后在 validation set 上选一个最好的，来初始化 target task 的 prompt 训练。然后根绝 prompt tokens 比对看哪个 source task 和 target task 最接近。最后再用选中的这个 source prompt 做为初始化训练，得到 target prompt
    
        <p align="left" >
        <img src="./pictures/spot.png" width="800">
        </p>

<br>

### Black Box
> 会用到一些无梯度优化，很多都来源于对一些自然现象的总结，例如遗传算法，见：  
[无梯度优化算法（上）](https://www.bilibili.com/video/BV1d5411475Y/?from=seopage&vd_source=93c3a9b0afc9334d69915ec59d8c3a87)  [无梯度优化算法（下）](https://www.bilibili.com/video/BV1uC4y1s7AQ/?vd_source=93c3a9b0afc9334d69915ec59d8c3a87)    

* Discrete
    * Black-Box Prompt Learning for Pre-trained Language Models
        * 不需要 LLM 反传梯度，用 policy gradient 算法直接优化 prompt；prompt 作为 input sentence 前缀
    * RLPROMPT: Optimizing Discrete Text Prompts with Reinforcement Learning
* Continuous
    * BBT：Black-Box Tuning for Language-Model-as-a-Service
    * BBTv2：Towards a Gradient-Free Future with Large Language Models 

<br>

## In-Context Learning（ICL）：提升 LLM 的 few-shot 能力
> 算是 prompt tuning 的一种：[How does in-context learning work?](http://ai.stanford.edu/blog/understanding-incontext/)  
Paper: Rethinking the Role of Demonstrations:
What Makes In-Context Learning Work?

* 方法
    * 将几对 sentence 和 label 作为 prompt（或称作 context），拼接在 input sentence 前面输入，不改变模型参数（算是 prompt engineering 的一种，也属于 black box）
    * 可以类比 few-shot learning 中的 transductive learning
* 一些 observation：
    * prompt 中的 Input-output pairing 有帮助，但不如正确的 input 重要。即便 output 变为 noise，也能相比原始模型大幅提高。这主要是因为模型与训练的时候，就见过非常多正确的 input-output pairing 了，所以 prompt 主要的目的是在 大模型 带来的大空间中，划定接下来的任务所处的语义空间。
    * output space（classes or answer choices）比较重要，例如是分类 “positive/neutral/negative”，还是分类 “tech/sports/finance”
    * 优点
        * 一个模型解决无数问题：GPT3 175B 做 in-context learning 性能约等于T5-large 770M 全数据 finetune 
    * 缺点
        * 模型对不同的 context 较为敏感，例如几个例子的顺序
        * 由于 context size 的限制（例如 2048 个字符），主要用于 NLU 分类任务，NLG 任务应用较少
        * few-shot 下的性能饱和问题，即随着 training examples 的数量的增加 (一般是 16 或者 32 左右)，in-context learning 的性能不再提升

<br>

## CoT：提升 LLM 的推理能力
> CoT Paper List：https://github.com/Timothyxxx/Chain-of-ThoughtsPapers   
> 解读：https://zhuanlan.zhihu.com/p/589087074   

* Chain-of-Thought Prompt（CoT）  
    * 上面的 ICL 在一些需要逻辑推理的任务上表现很差。所以考虑对任务进行拆分，在 prompt 中就给出一些 QA 的例子，并且这些例子中就包含一些推理的步骤（few-shot CoT）  

    * 但实验结果上，CoT 对 10B 参数量以下的模型不太有用

        <p align="left" >
        <img src="./pictures/zeroshot-cot0.png" width="700">
        </p>

* Automatic Prompting
    * `Large Language Models Are Human-Level Prompt Engineers`：发现了比 `让我们一步一步思考` 更好的提示词，用 log probability 作为衡量，越大的 prompt 认为越好

        <p align="left" >
        <img src="./pictures/APE.png" width="600">
        </p>


* 上面解决了 few shot 问题，但对于 zero shot 怎么办呢？ 可以参考 Paper：[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf)  

    原理是用 pipeline 诱导出一些思考过程，“Let's think step by step” 会让 LLM 尽可能生成一些思考过程，然后再将生成的 rationale 和 question 拼在一起，重新配合一个 answer 指向的 prompt 如 “The answer is ” 来激励模型生成答案
    
    <p align="left" >
    <img src="./pictures/zeroshot-cot.png" width="700">
    </p>

<br>

* 怎么样解决更困难的问题呢？ 
    > [Paper: Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/pdf/2205.10625.pdf)  

    * 把一个困难的问题，分两个阶段解决  
        * 第一个阶段进行 problem reduction  
            * 用一些模板，例如 `To solve {problem}, we need to {XXX}` 等
            * 该阶段可以用上文的 CoT 
        * 第二个阶段做 problem solving  
            * 把 CoT 的输出作为输入    

                <p align="left" >
                <img src="./pictures/least-to-most.png" width="700">
                </p>

* 引入 self-consistency
    * `Self-Consistency Improves Chain of Thought Reasoning in Language Models`: 思想是通过小样本链式思考生成多个不同的推理路径，并利用生成结果选择最一致的答案 


<br>

## Instruction Tuning / RLHF
* 通过自然语言的形式，把预训练模型在多个已知任务上进行微调，然后再在某个 **新任务** 上进行 zero-shot 推理。所以主要是解决 **cross-task** 问题，其中微调是要模型学会理解指令，指令很大程度上有共通之处，这也是能 zero-shot 的缘由
    * FLAN: Finetuned Language Models Are Zero-Shot Learners 
    * NatInst: Cross-Task Generalization via Natural Language Crowdsourcing Instructions
    * MiltiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning  

* 引入 reward model，RL 进行模型训练 
    * WebGPT
    * InstructPT：有一个测试关于 truthfulQA

<br>


