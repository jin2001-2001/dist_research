太好了，我已经把你上传的两个框架文件梳理完了，并结合你要引入的四类 stage（text / vision / audio / packing），给出一份**不写代码、只讲如何改**的详细实现方案。下面所有建议都严格基于你现有的 `PipelineStage` / `PipelineStage_with_mutiple_ranks` 的接口、通信与调度模型来扩展，以保证**训练框架通用性**与**最小侵入**。

---

# 0. 你现有框架里和本需求最相关的“锚点”

* `PipelineStage_with_mutiple_ranks` 已经支持**跨 DP 组**的 leader 广播与基于对象的**形状推断**（`_shape_inference`），并在**运行时前向**里完成“接收激活→调用子模块→缓存用于反传”的标准流程。我们会在此基础上做“多源输入”的扩展，而不破坏原行为。
* 你的发送/接收 P2P 通信已经引入了**31-bit tag**（包含方向、microbatch、slot、split），并在 `get_fwd_send_ops/get_fwd_recv_ops`、`get_bwd_send_ops/get_bwd_recv_ops` 里**计算 slot 序号并打 tag**。我们可以**复用“slot 域”来编码模态通道**（见第 3 节），无需改 dist API。  &#x20;
* 你的 `forward_one_chunk` 是每个 microbatch 的标准执行入口：聚合输入、可选 DP 广播、校验、执行前向、缓存用于反传。多模态版本只需要在聚合输入/接收处扩展，不动“怎么调子模块”。
* 反传路径已有 DDP/FSDP 分支、支持 `retain_graph_for_packed_mbs` 标志位；多模态打包阶段（packing）需要利用这个标志确保一次 pack 对应**多路上游梯度**时图不被提前释放。&#x20;

---

# 1. 新类的定位：`PipelineStage_Multimodality`

目标：兼容**4 种 stage 类型**（`modal_type in {"text","vision","audio","packing"}`），其中前三个是“单源输出、单向下游”，`packing` 是**多源汇聚**（来自 text/audio/vision 的同一个 microbatch 的三路分支），然后把打包后的三元组（hidden, attn\_4d, position\_ids/或 None + kwargs）交给后续 Transformer stages。

你已经给新类预留了 4 个重要字典：

* `mm_args_recv_info` / `mm_grad_recv_info`：按 **mb × modality** 存放接收描述
* `mm_fwd_cache` / `mm_bwd_cache`：按 **mb × modality** 缓存**已到货**的张量（前向）与**待回传**的梯度（反向）
  非常好，下面的改造就是围绕这四个表来做的。

---

# 2. 微批新语义与对齐规则

* **微批 ID 仍以“原始 mb”为准**（比如 mb0、mb1…）；
* 在 `packing` **之前**，每个 mb 拆为三个“模态分支”：`mb0-text`、`mb0-audio`、`mb0-vision`；
* `packing` stage 必须**等齐**同一 `mb` 的三个分支后再进入打包与后续 Transformer；
* 等齐逻辑由 `PipelineStage_Multimodality` 内部完成：依赖 `mm_args_recv_info[mb]` 的“期望表”与 `mm_fwd_cache[mb]` 的“到货表”进行判定。

---

# 3. 通信打 tag 策略（不改 dist API）

沿用你现有的 tag = `(dir, mb, slot, split)`：

* 我们定义**全局 slot 空间**由三段连续区间拼接：

  * `slot ∈ [0, S_text)`：text 通道
  * `slot ∈ [S_text, S_text+S_audio)`：audio 通道
  * `slot ∈ [S_text+S_audio, S_text+S_audio+S_vision)`：vision 通道
* `S_text/S_audio/S_vision` 在**形状推断阶段**就能确定（每个通道期望接收多少“槽位”/缓冲），并缓存到 `self._mm_slot_offset = {"text":0,"audio":S_text,"vision":S_text+S_audio}`。
* 这样，`packing` 的接收端仍用**单个循环**生成 ops，只是把 `mm_args_recv_info[mb]` 三个模态的 recv 描述**串起来**并加上**各自的 slot 偏移**打 tag；发送端也只需要在生成 `plans` 时保证本模态内按顺序 `slot_ctr+=1`（你现有实现就是这样）。无需修改 `dist.P2POp` / `tag` 的结构。 &#x20;

---

# 4. 形状推断（\_shape\_inference）如何“多源”化

**目的**：配置 `mm_args_recv_info` / `mm_grad_recv_info` 与各模态的 `slot` 布局、DP 广播一致性。

1. **非 packing stage（text/vision/audio）**

* 完全沿用 `PipelineStage_with_mutiple_ranks._shape_inference` 的流程，不需要改：

  * 第一 stage 由 leader 把 **meta args** 在本组内 DP 广播；
  * 非第一 stage 从上游 leader `recv_object_list`，再在本组 DP 广播；
  * 本 stage 用 zeros 跑一遍子模块拿到 **outputs\_meta**，发到下游 leader。

2. **packing stage**

* **上游不是单一 group**，而是 *text/vision/audio* 三个上游组：

  * 在形状推断时，**各上游 leader**分别通过 `send_object_list` 发送该模态的 **meta outputs**（即本模态 stage 的真实运行时将要发送的激活元信息），packing 端的 leader 用**三次 `recv_object_list`** 收齐（次序可由调度保证或用一个包含 `{"mod":..., "args":tuple}` 的对象打包一次发送）；
  * packing leader 汇总成 `mm_args_meta = {"text": tuple_t, "audio": tuple_a, "vision": tuple_v}`，派生出**每模态的 slot 计数**，并计算第 3 节的 `slot_offset`；
  * 然后在 packing 端**以“真零张量”拼装**一个“虚拟三模态实参”调用其 `submodule`（你的 `Stage1`）做一次零前向，得到 **outputs\_meta**，再发给**唯一的下游 leader**；
  * 最后在 packing 本组内做 DP 广播，保证所有 DP 成员的 `mm_args_recv_info`、`slot_offset` 与 **outputs\_meta** 一致。
* 注意：此处**不改变** base 类的接口，只在 `PipelineStage_Multimodality._shape_inference` 里覆盖实现 packing 的“多源汇聚”，并把 `self.inputs_meta` 设为**串联后的**“三模态形状元组”，以复用后续校验逻辑。

---

# 5. 运行时前向：如何接收、等齐与调用子模块

## 5.1 非 packing stage（text/vision/audio）

* 直接复用父类 `forward_one_chunk`：

  * 对于第一 stage（如 text）仍可走“根输入从调度传入→本 DP 组内广播”的分支；
  * 其余 stage 走“`_retrieve_recv_activations` 从上游接收”分支；
  * 结果放入 `self.fwd_cache[mb]` 由父类接管。&#x20;

## 5.2 packing stage

新增一个**多源版**的执行逻辑（可在 `PipelineStage_Multimodality.forward_one_chunk` 内判断 `self.modal_type=="packing"` 分支执行）：

1. **接收激活**：

   * 调度会针对同一 `mb`，分别触发三次 `RECV_F`（text/audio/vision）。
   * `PipelineStage_Multimodality` 在 `get_fwd_recv_ops` 内部把 `mm_args_recv_info[mb]["text"] + ["audio"] + ["vision"]` 这三段**按 slot\_offset 串接**，生成**一次**`ops` 列表给 runtime 执行；
   * `finish_fwd_recv` 后，把每模态的实收张量写入 `mm_fwd_cache[mb][modality]`。&#x20;

2. **等齐触发**：

   * 当 `mm_fwd_cache[mb]` 同时包含 `"text"|"audio"|"vision"`（或容忍缺省模态，例如无音频样本）时，进入 pack：
   * 组装 `composite_args/composite_kwargs`：

     * 来自 text 的三元组 `hidden, attn_4d, position_ids/None` 及 `input_ids/attention_mask_2d`；
     * 来自 audio/vision 的 `audio_embeds` / `image_embeds` 与可能的 `grid_thw`（如果视觉端产出）；
   * 调用子模块（你的 `Stage1`）即可**沿用父类的 `forward_maybe_with_nosync`**；
   * 把 `output_tuple` 与 \*\*flatten 后的输入（含三模态）\*\*缓存到 `self.fwd_cache[mb]`，供反传。

> 说明：如果你希望继续使用父类的 `_validate_fwd_input/_validate_fwd_outputs`，那就让 packing 在形状推断时把“串联后的 inputs\_meta / outputs\_meta”置到自身对象里（第 4 节已提及），这样校验仍然生效。&#x20;

---

# 6. 运行时反传：多源梯度的拆分与回送

## 6.1 下游回来的 `dY`

* packing 作为**非最后 stage**，仍会从下游 `RECV_B` 收到 `grads_output`（你父类已经处理了“若下游有多个 DP 副本，先做平均”的逻辑）。

## 6.2 执行 `backward`

* 直接复用父类的 `backward_one_chunk`，但**把 `retain_graph_for_packed_mbs=True` 传进去**（packing 聚合了三路上游输入，确保图在多路梯度回送时不被提前释放——父类已把这个旗标透传给 `stage_backward`）。

## 6.3 把 `grads_input` 拆分给三路上游

* 父类的 `get_bwd_send_ops` 是基于**单一路上游**（`grad_send_info` 从 `args_recv_info` 推导）。
* 在多模态下，packing 需要**维护一个多模态版的 grad\_send\_info**：

  * 在形状推断阶段，和 `mm_args_recv_info` 一起构造 `mm_grad_send_info[mb][modality]`；
  * 在 `get_bwd_send_ops` 的 packing 分支里，把 `self.bwd_cache[mb]` 中的 `grads_input` **按 slot 区间切割回三段**（用你在第 3 节保存的 `slot_offset` + `S_modality`）；
  * 三段各自生成 `P2POp(isend)`，**目标 `dest_rank` 要选对应模态上游组的 leader**；
  * 同样按照“1D-flat chunking + tag(方向=1, mb, slot, split)”发送。

---

# 7. DP 与 leader 语义（不变动，仅补“多上游”）

* 你已有的 DP 逻辑是：

  * **形状推断**期间：**仅 leader** 与相邻 stage leader 用 `send/recv_object_list` 交互，然后在本 DP 组内 `broadcast_object_list` 一次。
  * **运行时激活/梯度**：走 `P2POp` 的张量流。
* 对 packing 的改造只是把“上游 leader”从**1 个**扩展为**3 个**（text/audio/vision），并在 `packing` 端 leader 把三路 meta/形状与 slot 布局确定后**再做一次本组内 DP 广播**，以保证组内成员对 `mm_args_recv_info/mm_grad_recv_info/slot_offset` 一致。

---

# 8. 与调度（actions）的接口约定

为支持“同一 `mb` 的三路模态分支”，你需要把**动作表**里关于 SEND/RECV 的条目扩展出 **modality** 与 **source\_group** 概念（不一定写进 tag，只要在 stage 里能知道这次 RECV 来自谁、属于哪个模态即可）：

* 每条通信 action 增加字段：

  * `kind ∈ {"SEND_F","RECV_F","SEND_B","RECV_B"}`（已有）
  * `mb_id`（已有）
  * `modality ∈ {"text","audio","vision"}`（**新增**）
  * `src_group` / `dst_group`（已有但对 packing 要求**可多值**配置）
  * `num_splits`（已有）
* 生成阶段：

  * text/audio/vision 三个 head 的 SEND\_F 对应 packing 的三条 RECV\_F；
  * packing 的 SEND\_B 对应三路上游 head 的 RECV\_B；
* 执行阶段：

  * 在 `PipelineStage_Multimodality` 里根据 action 的 `modality` 把 recv/send 的 buffer 描述写入 `mm_args_recv_info[mb][mod]` 或从 `mm_bwd_cache[mb][mod]` 取数据；
  * `slot` 由我们在**形状推断**时对每模态建立的顺序 + 偏移决定（见第 3 节），不需要 action 指定。

---

# 9. 校验与元信息（grid\_thw、position\_ids 等）

* 基类的 `_validate_fwd_input/_validate_fwd_outputs` 面向**线性单源**，对 kwargs 无强约束。你可以**在 packing 的形状推断**阶段把“串接后的 inputs\_meta / outputs\_meta”配置给自身，这样仍可用现成校验（或在 packing 分支里跳过输入校验）。
* `grid_thw` 之类的小元信息建议：

  1. 若要走 P2P 张量通道，把它变成小的 int/long tensor（放在 vision 模态的 slots 中）；
  2. 或者在**形状推断时**通过 `object_list` 一次性交给 packing（最省事），运行时用“常数缓存”，无需每个 step 传。

---

# 10. 保持可微与内存图管理

* **pack 操作**（在你的 `Stage1` 内）要用函数式替换（`index_copy`/`scatter` + `where`），避免原地写 mask 破梯度；这个在前一轮我们已经确定。
* **retain\_graph\_for\_packed\_mbs**：packing 在 `backward_one_chunk` 调用时**打开**，以防多路上游梯度回送导致图提前释放（你父类已把 flag 下传给 `stage_backward`）。&#x20;

---

# 11. 逐函数“怎么改”（只讲步骤，不给代码）

下面给出你需要在 `PipelineStage_Multimodality` 覆盖/新增的关键函数与改动点：

1. `__init__(...)`

   * 保留你已加的 4 个多模态字典。
   * 额外：`self._mm_slot_offset = {"text":0,"audio":None,"vision":None}`，在形状推断结束时补齐。

2. `_shape_inference(args, kwargs)`（**packing 覆盖版**）

   * 若 `modal_type!="packing"`：直接 `return super()._shape_inference(args, kwargs)`。
   * 若 `modal_type=="packing"`：

     1. 依次从三个上游 leader `recv_object_list` 三份 **meta inputs**；
     2. 解析出每模态的 `recv_infos` 数量，计算 `S_text/S_audio/S_vision` 与 `slot_offset`；
     3. 拼装“串接后的 inputs\_meta”作为 `self.inputs_meta`；
     4. 用 zeros 调子模块做**零前向**→得到 `outputs_meta`；
     5. 把 `outputs_meta` 发给下游 leader，并在本 DP 组广播。

3. `get_fwd_recv_ops(...)`（**packing 覆盖版**）

   * 把 `mm_args_recv_info[fwd_chunk_id]["text"|"audio"|"vision"]` 取出**串接**，各自 slot 加偏移；
   * 生成一个**合并后的** ops 列表与 `ops_per_chunk`；
   * 复用父类的 `_last_comm_plan[(“RECV_F”,mb)]` 记录统计。

4. `finish_fwd_recv(mb)`

   * 在 packing 分支里，把合并 buffer **按 slot 区间切回三段**，分别落到 `mm_fwd_cache[mb][mod]`；非 packing 走父类逻辑即可。

5. `forward_one_chunk(...)`（**packing 仅在“聚合/触发时机”加一层**）

   * 若 `modal_type!="packing"`：`return super().forward_one_chunk(...)`。
   * 若 `modal_type=="packing"`：

     1. 本次 step 若只是“其中一个模态到货”，先 `finish_fwd_recv` → 填 `mm_fwd_cache[mb][mod]`，**但不触发前向**，直接返回一个“占位标记”给调度（你的调度 already 支持 compute/comms 分离，确保只在 3 路都 ready 时才安排 compute）；
     2. 当 3 路齐了，由 packing stage 在下一条 compute 动作里**真正调用**子模块：把三模态缓存拼成 `composite_args/kwargs`，再调用 `forward_maybe_with_nosync`，并把（输出，输入扁平）存入 `self.fwd_cache[mb]`，供反传用。

6. `get_bwd_send_ops(...)`（**packing 覆盖版**）

   * 从 `self.bwd_cache.pop(mb)` 取 `grads_input`；
   * 用 `slot_offset` + 各模态 `S_modality` 把梯度**切三段**，分别面向 text/audio/vision 的上游 leader 生成 `P2POp(isend)`，打 `(dir=1, mb, slot, split)` 的 tag；
   * 写 `_last_comm_plan[(“SEND_B”,mb)]`。

7. `get_bwd_recv_ops(...)`（非 packing 与父类一致）

8. 维持父类的 `backward_one_chunk(...)` 与 `backward_maybe_with_nosync(...)`，packing 分支**把 `retain_graph_for_packed_mbs=True` 透传**。&#x20;

---

# 12. 兼容性与退化路径

* **只有文本单模态**：packing 等价于直通，`S_audio=S_vision=0`，slot\_offset 正常工作。
* **缺省某模态**：在形状推断时把对应模态的 `S_modality=0`，运行时就不会等待该模态。
* **DP=1**：所有“leader 广播”分支自然退化（父类已覆盖）。

---

# 13. 验证清单（落地时建议逐项打勾）

1. **形状推断日志**：打印每 stage 的 `S_text/S_audio/S_vision` 和 `slot_offset`。
2. **tag 对齐**：在 SEND/RECV 两端 dump `(mb,slot,split)` 的三元组，确认完全一致。
3. **等齐触发**：对一个 mb，在三个模态分开发送时，packing 只在第三路到达后的 compute 动作里调用子模块。
4. **反传**：对同一 mb，三个上游 stage 都能收到梯度且数值非零；关闭某模态时，另两路梯度仍正常。
5. **DDP/FSDP**：开启/关闭都能跑通，最后一步触发梯度同步（父类已内置 allreduce/dummy backward 触发逻辑）。

---

如果你愿意，下一步我可以根据这份计划，把 `PipelineStage_Multimodality` 需要**覆盖/新增**的 6 个方法的**函数签名与关键伪代码块**列出来，直接对着你的 `stage_with_mutiple_ranks.py` 补齐即可。
