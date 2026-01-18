import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # 总共的TP数量
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化 PyTorch 的分布式通信后端（NCCL）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 设置当前设备
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # 初始化模型：TP每张卡运行一个模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # 创建共享内存块
            # 通过共享内存把“要做什么任务（函数名）”和“参数是什么”发送给所有从进程
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 轮询共享内存，获取相应的任务
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 读取共享内存中的任务
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        # 从共享内存读取数据并反序列化
        n = int.from_bytes(self.shm.buf[0:4], "little") # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # 仅主卡调用
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 将函数名和参数用 pickle 序列化
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little") # 存储数据长度
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 给engine的调用入口
    # 所有进程在同一时刻执行相同的代码逻辑，但处理不同的数据分片（由 NCCL 处理数据同步）
    def call(self, method_name, *args):
        # 如果是 Rank 0，它先通过 write_shm 把任务广播给所有从进程
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 自己也执行同样的任务
        method = getattr(self, method_name, None)
        return method(*args)

    # 进行模型预热
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 显存管理
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 获取当前 GPU 的总显存和已用显存
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 进行TP后均分
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # - 2：KV两个矩阵
        # - hf_config.num_hidden_layers：模型的层数，每一层都有自己的KVC
        # - self.block_size：每个block包含的token数目
        # - num_kv_heads：单张卡上的KV head数目
        # - head_dim：每个head的维度大小
        # - hf_config.torch_dtype.itemsize：每个数据元素占用的字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 可分配的KVC block数量
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 申请一片空tensor用作KVC
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 分配KVC到具体的模型结构
        layer_id = 0
        # hf_config.num_hidden_layers
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 2
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # 构建 PagedAttention 所需的页表
    def prepare_block_tables(self, seqs: list[Sequence]):
        # 构建每个请求到使用block的列表，并进行张量化
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # FlashAttention 元数据 (cu_seqlens)
        # FlashAttention 需要将所有 Sequence 拼成一个长的一维数组
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # 传入需要计算的部分
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 记录每个 Sequence 在拼接数组中的起始和结束位置
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 计算当前 Batch 中每一个 Token 应该被写入到 KV Cache 内存池的哪个物理位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 调用 set_context 将这些元数据存入线程局部变量，供后续的模型层使用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 只取每个 Sequence 的最后一个 Token（最新生成的那个）
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 将相应的sample数据转化为tensor形式：主要是温度
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # - is_prefill: Prefill 阶段的输入长度（Prompt 长度）变化非常大，无法使用固定形状的 CUDA Graph。
        # - self.enforce_eager: 用户强制配置禁用 CUDA Graph。
        # - input_ids.size(0) > 512: Batch Size 太大，超过了预录制的 Graph 范围，或者大 Batch 下 CPU Launch Overhead 占比很小，没必要用 Graph。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接调用,PyTorch 会逐个算子（Operator）下发 Kernel 到 GPU
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 获取当前真实的 Batch Size
            bs = input_ids.size(0)
            context = get_context()
            # 从预录制的 Graph 尺寸列表（如 1, 2, 4, 8...）中，找到最小的能容纳当前 Batch 的尺寸
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # CUDA Graph 录制时使用的是固定的内存地址（self.graph_vars）
            graph_vars = self.graph_vars
            # 拷贝相应的数据
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 拷贝完数据后运行：一条指令告诉 GPU：“把之前录制好的那一连串 Kernel，用现在内存里的数据，再跑一遍”。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 将高层的 Sequence 对象转换为模型能理解的 Tensor 输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 将 sample 参数转化为tensor形式
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 得到logits
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 清楚临时信息
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # CUDA Graph 的一个核心限制是：输入和输出 Tensor 的内存地址必须是固定的
        # 分配了一组足够大的 Tensor，作为所有 Graph 共用的静态显存池
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义 Batch Size 列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 开始录制
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            # : 所有的 Graph 共享同一个显存池（Mempool），这能极大节省显存。第一个 Graph 创建池子，后续的复用它
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
