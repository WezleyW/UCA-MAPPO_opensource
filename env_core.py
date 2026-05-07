import numpy as np

CURRENT_ENV = None

class EnvCore():
    """
    # 定义环境中的任务类
    """
    @classmethod
    def from_config(cls, scene_cfg):
        """
        从场景配置字典中创建环境实例。

        input: scene_cfg (dict)
        output: EnvCore 实例
        """
        return cls(scene_cfg=scene_cfg)
    
    def __init__(self, scene_cfg=None):
        if not isinstance(scene_cfg, dict):
            raise ValueError(f"scene_cfg must be a mapping(dict), got: {type(scene_cfg)}")

        env_cfg = scene_cfg["env"]
        uav_cfg = scene_cfg["UAV"]

        # 1) 战场基础配置
        self.field_width = int(env_cfg["field_width"])      # 战场宽度（x轴网格数）
        self.feild_length = int(env_cfg["field_length"])    # 战场长度（y轴网格数）
        self.max_steps  = int(env_cfg["max_steps"])         # 任务最大步数

        # 2) 目标配置 + 校验（target_num 与 target_pos 一致，且目标在战场内）
        self.target_num = int(env_cfg["target_num"])
        self.target_pos = np.asarray(env_cfg["target_pos"], dtype=np.int32)
        if self.target_pos.shape[0] != self.target_num:
            raise ValueError(
                f"env.target_num ({self.target_num}) does not match len(env.target_pos) ({self.target_pos.shape[0]})."
            )
        tx = self.target_pos[:, 0]
        ty = self.target_pos[:, 1]
        if ((tx < 1) | (tx > self.field_width) | (ty < 1) | (ty > self.feild_length)).any():
            raise ValueError(
                f"env.target_pos contains out-of-bounds position(s) for grid [1..{self.field_width}] x [1..{self.feild_length}]."
            )

        # 3) 无人机配置 + 校验（num 与 initial_pos 一致，且无人机初始位置在战场内）
        self.agent_num = int(uav_cfg["num"])
        self.UAV_pos0 = np.asarray(uav_cfg["initial_pos"], dtype=np.int32)
        if self.UAV_pos0.shape[0] != self.agent_num:
            raise ValueError(
                f"UAV.num ({self.agent_num}) does not match len(UAV.initial_pos) ({self.UAV_pos0.shape[0]})."
            )
        ux = self.UAV_pos0[:, 0]
        uy = self.UAV_pos0[:, 1]
        if ((ux < 1) | (ux > self.field_width) | (uy < 1) | (uy > self.feild_length)).any():
            raise ValueError(
                f"UAV.initial_pos contains out-of-bounds position(s) for grid [1..{self.field_width}] x [1..{self.feild_length}]."
            )

        # 4) 感知参数、通信半径与终止阈值（由置信区间推导 U 阈值）
        self.sense_radius = float(uav_cfg["sense_radius"])
        self.sense_alpha_max = float(uav_cfg.get("sense_alpha_max", 0.98))
        self.sense_alpha_thr = float(uav_cfg.get("sense_alpha_thr", 0.90))
        self.sense_declay = str(uav_cfg.get("sense_declay", uav_cfg.get("sense_decay", "linear"))).strip().lower()
        self.sense_distance_weight_k = float(uav_cfg.get("sense_distance_weight_k", 2.0))
        self.connect_radius = float(uav_cfg["connect_radius"])
        self.confidential_interval = float(uav_cfg["confidential_interval"])
        self.U_thresh = float(np.exp(-abs(np.log(self.confidential_interval / (1.0 - self.confidential_interval)))))
        if self.sense_declay not in {"linear", "exp"}:
            raise ValueError(
                f"UAV.sense_declay must be 'linear' or 'exp', got: {self.sense_declay}"
            )

        # 5) 初始连接状态（边列表）转换为对称邻接矩阵
        self.matrix0 = np.zeros((self.agent_num, self.agent_num), dtype=np.int8)
        initial_connect_status = uav_cfg["initial_connect_status"]
        if initial_connect_status:
            edges = np.asarray(initial_connect_status, dtype=np.int32)
            rows = edges[:, 0] - 1
            cols = edges[:, 1] - 1
            self.matrix0[rows, cols] = 1
            self.matrix0[cols, rows] = 1

        # 6) 初始 P/Q/U/visited（固定值）
        self.P_map0 = np.full((self.agent_num, self.field_width, self.feild_length), 0.5, dtype=np.float32)
        self.Q_map0 = np.full((self.agent_num, self.field_width, self.feild_length), 0.0, dtype=np.float32)
        self.U_map0 = np.full((self.agent_num, self.field_width, self.feild_length), 1.0, dtype=np.float32)
        self.visited_cnt0 = np.full((self.agent_num, self.field_width, self.feild_length), 0.0, dtype=np.float32)

        # 7) 奖励函数权重（消融实验接口）
        self.reward_w1 = float(env_cfg.get("reward_w1", 1.0))  # 全局奖励中 R1 的权重
        self.reward_w2 = float(env_cfg.get("reward_w2", 0.1))  # 全局奖励中 R2 的权重
        self.reward_w3 = float(env_cfg.get("reward_w3", 1.5))  # 全局奖励中 R3 的权重

        # Agent 运动信息、动作空间
        self.obs_dim = self.get_obs_dim()        # 设置智能体的观测维度
        self.action_dim = 9                      # 设置智能体的动作维度
        self.action_offset = np.array([   # 设置智能体可选动作库
            [-1, 1],  # 0: 左上
            [0, 1],   # 1: 上
            [1, 1],   # 2: 右上
            [-1, 0],  # 3: 左
            [0, 0],   # 4: 原地
            [1, 0],   # 5: 右
            [-1, -1], # 6: 左下
            [0, -1],  # 7: 下
            [1, -1],  # 8: 右下
        ], dtype=np.int32)  
        
        # 定义运行期持续更新变量
        self.cur_step   = 0
        self.UAV_pos    = self.UAV_pos0.copy()
        self.P_map      = self.P_map0.copy()
        self.Q_map      = self.Q_map0.copy()
        self.U_map      = self.U_map0.copy()
        self.visited_cnt= self.visited_cnt0.copy()
        self.matrix     = self.matrix0.copy()

        # 注册当前环境实例
        global CURRENT_ENV
        CURRENT_ENV = self

    def reset(self):
        """
        # 观测信息初始化：
        输入：
        无
        输出：
        sub_agent_obs：返回一个长度为 self.agent_num 的 list ，list 中的每个元素是形状为 (self.obs_dim,) 的一维 numpy 数组
        """
        # 重置状态
        self.cur_step    = 0
        self.UAV_pos     = self.UAV_pos0.copy()
        self.P_map       = self.P_map0.copy()
        self.Q_map       = self.Q_map0.copy()
        self.U_map       = self.U_map0.copy()
        self.visited_cnt = self.visited_cnt0.copy()
        self.matrix      = self.matrix0.copy()

        # 生成初始状态
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = self.get_obs(i, self.UAV_pos, self.U_map, self.visited_cnt, self.matrix)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # 集群每一时间步的运动
        输入：
        actions ：每个agent的离散动作ID shape=[team_size]
        输出：
        next_s  ：每个 agent 的局部观测向量，长度为 team_size 的list；每个元素 shape 为 (state_dim,) 的一维 float32 向量
        r       ：每个agent的奖励（全局奖励，所有agent一致），长度为 team_size 的list
        done    ：每个 agent 是否在本步终止，长度为 team_size 的 list
        """
        self.cur_step += 1
        # 1) 应用动作，更新位置 UAV_pos
        for i, a in enumerate(actions):
            dx, dy = self.action_offset[int(a)]
            nx = self.UAV_pos[i, 0] + dx
            ny = self.UAV_pos[i, 1] + dy
            # 检查更新位置后仍不越界
            if not (1 <= nx <= self.field_width and 1 <= ny <= self.feild_length):
                raise RuntimeError(
                    f"Out-of-bounds action for agent {i}: "
                    f"pos={self.UAV_pos[i].tolist()}, move={(dx, dy)}"
                )
            self.UAV_pos[i, 0] = nx
            self.UAV_pos[i, 1] = ny

        # 2) 更新网格访问计数 visited_cnt
        N = int(self.agent_num)
        pos = self.UAV_pos.astype(np.int64)     # [N, 2]，1-based
        xi = pos[:, 0] - 1
        yi = pos[:, 1] - 1
        self.visited_cnt[np.arange(N), xi, yi] += 1.0
        visited_cnt_post = self.visited_cnt.copy()

        # 3) 动态通信拓扑,更新集群邻接矩阵 matrix
        self.matrix = self.matrix_update()

        # 4) 完成侦察任务，更新 P_map,Q_map,U_map
        # 备份更新前的 U_map
        U_map_prior = self.U_map.copy()

        # 开展侦察任务，更新 P_map,Q_map,U_map
        self.P_map, self.Q_map, self.U_map = self.joint_search()

        # 备份更新后的 U_map
        U_map_post = self.U_map.copy()

        # 5) 计算奖励 r
        R_global = self.global_reward(U_map_prior=U_map_prior,
                                      U_map_post=U_map_post,
                                      visited_cnt_post=visited_cnt_post)
        r = [float(R_global)] * self.agent_num

        # 6) 判断任务是否结束（所有网格平均 U < U_thresh，使用 U_map_post）
        done_flag = bool(np.all(U_map_post < float(self.U_thresh))) or (self.cur_step >= self.max_steps)
        done = [done_flag] * self.agent_num                 # 广播到每个 agent

        # 7) 生成下一时刻观测
        next_s = []
        for i in range(self.agent_num):
            next_s.append(self.get_obs(i, self.UAV_pos, self.U_map, self.visited_cnt, self.matrix))

        return next_s, r, done

    def get_obs(self, u, UAV_pos, U_map, visited_cnt, matrix):
        """
        生成第 u 个智能体的 Actor 输入状态向量（0-based 索引）。

        组成：
          1) one-hot 自身编号 (N)
          2) one-hot 邻接（排除自己）(N)
          3) 自身及邻接 UAV 的归一化绝对位置 (2*N) ；非自身/邻接置 0
          4) 局部 U 信息：U_map 的第 u 张图（展平）(field_width*feild_length)
          5) 联合 visited patch：把 {u ∪ 邻接} 的 visited_cnt 在平面上逐点相加，取 u 当前位置的 3×3 patch，
             做 exp(-patch) 后展平 (9)
        """
        N = self.agent_num
        x_len, y_len = self.field_width, self.feild_length

        # 1) one-hot 自身编号
        self_id = np.zeros(N, dtype=np.float32)
        self_id[int(u)] = 1.0

        # 2) one-hot 邻接（排除自己）
        nbr_id = matrix[int(u), :].astype(np.float32).copy()
        nbr_id[int(u)] = 0.0  # 明确排除自己

        # 3) UAV位置信息：仅“自身及邻接”的归一化绝对位置；其他 UAV 的位置置 0
        pos = UAV_pos.astype(np.float32)                                        # 提取UAV位置信息 [N,2]
        denom = np.array([float(x_len), float(y_len)], dtype=np.float32) # 归一化分母
        pos_norm = pos / denom                                                  # 位置归一化 [N,2] ∈ (0,1]
        mask = (nbr_id > 0).astype(np.float32)                                  # 构造掩码（邻接UAV）
        mask[int(u)] = 1.0                                                      # 构造掩码（UAV u）
        pos_masked = pos_norm * mask[:, None]                                   # 非自身/邻接 → 0
        pos_flat = pos_masked.reshape(-1)                                       # [2N]

        # 4) 局部 U 信息：U_map 的第 u 张图，展平为长度 M 的向量
        # U_map 形状为 [N, x_len, y_len]
        U_local = np.asarray(U_map, dtype=np.float32)[int(u)].reshape(-1)  # [M]

        # 5) 联合 visited cnt信息：exp(-cnt）
        # visited_cnt 形状为 [N, x_len, y_len]（每个 UAV 一张访问图）
        sel_mask = (mask > 0.0)                                                            # 选择 自身∪邻接 的通道，形状 [N] 的 bool
        joint_grid = np.asarray(visited_cnt, dtype=np.float32)[sel_mask].sum(axis=0)      # 对所选通道逐格相加，得到联合访问计数图 [x_len, y_len]
        padV = np.pad(joint_grid.astype(np.float32),                                     # 拓展矩阵，用 0 填充边缘，避免UAV移动至边缘时不足3*3
                      ((1, 1), (1, 1)),
                      mode='constant',
                      constant_values=0)
        x0 = int(UAV_pos[int(u), 0])  # 1..x_len
        y0 = int(UAV_pos[int(u), 1])  # 1..y_len
        patch = padV[(x0 - 1):(x0 + 2), (y0 - 1):(y0 + 2)]  # # 取 u 当前坐标位置的 3×3 patch（注意 UAV_pos 是 1-based）[3,3]
        patch_flat = np.exp(-patch).reshape(-1).astype(np.float32)  # [9]

        # 拼接
        s = np.concatenate([self_id, nbr_id, pos_flat, U_local, patch_flat]).astype(np.float32)
        return s

    def get_obs_dim(self):
        i=0
        obs = self.get_obs(i, self.UAV_pos0, self.U_map0, self.visited_cnt0, self.matrix0)
        len_obs = len(obs)
        return len_obs

    def joint_search(self):
        """
        距离相关正确率的贝叶斯联合搜查（坐标全部 1-based）。
        返回并写回：self.P_map, self.Q_map, self.U_map，形状均为 [N=agent_num, field_width, feild_length]
        """
        # 1) 参数设置
        alpha_max = float(self.sense_alpha_max)          # d=0 时正确率
        alpha_thr = float(self.sense_alpha_thr)          # d=r_s 时正确率（阈值）
        k = float(self.sense_distance_weight_k)          # 距离权重系数
        eps = 1e-6        # 防除 0

        # 2) 数据索引
        xL, yL = int(self.field_width), int(self.feild_length)
        N = int(self.agent_num)
        r_s = float(self.sense_radius)
        UAV_loc = self.UAV_pos.astype(np.float64, copy=True)  # [N,2]，假定为 1-based 坐标
        Q_prior = self.Q_map.astype(np.float64, copy=True)  # [N,xL,yL]
        matrix  = self.matrix.copy()  # [N,N]

        # 3) 目标网格布尔图（2D）
        target_pos = np.asarray(self.target_pos, dtype=np.int64).reshape(-1, 2)  # [T,2]
        is_target_grid = np.zeros((xL, yL), dtype=bool)
        is_target_grid[target_pos[:, 0] - 1, target_pos[:, 1] - 1] = True

        # 4) 网格坐标（1..xL, 1..yL）
        Xg, Yg = np.meshgrid(np.arange(1, xL + 1, dtype=np.float64),
                             np.arange(1, yL + 1, dtype=np.float64),
                             indexing='ij')

        # 5) 根据距离计算无人机对所有网格的侦查正确率
        # D_all / alpha_mat / senseMask: [N, xL, yL]
        dx = Xg[None, :, :] - UAV_loc[:, 0][:, None, None]
        dy = Yg[None, :, :] - UAV_loc[:, 1][:, None, None]
        D_all = np.sqrt(dx * dx + dy * dy)
        if self.sense_declay == "exp":
            # 指数衰减，并保证 alpha(0)=alpha_max, alpha(r_s)=alpha_thr
            beta = np.log(alpha_max / max(alpha_thr, eps)) / max(r_s, eps)
            alpha_mat = alpha_max * np.exp(-beta * D_all)
        else:
            # 线性衰减，并保证 alpha(0)=alpha_max, alpha(r_s)=alpha_thr
            alpha_mat = alpha_max - (alpha_max - alpha_thr) * (D_all / max(r_s, eps))
        alpha_mat[D_all > r_s] = np.nan
        senseMask = ~np.isnan(alpha_mat)

        # 6) 生成随机观测结果 Z ∈ {0：侦查结果为没有目标,1：侦查结果为有目标}（NaN=未侦察）
        isTarget_mat = np.broadcast_to(is_target_grid, (N, xL, yL))
        rnd = np.random.rand(N, xL, yL)
        Z = np.full((N, xL, yL), np.nan, dtype=np.float64)
        idx_hit = isTarget_mat & senseMask & (rnd <= alpha_mat) # 无人机认为有目标，且侦查结果正确（命中）
        idx_miss = isTarget_mat & senseMask & (rnd > alpha_mat) # 无人机认为没有目标，且侦查结果错误（漏检）
        idx_cr = ~isTarget_mat & senseMask & (rnd <= alpha_mat) # 无人机认为没有目标，且侦查结果正确
        idx_fa = ~isTarget_mat & senseMask & (rnd > alpha_mat)  # 无人机认为有目标，且侦查结果错误
        Z[idx_hit] = 1.0
        Z[idx_miss] = 0.0
        Z[idx_cr] = 0.0
        Z[idx_fa] = 1.0

        # 7) 计算信息增量 V
        V = np.zeros((N, xL, yL), dtype=np.float64)             # 所有无人机对所有网格的侦查信息增量
        hitMask, falseMask = (Z == 1.0), (Z == 0.0)             # 无人机认为“有目标/没有目标”的网格
        if hitMask.any():
            a_h = np.maximum(eps, alpha_mat[hitMask])           # 侦查结果为“有目标”的侦查正确率
            V[hitMask] = np.log(a_h / (1.0 - a_h))              # 侦查结果为“有目标”的信息增量
        if falseMask.any():
            a_f = np.maximum(eps, alpha_mat[falseMask])          # 侦查结果为“没有目标”的侦查正确率
            V[falseMask] = np.log((1.0 - a_f) / a_f)             # 侦查结果为“没有目标”的信息增量

        # 8) Q_map更新：信息更新与按距离信息融合
        Q_tmp = Q_prior + V     # 仅信息更新后的 Qmap,shape=[N,xL,yL]
        Q_new = Q_prior.copy()  # 信息融合后的 Qmap,shape=[N,xL,yL]
        for i in range(N):
            # 计算无人机i的邻接无人机
            neiMask = (matrix[i, :] != 0) | (matrix[:, i] != 0)     # 定义邻接掩码
            neiMask[i] = True                                       # 邻接掩码（含自身）
            Q_sub = Q_tmp[neiMask]                                  # [K,xL,yL]
            D_sub = D_all[neiMask]                                  # [K,xL,yL]
            W = np.ones_like(D_sub, dtype=np.float64)               # 定义信息融合时的权重系数
            inR = (D_sub <= r_s)                                    # 邻接无人机（含自身）是否侦查掩码，true表示该无人机侦查了该网格
            if inR.any():
                W[inR] = k * r_s / np.maximum(D_sub[inR], eps)      # 按距离计算权重系数
            sumW = W.sum(axis=0)                                    # [xL,yL]
            sumW[sumW == 0.0] = 1.0
            Q_new[i] = (Q_sub * W).sum(axis=0) / sumW               # 信息融合更新Qmap

        # 9) P_map与U_map更新
        Q_clip = np.clip(Q_new, -60.0, 60.0)
        P_new = 1.0 / (1.0 + np.exp(-Q_clip))
        U_new = np.exp(-np.abs(Q_new))

        return P_new.astype(np.float32), Q_new.astype(np.float32), U_new.astype(np.float32)

    def matrix_update(self):
        """
        仅做断链与孤立节点重连（不对断开的节点做概率重连）。
        使用 self.UAV_pos, self.matrix, self.connect_radius。
        返回：更新后的邻接矩阵（0/1，无向）。
        """
        # 1) 参数设置
        UAV_pos = np.asarray(self.UAV_pos, dtype=np.float64)  # [N,2]
        matrix = np.asarray(self.matrix, dtype=np.int8).copy()  # [N,N]
        N = UAV_pos.shape[0]
        r_c = float(self.connect_radius)

        # 2） 断连“超距/失效”节点
        for i in range(N - 1):
            for j in range(i + 1, N):
                if matrix[i, j] == 1:
                    if np.isnan(UAV_pos[i]).any() or np.isnan(UAV_pos[j]).any() \
                            or np.linalg.norm(UAV_pos[i] - UAV_pos[j]) > r_c:
                        matrix[i, j] = 0
                        matrix[j, i] = 0

        # 3） 对孤立节点强制重连
        alive = ~np.isnan(UAV_pos).any(axis=1)  # 判断节点是否失效
        alive_idx = np.where(alive)[0]          # 提取失效节点“行编号” 0-base
        if alive_idx.size > 0:
            deg = matrix[np.ix_(alive_idx, alive_idx)].sum(axis=1)  # 计算无人机节点的度
            isolated = alive_idx[deg == 0]                                # 提取孤立节点“行编号” 0-base
            for node in isolated:
                others = alive_idx[alive_idx != node]                     # 除 node 外的其他节点编号
                if others.size == 0:
                    continue
                dists_all = np.linalg.norm(UAV_pos[others] - UAV_pos[node], axis=1)

                # 提取候选节点
                not_linked = (matrix[node, others] == 0)                    #  除 node 外的其他未与 node 链接节点掩码 shape=[N-1]
                cand_mask = (dists_all <= r_c) & not_linked                 #  除 node 外的其他未与 node 链接节点，且在链接范围内掩码 shape=[N-1]
                if not np.any(cand_mask):
                    continue
                cands = others[cand_mask]                                   # 备选节点编号
                dists = dists_all[cand_mask]                                # 备选节点与 node 的距离

                # 权重计算：w = k_i * exp(-d / r_c)，k_i 为候选节点当前度
                ki = matrix[cands].sum(axis=1)  # 每个候选的度
                F = np.exp(-dists / r_c)
                w = ki * F

                # 概率链接
                if np.any(w > 0):
                    probs = w / w.sum()
                    pick = np.random.choice(cands, p=probs)  # 概率抽样
                else:
                    pick = cands[np.argmin(dists)]  # 权重全 0 则取最近的
                matrix[node, pick] = 1
                matrix[pick, node] = 1

        return matrix

    def global_reward(self, U_map_prior, U_map_post, visited_cnt_post):
        """
        计算全局奖励：
        R = w1*R1 + w2*R2 + w3*R3 + rTime
        R1: 全局平均不确定度下降（>0 表示进展）
        R2: 稀疏访问奖励（各 UAV 所在格的 e^(-Cnt)）
        R3: 覆盖率提升（均值不确定度 < U_thresh 的网格比例增量）
        """
        # 参数设置
        w1, w2, w3 = float(self.reward_w1), float(self.reward_w2), float(self.reward_w3)
        clip_R = 2.0
        rTime = -0.05

        # R1：全局平均不确定度下降
        # U_map_* 形状为 [N=agent_num, field_width, feild_length]
        U_bef = np.asarray(U_map_prior, dtype=np.float32)
        U_aft = np.asarray(U_map_post, dtype=np.float32)
        # 计算各网格不确定度在 UAV 维度的均值
        Uavg_bef = np.nanmean(U_bef, axis=0)  # [field_width, feild_length]
        Uavg_aft = np.nanmean(U_aft, axis=0)  # [field_width, feild_length]
        # 全局平均下降量
        R1 = float(np.nanmean(Uavg_bef - Uavg_aft))  # 标量

        # R2：稀疏访问奖励
        # 对每架 UAV，取其落点格子的联合访问次数 v，给 1/sqrt(max(v,1))，再在 UAV 维取均值
        xL, yL = int(self.field_width), int(self.feild_length)
        N = int(self.agent_num)
        nbr_mask = (self.matrix != 0).astype(bool)           # 取邻接掩码（对角为0）
        np.fill_diagonal(nbr_mask, True)                 # 取邻接掩码（对角线标记为true）
        V = np.asarray(visited_cnt_post, dtype=np.float64)   # [N, xL, yL]
        joint_cnts_all = np.einsum("aij,ba->bij", V, nbr_mask.astype(np.float64))  # [N, xL, yL]

        pos = self.UAV_pos.astype(np.int64)
        idx_x = pos[:, 0] - 1
        idx_y = pos[:, 1] - 1
        cnt_at_pos = joint_cnts_all[np.arange(N), idx_x, idx_y]  # [N]
        R2 = float(np.exp(-cnt_at_pos).mean())

        # R3：覆盖率提升
        U_thresh = float(self.U_thresh)
        cov_bef = float(np.mean(U_bef < U_thresh))
        cov_aft = float(np.mean(U_aft < U_thresh))
        R3 = cov_aft - cov_bef

        # 合成 + 轻微时间惩罚
        R = w1 * R1 + w2 * R2 + w3 * R3 + rTime

        # 裁剪（防爆）
        R = float(np.clip(R, -clip_R, clip_R))
        return R

    def get_action_masks(self):
        """
        输出：返回动作掩码（长度=agent_num 的列表；每项是 shape=[action_dim] 的 0/1 向量）。 1 表示该动作在当前步合法，0 表示非法。
        规则：基于 1-based UAV_pos 和边界 [1..field_width]×[1..feild_length]。若出现某 UAV 所有动作均非法（全 0），将抛出错误以报警。
        """
        masks = np.zeros((self.agent_num, self.action_dim), dtype=np.float32)

        for i in range(self.agent_num):
            # 当前 UAV 位置（1-based）
            x = int(self.UAV_pos[i, 0])
            y = int(self.UAV_pos[i, 1])
            for a, (dx, dy) in enumerate(self.action_offset):
                nx = x + int(dx)
                ny = y + int(dy)
                legal = (1 <= nx <= self.field_width) and (1 <= ny <= self.feild_length)
                masks[i, a] = 1.0 if legal else 0.0
        # 严格检查：任意 UAV 若掩码全 0，报警（抛异常或改成 warnings.warn）
        row_sums = masks.sum(axis=1)
        bad_idx = np.where(row_sums == 0)[0]
        if bad_idx.size > 0:
            # 也可用 warnings.warn(...) 替代
            raise RuntimeError(
                f"Illegal action mask detected for agents {bad_idx.tolist()}. "
                f"Positions(1-based)={self.UAV_pos[bad_idx].tolist()}, "
                f"grid=([1..{self.field_width}], [1..{self.feild_length}])."
            )
        return [masks[i].copy() for i in range(self.agent_num)]
