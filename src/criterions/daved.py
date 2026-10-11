import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.distributed.nn.functional import all_reduce as grad_all_reduce
from scipy.optimize import linear_sum_assignment
import joblib

class DAVED(nn.Module):
    def __init__(self, args):
        super(DAVED, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight

        self.counter = 0
        self.warm_up_sigreg = 0
        gmm_model = joblib.load(args.gmm_ckpt)["gmm"]
        if gmm_model.covariance_type != "diag":
            raise ValueError("kd_sigreg dưới đây yêu cầu GMM covariance_type='diag'")

        # Không phải tham số học; Module.to(device) sẽ chuyển các buffer theo model.
        self.register_buffer(
            "gmm_weights",
            torch.as_tensor(gmm_model.weights_, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gmm_means",
            torch.as_tensor(gmm_model.means_, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gmm_variances",
            torch.as_tensor(gmm_model.covariances_, dtype=torch.float32),
            persistent=False,
        )

    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors


    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    
    def sigreg(self, x: torch.Tensor, num_slices: int = 256) -> torch.Tensor:
        device = x.device
        # =====================================================
        # 1. Random projection seed
        #
        # Chỉ rank 0 sinh seed.
        # Sau đó broadcast để tất cả GPU dùng cùng seed.
        # =====================================================
        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device,)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        # =====================================================
        # 2. Local generator
        # =====================================================
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g,  device=device, dtype=x.dtype,)

        A = A / A.norm(p=2, dim=0, keepdim=True, ).clamp_min(1e-12)

        # =====================================================
        # 3. Epps-Pulley statistic
        # =====================================================
        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype,)

        exp_f = torch.exp(-0.5 * t.square())

        # x:   [N, K]
        # A:   [K, M]
        # x@A: [N, M]
        # x_t: [N, M, T]
        x_t = (x @ A).unsqueeze(-1) * t

        # [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # =====================================================
        # 4. Aggregate across GPUs
        # =====================================================
        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM,)
            ecf = ecf / self.world_size

        # =====================================================
        # 5. Weighted L2 distance
        # =====================================================
        err = ((ecf - exp_f).abs().square().mul(exp_f))

        global_batch_size = x.size(0) * self.world_size

        sigreg_per_slice = (torch.trapezoid(err, t, dim=1,) * global_batch_size)

        return sigreg_per_slice.mean()

    def kd_sigreg(
        self,
        student_x: torch.Tensor,
        num_slices: int = 256,
        num_t: int = 17,
        t_max: float = 5.0,
        num_sw_slices: int = 64,
    ) -> torch.Tensor:
        if student_x.ndim != 2 or student_x.size(0) == 0:
            raise ValueError("student_x phải có shape [N, D] và N > 0")
        if num_slices < 1 or num_t < 2 or t_max <= 0 or num_sw_slices < 1:
            raise ValueError("num_slices, num_sw_slices >= 1; num_t >= 2; t_max > 0")

        device = student_x.device
        distributed = self.world_size > 1
        seed = random.randint(0, 2**63 - 1) if self.process_rank == 0 else 0

        if distributed:
            seed_tensor = torch.tensor(seed, device=device, dtype=torch.int64)
            dist.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        with torch.autocast(device_type=device.type, enabled=False):
            means = self.gmm_means.to(device=device, dtype=torch.float32)
            variances = self.gmm_variances.to(device=device, dtype=torch.float32)
            weights = self.gmm_weights.to(device=device, dtype=torch.float32)

            A = torch.randn(student_x.size(1), num_slices, generator=generator, device=device)
            B = torch.randn(means.size(1), num_slices, generator=generator, device=device)
            A = F.normalize(A, dim=0)
            B = F.normalize(B, dim=0)

            t = torch.linspace(-t_max, t_max, num_t, device=device)
            student_phase = (student_x.float() @ A).unsqueeze(-1) * t
            student_sum = torch.stack(
                [student_phase.cos().sum(dim=0), student_phase.sin().sum(dim=0)], dim=-1
            )

            count = torch.tensor(student_x.size(0), device=device, dtype=torch.float32)
            if distributed:
                student_sum = grad_all_reduce(student_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            student_ecf = student_sum / count  # [M, T, 2]

            with torch.no_grad():
                projected_means = means @ B
                projected_vars = variances @ B.square()
                phase = projected_means.unsqueeze(-1) * t
                decay = torch.exp(-0.5 * projected_vars.unsqueeze(-1) * t.square())
                weighted_decay = weights[:, None, None] * decay
                teacher_cf = torch.stack(
                    [(weighted_decay * phase.cos()).sum(dim=0),
                    (weighted_decay * phase.sin()).sum(dim=0)],
                    dim=-1,
                )  # [M, T, 2]

            # Đưa trọng số tích phân theo t vào từng tọa độ của vector CF.
            dt = t[1] - t[0]
            quad_weight = torch.exp(-0.5 * t.square()) * dt
            quad_weight = quad_weight.clone()
            quad_weight[0] *= 0.5
            quad_weight[-1] *= 0.5
            scale = quad_weight.sqrt()[None, :, None]

            student_feat = (student_ecf * scale).reshape(num_slices, 2 * num_t)
            teacher_feat = (teacher_cf * scale).reshape(num_slices, 2 * num_t)

            # Sliced Wasserstein giữa hai tập gồm M vector CF.
            R = torch.randn(2 * num_t, num_sw_slices, generator=generator, device=device)
            R = F.normalize(R, dim=0)
            student_sorted = (student_feat @ R).sort(dim=0).values
            teacher_sorted = (teacher_feat @ R).sort(dim=0).values

            return count * (student_sorted - teacher_sorted).square().mean()


    def mi_loss(
            self,
            student_features: torch.Tensor,  # [B, D]
            teacher_features: torch.Tensor,  # [B, D]
            temperature: float = 0.02,
        ) -> torch.Tensor:
            """Equation (13) của PGA-KD: student_i khớp teacher_i."""
            if student_features.ndim != 2 or student_features.shape != teacher_features.shape:
                raise ValueError("student_features và teacher_features phải cùng shape [B, D]")
            if temperature <= 0:
                raise ValueError("temperature phải > 0")
    
            student = F.normalize(student_features.float(), dim=-1)
            teacher = F.normalize(teacher_features.float(), dim=-1)
    
            # logits[i, j] = cosine(student_i, teacher_j) / temperature
            logits = student @ teacher.T / temperature  # [B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
    
            return F.cross_entropy(logits, labels)


    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer 
        projectors = model_wrapper.projectors

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)
        self.counter += batch_size

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None

        tea_text_qry_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_text_token'] is not None else None
        tea_text_pos_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_text_token'] is not None else None
        
        if getattr(self, 'world_size', 1) > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
            student_qry_hidden_states = [self._dist_gather_tensor(h) for h in student_qry_hidden_states]
            student_pos_hidden_states = [self._dist_gather_tensor(h) for h in student_pos_hidden_states]
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        
        ##################################
        # student_special_ids = torch.tensor(
        #     list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
        #          - set([student_tokenizer.eos_token_id])),
        #     device=student_qry_input['input_ids'].device,
        #     dtype=torch.long
        # )

        # num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        # num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)


        # ==============================================================

        kd_loss = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_loss:
            
            unnorm_student_qry_reps = pooling(student_qry_hidden_states[-1], student_qry_input['attention_mask'], mode='eos', normalize=False)
            unnorm_student_pos_reps = pooling(student_pos_hidden_states[-1], student_pos_input['attention_mask'], mode='eos', normalize=False)
            student_rkd_reps = torch.cat([unnorm_student_qry_reps, unnorm_student_pos_reps], dim=0)
            teacher_rkd_reps = torch.cat([all_teacher_qry_reps, all_teacher_pos_reps], dim=0)

            kd_loss = self.structure_loss(student_rkd_reps, teacher_rkd_reps)
            # print(f'structure_loss: {kd_loss.item()}')

        sigreg_loss = torch.zeros_like(contrastive_loss)       
        if self.args.use_sigreg_loss:
            # Dùng reps local: kd_sigreg tự all_reduce ECF khi chạy DDP.

            selected_layers = self.args.num_layers
            student_qry_reps = pooling(student_qry_hidden_states[selected_layers], student_qry_input['attention_mask'], mode='eos', normalize=False)
            student_pos_reps = pooling(student_pos_hidden_states[selected_layers], student_pos_input['attention_mask'], mode='eos', normalize=False)

            student_eos = torch.cat(
                [student_qry_reps, student_pos_reps],
                dim=0,
            )  # [2 * local_batch_size, D_student]

            sigreg_loss += self.kd_sigreg(
                student_eos,
                num_slices=256,
                num_t=self.args.num_t,
                t_max=self.args.t_max,
                # num_sw_slices=64
            )


        # overall loss
        loss = contrastive_loss + self.kd_weight * kd_loss + self.args.sigreg_weight * sigreg_loss
        
        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': kd_loss,
            'sigreg_loss': sigreg_loss,
        }