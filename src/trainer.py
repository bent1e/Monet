from trl import SFTTrainer, SFTConfig
import torch
import os, csv, torch, datetime
import gc
import numpy as np

class CustomTrainerStage1(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss
    
class CustomTrainerAVTStage1(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.weight = 1.0
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        # 日志文件路径
        log_dir = self.args.logging_dir or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.loss_log_path = os.path.join(log_dir, f"loss_history_w{self.weight}_{self.exp_name}_{timestamp}.csv")

        # 如果文件不存在，就写表头
        if self.is_main_process and not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step","epoch",
                    "loss_total", "loss_ce",
                    "loss_student_ce", "loss_teacher_ce",
                    "loss_align"
                ])
                
    def alignment_loss(self, student_reps_all_layers, teacher_reps_all_layers, student_poss, teacher_poss):
        total_loss = 0.
        for student_reps, teacher_reps in zip(student_reps_all_layers, teacher_reps_all_layers):
            layer_loss = 0.
            for batch_idx, (student_pos, teacher_pos) in enumerate(zip(student_poss, teacher_poss)):
                if len(student_pos) == 0 and len(teacher_pos) == 0:
                    continue
                student_reps = student_reps[batch_idx, student_pos, :]
                teacher_reps = teacher_reps[batch_idx, teacher_pos, :].detach() # stop gradient
                sim = torch.nn.functional.cosine_similarity(student_reps, teacher_reps).mean()
                layer_loss += 1 - sim
            total_loss += layer_loss/ len(student_poss)
        total_loss = total_loss / len(student_reps_all_layers)
        '''if torch.isnan(total_loss):
            #print("student_reps_all_layers =", student_reps_all_layers)
            #print("teacher_reps_all_layers =", teacher_reps_all_layers)
            print("student_poss =",student_poss)
            print("teacher_poss =",teacher_poss)'''
        return total_loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """

        (ce_loss, outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            
        #print(outputs.student_hidden_states, outputs.teacher_hidden_states)
        alignment_loss = self.alignment_loss(
            outputs.student_hidden_states,
            outputs.teacher_hidden_states,
            outputs.student_alignment_poss,
            outputs.teacher_alignment_poss
        )
        loss = ce_loss + self.weight *alignment_loss
        
        outputs_student_loss = outputs.student_loss.item()
        outputs_teacher_loss = outputs.teacher_loss.item()
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # --------  写本地文件  --------
        if self.is_main_process:
            with open(self.loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step,
                    self.state.epoch,
                    loss.item(),
                    ce_loss.item(),
                    outputs_student_loss,
                    outputs_teacher_loss,
                    alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss,
                ])
        # --------------------------------------------
        
        
        return (loss, outputs) if return_outputs else loss