# from typing import List, Dict, Optional, Tuple
# import torch
# import torch.optim as optim
# from torch.nn import functional as F
# from torch.nn.utils.rnn import pad_sequence  # Ensure this is at the top

# import gc
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from .config import GRPOConfig
# from .reward import RewardCalculator
# from .conversation import ConversationGenerator, ConversationTurn

# class GRPOTrainer:
#     def __init__(
#         self,
#         tokenizer: AutoTokenizer,
#         config: GRPOConfig,
#         medical_data,  # <--- Add this line
#         patient_model: Optional[AutoModelForCausalLM] = None,
#         doctor_model: Optional[AutoModelForCausalLM] = None
#     ):
#         self.config = config
#         self.tokenizer = tokenizer
#         self.reward_calculator = RewardCalculator()
        
#         # Initialize models
#         self.patient_model = patient_model or self._init_model(eval_mode=True)
#         self.doctor_model = doctor_model or self._init_model(eval_mode=False)
        
#         # Initialize components
#         self.optimizer = optim.AdamW(
#             self.doctor_model.parameters(),
#             lr=self.config.learning_rate
#         )
#         self.conversation_generator = ConversationGenerator(
#             self.config,
#             self.tokenizer,
#             self.patient_model,
#             self.doctor_model,
#             medical_data
#         )

#     def _init_model(self, eval_mode: bool) -> AutoModelForCausalLM:
#         """Initialize and configure a model instance"""
#         print(f"Initializing {'patient' if eval_mode else 'doctor'} model")
        
#         if not eval_mode:
#             print(f"========================= 100% ==========================")
#         dtype = torch.float16 if self.config.fp16 else torch.float32
        
#         model = AutoModelForCausalLM.from_pretrained(
#             self.config.model_name,
#             torch_dtype=dtype,
#             device_map="auto" if self.config.device == "cuda" else None,
#             trust_remote_code=True
#         )
        
#         if eval_mode:
#             model.eval()
#         else:
#             if self.config.gradient_checkpointing:
#                 model.gradient_checkpointing_enable()
        
#         return model

#     def train_step(
#         self,
#         queries: torch.Tensor,
#         responses: torch.Tensor,
#         rewards: torch.Tensor
#     ) -> Dict[str, float]:
#         """Perform a single training step"""
#         self.doctor_model.train()
        
#         # Forward pass
#         outputs = self.doctor_model(input_ids=queries, labels=responses)
#         logprobs = F.log_softmax(outputs.logits, dim=-1)
#         response_logprobs = torch.gather(logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

#         # Reference pass with frozen model
#         with torch.no_grad():
#             ref_outputs = self.patient_model(input_ids=queries, labels=responses)
#             ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
#             old_logprobs = torch.gather(ref_logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

#         # GRPO loss calculation
#         ratios = torch.exp(response_logprobs - old_logprobs)
#         advantages = rewards - rewards.mean()
#         policy_loss = -(ratios * advantages).mean()
#         kl_loss = self.config.kl_coeff * (old_logprobs - response_logprobs).mean()
#         total_loss = policy_loss + kl_loss

#         # Optimization step
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.doctor_model.parameters(), 0.5)
#         self.optimizer.step()

#         return {
#             'total_loss': total_loss.item(),
#             'policy_loss': policy_loss.item(),
#             'kl_loss': kl_loss.item(),
#             'mean_reward': rewards.mean().item()
#         }

#     def train_on_conversation_tree(self, patient_input: str) -> Dict[str, float]:
#         """Complete training process for a conversation tree"""
#         print(f"\n=== Starting GRPO Training ===")
#         print(f"Branches per turn: {self.config.branches}")
#         print(f"Max turns: {self.config.num_turns}")
#         print(f"Initial input: {patient_input[:50]}...")
        
#         try:
#             # Generate conversation tree
#             branches = self.conversation_generator.generate_branched_conversation(patient_input)
            
#             # Prepare training data
#             training_data = self._prepare_training_data(branches)
#             if training_data is None:
#                 print("No valid training data generated")
#                 return {}

#             # Train on collected data
#             stats = self.train_step(*training_data)
#             self._print_training_stats(stats, branches)

#             return stats

#         except Exception as e:
#             print(f"\n❌ Training Error: {str(e)}")
#             raise
#         finally:
#             self._cleanup()

#     def _prepare_training_data(
#         self,
#         branches: List[List[ConversationTurn]]
#     ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         """Prepare training data from conversation branches"""
#         all_queries, all_responses, all_rewards = [], [], []
        
#         for level in range(1, self.config.num_turns, 2):
#             for i in range(0, len(branches), self.config.branches**level):
#                 sibling_group = branches[i:i + self.config.branches**level]
#                 sibling_texts = [b[level].text for b in sibling_group if len(b) > level]

#                 for branch in sibling_group:
#                     if len(branch) > level:
#                         turn = branch[level]
#                         # Construct full conversation for this branch
#                         full_conversation = "\n".join([
#                             f"{'Patient' if j % 2 == 0 else 'Doctor'}: {turn.text}"
#                             for j, turn in enumerate(branch[:level+1])
#                         ])
#                         initial_case = branch[0].text
                        
#                         reward = self.reward_calculator.calculate_reward(
#                             turn.text, 
#                             sibling_texts,
#                             full_conversation,
#                             initial_case
#                         )
#                         all_queries.append(turn.input_ids)
#                         all_responses.append(turn.response_ids)
#                         all_rewards.append(reward)

#         if not all_queries:
#             return None

#         # Convert to tensors and normalize rewards
#         # queries = torch.stack(all_queries).to(self.config.device)

#         queries = pad_sequence(
#             all_queries, 
#             batch_first=True, 
#             padding_value=self.tokenizer.pad_token_id  # Use self.tokenizer here
#         ).to(self.config.device)


#         responses = torch.stack(all_responses).to(self.config.device)
#         rewards = torch.tensor(all_rewards, device=self.config.device).float()
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

#         return queries, responses, rewards

#     def _print_training_stats(
#         self,
#         stats: Dict[str, float],
#         branches: List[List[ConversationTurn]]
#     ):
#         """Print training statistics"""
#         print("\n=== Training Results ===")
#         print(f"Total Loss: {stats['total_loss']:.4f}")
#         print(f"Policy Loss: {stats['policy_loss']:.4f}")
#         print(f"KL Loss: {stats['kl_loss']:.4f}")
#         print(f"Avg Reward: {stats['mean_reward']:.2f}")
#         print(f"Processed {len(branches)} branches")

#     def _cleanup(self):
#         """Clean up resources"""
#         torch.cuda.empty_cache()
#         gc.collect()









from typing import List, Dict, Optional, Tuple
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GRPOConfig
from .reward import RewardCalculator
from .conversation import ConversationGenerator, ConversationTurn

class GRPOTrainer:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: GRPOConfig,
        medical_data,  # Medical data needed for generating conversation
        patient_model: Optional[AutoModelForCausalLM] = None,
        doctor_model: Optional[AutoModelForCausalLM] = None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_calculator = RewardCalculator()

        # Initialize models
        self.patient_model = patient_model or self._init_model(eval_mode=True)
        self.doctor_model = doctor_model or self._init_model(eval_mode=False)

        # Initialize components
        self.optimizer = optim.AdamW(
            self.doctor_model.parameters(),
            lr=self.config.learning_rate
        )
        self.conversation_generator = ConversationGenerator(
            self.config,
            self.tokenizer,
            self.patient_model,
            self.doctor_model,
            medical_data
        )

    def _init_model(self, eval_mode: bool) -> AutoModelForCausalLM:
        """Initialize and configure a model instance"""
        print(f"Initializing {'patient' if eval_mode else 'doctor'} model")

        if not eval_mode:
            print(f"========================= 100% ==========================")
        dtype = torch.float16 if self.config.fp16 else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )

        if eval_mode:
            model.eval()
        else:
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

        return model

    def train_step(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Perform a single training step"""
        self.doctor_model.train()

        # Forward pass
        outputs = self.doctor_model(input_ids=queries, labels=responses)
        logprobs = F.log_softmax(outputs.logits, dim=-1)
        response_logprobs = torch.gather(logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

        # Reference pass with frozen model
        with torch.no_grad():
            ref_outputs = self.patient_model(input_ids=queries, labels=responses)
            ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
            old_logprobs = torch.gather(ref_logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)

        # GRPO loss calculation
        ratios = torch.exp(response_logprobs - old_logprobs)
        advantages = rewards - rewards.mean()
        policy_loss = -(ratios * advantages).mean()
        kl_loss = self.config.kl_coeff * (old_logprobs - response_logprobs).mean()
        total_loss = policy_loss + kl_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.doctor_model.parameters(), 0.5)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': rewards.mean().item()
        }

    def train_on_conversation_tree(self, patient_input: str) -> Dict[str, float]:
        """Complete training process for a conversation tree"""
        print(f"\n=== Starting GRPO Training ===")
        print(f"Branches per turn: {self.config.branches}")
        print(f"Max turns: {self.config.num_turns}")
        print(f"Initial input: {patient_input[:50]}...")

        try:
            # Generate conversation tree
            branches = self.conversation_generator.generate_branched_conversation(patient_input)

            # Prepare training data
            training_data = self._prepare_training_data(branches)
            if training_data is None:
                print("No valid training data generated")
                return {}

            # Train on collected data
            stats = self.train_step(*training_data)
            self._print_training_stats(stats, branches)

            return stats

        except Exception as e:
            print(f"\n❌ Training Error: {str(e)}")
            raise
        finally:
            self._cleanup()

    def _prepare_training_data(
        self,
        branches: List[List[ConversationTurn]]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare training data from conversation branches"""
        all_queries, all_responses, all_rewards = [], [], []

        for level in range(1, self.config.num_turns, 2):
            for i in range(0, len(branches), self.config.branches**level):
                sibling_group = branches[i:i + self.config.branches**level]
                sibling_texts = [b[level].text for b in sibling_group if len(b) > level]

                for branch in sibling_group:
                    if len(branch) > level:
                        turn = branch[level]
                        # Construct full conversation for this branch
                        full_conversation = "\n".join([  # Prepare the conversation for training
                            f"{'Patient' if j % 2 == 0 else 'Doctor'}: {turn.text}"
                            for j, turn in enumerate(branch[:level+1])
                        ])
                        initial_case = branch[0].text

                        reward = self.reward_calculator.calculate_reward(
                            turn.text, 
                            sibling_texts,
                            full_conversation,
                            initial_case
                        )
                        all_queries.append(turn.input_ids)
                        all_responses.append(turn.response_ids)
                        all_rewards.append(reward)

        if not all_queries:
            return None

        # Pad sequences to the same length (ensure padding is correctly handled)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        
        # queries = pad_sequence(
        #     all_queries, 
        #     batch_first=True, 
        #     padding_value=pad_token_id  # Ensure padding value is set
        # ).to(self.config.device)

        # responses = pad_sequence(
        #     all_responses, 
        #     batch_first=True, 
        #     padding_value=pad_token_id  # Ensure padding value is set
        # ).to(self.config.device)

        queries = self.tokenizer(
            all_queries,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        responses = self.tokenizer(
            all_responses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


        rewards = torch.tensor(all_rewards, device=self.config.device).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return queries, responses, rewards

    def _print_training_stats(
        self,
        stats: Dict[str, float],
        branches: List[List[ConversationTurn]]
    ):
        """Print training statistics"""
        print("\n=== Training Results ===")
        print(f"Total Loss: {stats['total_loss']:.4f}")
        print(f"Policy Loss: {stats['policy_loss']:.4f}")
        print(f"KL Loss: {stats['kl_loss']:.4f}")
        print(f"Avg Reward: {stats['mean_reward']:.2f}")
        print(f"Processed {len(branches)} branches")

    def _cleanup(self):
        """Clean up resources"""
        torch.cuda.empty_cache()
        gc.collect()
