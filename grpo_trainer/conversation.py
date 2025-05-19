from typing import Dict, List, Tuple
import torch
import pandas as pd
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    input_ids: torch.Tensor
    response_ids: torch.Tensor

class ConversationGenerator:
    def __init__(
        self,
        config,
        tokenizer: AutoTokenizer,
        patient_model: AutoModelForCausalLM,
        doctor_model: AutoModelForCausalLM,
        medical_data: pd.DataFrame
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.patient_model = patient_model
        self.doctor_model = doctor_model
        self.medical_data = medical_data
        self._setup_generation_config()

    def _setup_generation_config(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=self.config.branches
        )

    def generate_turn(self, prompt: str, speaker: str) -> ConversationTurn:
        """Generate a single conversation turn"""
        model = self.patient_model if speaker == "Patient" else self.doctor_model
        use_grad = speaker != "Patient"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.set_grad_enabled(use_grad):
            outputs = model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        return ConversationTurn(
            speaker=speaker,
            text=self._decode_response(outputs, inputs),
            input_ids=inputs.input_ids[0],
            response_ids=outputs[0]
        )

    def _decode_response(self, outputs, inputs) -> str:
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

    def generate_branched_conversation(self, patient_input: str) -> List[List[ConversationTurn]]:
        """Generate complete branched conversation"""
        initial_input = self.tokenizer(f"Patient: {patient_input}", return_tensors="pt")
        initial_input_ids = initial_input["input_ids"].to(self.config.device)

        branches = [[self._create_initial_turn(patient_input, initial_input_ids)]]

        for turn_idx in range(self.config.num_turns - 1):
            current_speaker = "Doctor" if turn_idx % 2 == 0 else "Patient"
            branches = self._process_turn(branches, current_speaker, turn_idx)

        return branches

    def _create_initial_turn(self, text: str, input_ids: torch.Tensor) -> ConversationTurn:
        return ConversationTurn(
            speaker="Patient",
            text=text,
            input_ids=input_ids[0],
            response_ids=input_ids[0]
        )

    def _process_turn(
        self,
        branches: List[List[ConversationTurn]],
        speaker: str,
        turn_idx: int
    ) -> List[List[ConversationTurn]]:
        new_branches = []
        for branch_idx, branch in enumerate(branches):
            if speaker == "Doctor":
                new_branches.extend(self._generate_doctor_turns(branch, branch_idx))
            else:
                new_branches.append(self._generate_patient_turn(branch))
        return new_branches

    def _build_doctor_prompt(self, branch: List[ConversationTurn]) -> str:
        history = "\n".join(f"{turn.speaker}: {turn.text}" for turn in branch)
        return f"{history}\nDoctor: "

    def _build_patient_prompt(self, branch: List[ConversationTurn], case_idx: int) -> str:
        history = "\n".join(f"{turn.speaker}: {turn.text}" for turn in branch)
        details = self.medical_data.iloc[case_idx]['details']
        return f"Additional context: {details}\n{history}\nPatient: "

    def _build_prompt(self, branch: List[ConversationTurn], case_idx: int = None) -> str:
        current_speaker = "Doctor" if branch[-1].speaker == "Patient" else "Patient"
        if current_speaker == "Doctor":
            return self._build_doctor_prompt(branch)
        else:
            return self._build_patient_prompt(branch, case_idx)

    def _generate_doctor_turns(
        self,
        branch: List[ConversationTurn],
        branch_idx: int
    ) -> List[List[ConversationTurn]]:
        prompt = self._build_prompt(branch)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        repeated_inputs = {
            "input_ids": inputs["input_ids"].repeat(self.config.branches, 1),
            "attention_mask": inputs["attention_mask"].repeat(self.config.branches, 1)
        }

        with torch.enable_grad():
            outputs = self.doctor_model.generate(
                **repeated_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True
            )

        new_branches = []
        # In _generate_doctor_turns:
        for i in range(self.config.branches):
            response_ids = outputs.sequences[i][inputs["input_ids"].shape[1]:]
            # Truncate to max_length tokens
            response_ids = response_ids[:self.config.max_length]
            turn = ConversationTurn(
                speaker="Doctor",
                text=self.tokenizer.decode(response_ids, skip_special_tokens=True),
                input_ids=inputs["input_ids"][0],
                response_ids=response_ids
            )
            new_branch = branch.copy()
            new_branch.append(turn)
            new_branches.append(new_branch)
            self._print_conversation(new_branch, branch_idx * self.config.branches + i + 1)
                
        return new_branches

    def _generate_patient_turn(self, branch: List[ConversationTurn]) -> List[ConversationTurn]:
    # Determine case index from the initial patient input
        initial_case = branch[0].text
        case_idx = self.medical_data[self.medical_data['case'] == initial_case].index[0]
        
        prompt = self._build_prompt(branch, case_idx)
        turn = self.generate_turn(prompt, "Patient")
        # Truncate response_ids to max_length
        turn.response_ids = turn.response_ids[:self.config.max_length]
        new_branch = branch.copy()
        new_branch.append(turn)
        self._print_conversation(new_branch, len(new_branch))
        return new_branch

    # def _print_conversation(self, branch: List[ConversationTurn], branch_num: int):
    #     print(f"\nBranch {branch_num}:")
    #     for turn in branch:
    #         print(f"{turn.speaker}: {turn.text}")

    def _print_conversation(self, branch: List[ConversationTurn], branch_num: int):
        print("\n" + "=" * 50)
        print(f"🌿 Branch {branch_num}")
        print("=" * 50)
        for i, turn in enumerate(branch):
            print(f"[Turn {i+1}] {turn.speaker}: {turn.text}")
