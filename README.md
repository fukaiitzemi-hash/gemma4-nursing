Gemma 4: AI-Powered Nursing Assessment & Diagnosis
This repository features an advanced nursing assessment tool powered by Gemma 4, designed to bridge the gap between clinical expertise and Generative AI.

🌟 Overview
Developed by an educational professional with 38 years of instructional experience and 26 years in seminar management, this project demonstrates how Large Language Models (LLMs) can support nursing educators and practitioners in clinical reasoning.

The core of this tool is a specialized prompt engineering strategy that instructs the AI to act as an expert nurse educator. It processes complex patient data—specifically Subjective (S) and Objective (O) findings—to generate professional, concise assessments and evidence-based nursing diagnoses.

🛠 Features
Memory-Optimized Execution: Includes a robust clear_memory() function to manage GPU resources effectively, preventing common "Out of Memory" errors in environments like Kaggle.

Professional Medical English: Constraints are set to ensure the output meets the high standards of international medical documentation.

Structured Output: Automatically generates:

Concise Clinical Assessments

Prioritized Nursing Diagnoses

Specific Nursing Interventions

🚀 Technical Requirements
Model: Gemma 4 (7b/2b-it versions)

Hardware: GPU-enabled environment (e.g., Kaggle GPU T4 x2 or P100)

Libraries: transformers, accelerate, torch

📖 How it Works
The script utilizes the apply_chat_template to maintain a consistent persona of a "Nurse Educator with 27 years of experience." It specifically addresses conditions like Congestive Heart Failure (CHF), analyzing symptoms such as orthopnea and pitting edema to provide a structured diagnostic framework.
