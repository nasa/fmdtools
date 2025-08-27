#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI wizard for fmdtools CLI that guides users through model creation.

This module provides an interactive AI assistant that asks focused questions
to collect the information needed to generate a complete fmdtools LevelSpec.
"""

from __future__ import annotations
import json
import os
import sys
from time import sleep
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv, find_dotenv
from pydantic import ValidationError
from openai import OpenAI

from .schemas import LevelSpec

SYSTEM_PROMPT = """You are an expert fmdtools modeler embedded in a CLI.
Ask one concise question at a time to collect only the missing info needed
to build a high-quality LevelSpec. Avoid theory; be precise. Stop asking
when you can emit a complete, coherent spec.

Rules:
- Keep questions concrete and scoped; never ask multiple at once.
- Prefer sane defaults; only ask if a default would be wrong.
- When ready, return action=spec with a complete LevelSpec that compiles.
- Be conversational but efficient; guide users to the right level of detail.
- For functions, ask about states, modes, and faults only if they're important.
- For flows, focus on the essential variables that matter for simulation.
- For architecture, ensure logical connections between functions.
"""


def _decision_schema() -> Dict[str, Any]:
    """Return the JSON schema for AI responses."""
    # Create a simplified schema that OpenAI can handle
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["ask", "spec"]},
            "question": {"type": "string"},
            "spec": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "functions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "states": {"type": "object", "additionalProperties": False},
                                "modes": {"type": "array", "items": {"type": "string"}},
                                "faults": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"}
                                        },
                                        "required": ["name"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["name", "description", "modes", "faults"],
                            "additionalProperties": False
                        }
                    },
                    "flows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "vars": {"type": "object", "additionalProperties": False}
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False
                        }
                    },
                    "architecture": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "functions": {"type": "array", "items": {"type": "string"}},
                            "connections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from_fn": {"type": "string"},
                                        "to_fn": {"type": "string"},
                                        "flow_name": {"type": "string"}
                                    },
                                    "required": ["from_fn", "to_fn", "flow_name"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["name", "functions", "connections"],
                        "additionalProperties": False
                    },
                    "simulation": {
                        "type": "object",
                        "properties": {
                            "sample_run": {"type": "boolean"},
                            "fault_analysis": {"type": "boolean"},
                            "parameter_study": {"type": "boolean"}
                        },
                        "required": ["sample_run", "fault_analysis", "parameter_study"],
                        "additionalProperties": False
                    },
                    "is_quick_mode": {"type": "boolean"}
                },
                "required": ["name", "description", "functions", "flows", "architecture", "simulation", "is_quick_mode"],
                "additionalProperties": False
            }
        },
        "required": ["action", "question", "spec"],
        "additionalProperties": False
    }


class AIWizard:
    """AI wizard that guides users through fmdtools model creation."""
    
    def __init__(self, model: Optional[str] = None):
        """Initialize the AI wizard with API credentials."""
        # Load .env from current directory first, then project root
        load_dotenv(find_dotenv(usecwd=True))
        load_dotenv()  # second chance if project root has .env
        
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment or .env file. "
                "Please create a .env file with your OpenAI API key."
            )
        
        self.model = model or os.getenv("FMDTOOLS_AI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=key, 
            default_headers={"User-Agent": "fmdtools-cli/1.0"}
        )
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Track conversation for debugging
        self.conversation_log = []

    def _call(self) -> Dict[str, Any]:
        """Make a call to the OpenAI API with structured output and retries."""
        for i in range(4):  # 0,1,2,3 retries
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.2,
                    timeout=30,  # seconds
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "LevelWizardDecision",
                            "schema": _decision_schema(),
                            "strict": True,
                        },
                    },
                )
                content = resp.choices[0].message.content
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Model returned invalid JSON: {e}")
                    
            except Exception as e:
                if i == 3:  # Last retry
                    if "rate_limit" in str(e).lower():
                        raise RuntimeError("API rate limit exceeded. Please wait a moment and try again.")
                    elif "quota" in str(e).lower():
                        raise RuntimeError("API quota exceeded. Please check your OpenAI account.")
                    elif "timeout" in str(e).lower():
                        raise RuntimeError("API request timed out. Please try again.")
                    else:
                        raise RuntimeError(f"API call failed: {e}")
                
                # Exponential backoff
                sleep(2 ** i)
                continue

    def _truncate_history(self):
        """Truncate message history if it gets too long."""
        if len(self.messages) > 40:
            # Keep system message and last 20 messages
            self.messages = self.messages[:1] + self.messages[-20:]

    def run(self) -> LevelSpec:
        """Run the AI wizard to collect information and generate a spec."""
        print("\nAI Wizard Mode")
        print("=" * 50)
        print("I'll help you create a fmdtools model by asking focused questions.")
        print("Let's get started!\n")
        
        # Seed with kickoff instruction
        self.messages.append({"role": "user", "content": "Start the intake. Ask the first question to understand what kind of system the user wants to model."})
        
        question_count = 0
        while True:
            try:
                # Truncate history if needed
                self._truncate_history()
                
                data = self._call()
                
                if data["action"] == "ask":
                    # Validate ask action has question
                    if "question" not in data or not data["question"]:
                        print("AI returned ask action without question. Retrying...")
                        continue
                        
                    question_count += 1
                    q = data["question"].strip()
                    
                    # Show question and get answer
                    print(f"Q{question_count}: {q}")
                    answer = input("A: ").strip()
                    
                    # Log the exchange
                    self.conversation_log.append({"question": q, "answer": answer})
                    
                    # Add to conversation context
                    self.messages.append({"role": "assistant", "content": q})
                    self.messages.append({"role": "user", "content": answer})
                    
                    # Prevent infinite loops
                    if question_count > 50:
                        raise RuntimeError("Too many questions asked. Please try again with more specific requirements.")
                    
                    continue
                    
                elif data["action"] == "spec":
                    # Validate spec action has spec
                    if "spec" not in data:
                        print("AI returned spec action without spec. Retrying...")
                        continue
                        
                    try:
                        spec = LevelSpec.model_validate(data["spec"])
                        print(f"\nSpecification complete after {question_count} questions!")
                        return spec
                        
                    except ValidationError as ve:
                        # Feed validation errors back to the model to self-correct
                        err = ve.errors()
                        error_msg = f"Validation errors in the generated spec. Please fix and re-emit the complete spec JSON:\n{json.dumps(err, indent=2)}"
                        
                        self.messages.append({
                            "role": "user", 
                            "content": error_msg
                        })
                        
                        print(f"\nValidation errors detected. Let me fix this...")
                        continue
                        
                else:
                    raise RuntimeError(f"Invalid action from AI: {data.get('action', 'unknown')}")
                    
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or use interactive mode instead.")
                raise

    def get_conversation_summary(self) -> List[Dict[str, str]]:
        """Get a summary of the conversation for debugging."""
        return self.conversation_log.copy()
