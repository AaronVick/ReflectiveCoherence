import os
import json
import openai
from typing import Dict, List, Optional, Union, Any
import requests

class LLMClient:
    """
    A unified client for interacting with large language models (LLMs) 
    like OpenAI's GPT and Anthropic's Claude.
    
    This client provides methods for generating insights, explanations,
    and analyses based on coherence experiment data.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        default_provider: str = "openai"
    ):
        """
        Initialize the LLM client with API keys.
        
        Args:
            openai_api_key: OpenAI API key
            claude_api_key: Claude API key
            default_provider: Default LLM provider to use ("openai" or "claude")
        """
        # Set OpenAI API key
        self.openai_available = False
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_available = True
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.openai_available = True
            
        # Set Claude API key
        self.claude_available = False
        self.claude_api_key = None
        if claude_api_key:
            self.claude_api_key = claude_api_key
            self.claude_available = True
        elif "CLAUDE_API_KEY" in os.environ:
            self.claude_api_key = os.environ["CLAUDE_API_KEY"]
            self.claude_available = True
            
        # Set default provider
        if default_provider not in ["openai", "claude"]:
            raise ValueError("default_provider must be 'openai' or 'claude'")
        
        # Validate that at least one provider is available
        if default_provider == "openai" and not self.openai_available:
            if self.claude_available:
                print("OpenAI API key not found, using Claude as default provider.")
                default_provider = "claude"
            else:
                raise ValueError("No API keys provided for any LLM provider.")
        elif default_provider == "claude" and not self.claude_available:
            if self.openai_available:
                print("Claude API key not found, using OpenAI as default provider.")
                default_provider = "openai"
            else:
                raise ValueError("No API keys provided for any LLM provider.")
                
        self.default_provider = default_provider
        
    def query_openai(
        self, 
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Query OpenAI's API with a prompt.
        
        Args:
            prompt: The prompt to send to the API
            model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text response
        """
        if not self.openai_available:
            raise ValueError("OpenAI API key not provided.")
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a scientifically-minded AI assistant with expertise in mathematics, complex systems, and cognitive science."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def query_claude(
        self,
        prompt: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Query Anthropic's Claude API with a prompt.
        
        Args:
            prompt: The prompt to send to the API
            model: The model to use (e.g., "claude-3-opus-20240229")
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text response
        """
        if not self.claude_available:
            raise ValueError("Claude API key not provided.")
        
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": "You are a scientifically-minded AI assistant with expertise in mathematics, complex systems, and cognitive science.",
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            print(f"Error querying Claude: {e}")
            return f"Error: {str(e)}"
    
    def generate_insight(
        self,
        experiment_data: Dict[str, Any],
        provider: Optional[str] = None,
        specificity: str = "general"
    ) -> Dict[str, str]:
        """
        Generate insights about experiment results.
        
        Args:
            experiment_data: Dictionary containing experiment data
            provider: LLM provider to use (overrides default)
            specificity: Level of detail ("general", "detailed", "technical")
            
        Returns:
            Dictionary with various insights
        """
        provider = provider or self.default_provider
        
        # Format experiment data for the prompt
        exp_id = experiment_data.get('experiment_id', 'unknown')
        params = experiment_data.get('parameters', {})
        results = experiment_data.get('results', {})
        
        # Build the prompt based on specificity
        if specificity == "general":
            prompt = f"""
            Analyze this coherence experiment and provide a brief, simple explanation 
            of what happened in plain language for a general audience:
            
            Experiment ID: {exp_id}
            Parameters:
            - Coherence growth rate (α): {params.get('alpha', 'unknown')}
            - Maximum coherence (K): {params.get('K', 'unknown')}
            - Entropy influence (β): {params.get('beta', 'unknown')}
            
            Results:
            - Initial coherence: {results.get('coherence', {}).get('start', 'unknown')}
            - Final coherence: {results.get('coherence', {}).get('end', 'unknown')}
            - Coherence change: {results.get('coherence', {}).get('change', 'unknown')}
            - Mean entropy: {results.get('entropy', {}).get('mean', 'unknown')}
            - Threshold: {results.get('threshold', 'unknown')}
            - Phase state: {experiment_data.get('interpretation', {}).get('phase_state', 'unknown')}
            
            Please explain:
            1. What happened to coherence over time and why
            2. How entropy affected the system
            3. The significance of crossing or not crossing the threshold
            4. A simple real-world metaphor for what we observed
            
            Your explanation should be very accessible to non-experts.
            """
        elif specificity == "detailed":
            prompt = f"""
            Provide a detailed analysis of this coherence experiment for someone with 
            some background in mathematics or science:
            
            Experiment ID: {exp_id}
            Parameters:
            - Coherence growth rate (α): {params.get('alpha', 'unknown')}
            - Maximum coherence (K): {params.get('K', 'unknown')}
            - Entropy influence (β): {params.get('beta', 'unknown')}
            - Initial coherence: {params.get('initial_coherence', 'unknown')}
            
            Results:
            - Initial coherence: {results.get('coherence', {}).get('start', 'unknown')}
            - Final coherence: {results.get('coherence', {}).get('end', 'unknown')}
            - Coherence change: {results.get('coherence', {}).get('change', 'unknown')} 
              ({results.get('coherence', {}).get('change_percent', 'unknown')}%)
            - Mean entropy: {results.get('entropy', {}).get('mean', 'unknown')}
            - Entropy variance: {results.get('entropy', {}).get('variance', 'unknown')}
            - Threshold: {results.get('threshold', 'unknown')}
            - Phase state: {experiment_data.get('interpretation', {}).get('phase_state', 'unknown')}
            
            Please provide:
            1. A clear explanation of what happened in the experiment
            2. An analysis of the relationship between entropy and coherence
            3. How the parameters influenced the outcome
            4. What we can conclude about reflective coherence from this experiment
            5. What follow-up experiments would be valuable
            """
        else:  # "technical"
            prompt = f"""
            Provide a technical analysis of this coherence experiment for an expert in 
            complex systems, dynamical systems, or cognitive science:
            
            Experiment ID: {exp_id}
            Parameters:
            - Coherence growth rate (α): {params.get('alpha', 'unknown')}
            - Maximum coherence (K): {params.get('K', 'unknown')}
            - Entropy influence (β): {params.get('beta', 'unknown')}
            - Initial coherence: {params.get('initial_coherence', 'unknown')}
            
            Results:
            - Initial coherence: {results.get('coherence', {}).get('start', 'unknown')}
            - Final coherence: {results.get('coherence', {}).get('end', 'unknown')}
            - Coherence change: {results.get('coherence', {}).get('change', 'unknown')} 
              ({results.get('coherence', {}).get('change_percent', 'unknown')}%)
            - Mean entropy: {results.get('entropy', {}).get('mean', 'unknown')}
            - Entropy variance: {results.get('entropy', {}).get('variance', 'unknown')}
            - Threshold: {results.get('threshold', 'unknown')}
            - Phase state: {experiment_data.get('interpretation', {}).get('phase_state', 'unknown')}
            
            Please provide:
            1. A detailed mathematical analysis of the coherence dynamics
            2. An explanation of the phase transition behavior (or lack thereof)
            3. How the parameters affected the system's trajectory in phase space
            4. Theoretical implications for the ΨC Principle
            5. Recommendations for parameter modifications to explore different regimes
            6. Potential connections to other dynamical systems and phase transition models
            """
        
        # Query the selected provider
        if provider == "openai":
            detailed_analysis = self.query_openai(prompt)
        else:  # claude
            detailed_analysis = self.query_claude(prompt)
            
        # Generate a very simple headline regardless of specificity
        headline_prompt = f"""
        Based on the information below, provide a single, clear headline (10 words or fewer) 
        that captures the key finding of this coherence experiment:
        
        Experiment parameters: α={params.get('alpha', 'unknown')}, β={params.get('beta', 'unknown')}, K={params.get('K', 'unknown')}
        Results: Initial coherence={results.get('coherence', {}).get('start', 'unknown')}, 
                Final coherence={results.get('coherence', {}).get('end', 'unknown')},
                Mean entropy={results.get('entropy', {}).get('mean', 'unknown')},
                Threshold={results.get('threshold', 'unknown')},
                State={experiment_data.get('interpretation', {}).get('phase_state', 'unknown')}
        """
        
        if provider == "openai":
            headline = self.query_openai(headline_prompt, max_tokens=50)
        else:  # claude
            headline = self.query_claude(headline_prompt, max_tokens=50)
        
        return {
            "headline": headline,
            "analysis": detailed_analysis
        }
    
    def explain_concept(
        self,
        concept: str,
        audience_level: str = "beginner",
        provider: Optional[str] = None
    ) -> str:
        """
        Generate an explanation of a ΨC-related concept.
        
        Args:
            concept: The concept to explain
            audience_level: Target audience expertise level
            provider: LLM provider to use (overrides default)
            
        Returns:
            Generated explanation text
        """
        provider = provider or self.default_provider
        
        # Build prompt based on audience level
        if audience_level == "beginner":
            prompt = f"""
            Explain the concept of '{concept}' in the context of the Reflective Coherence (ΨC) 
            framework for a complete beginner who has no background in mathematics, 
            cognitive science, or complex systems.

            Your explanation should:
            1. Use everyday language and concrete examples
            2. Compare the concept to familiar real-world situations
            3. Avoid mathematical formulas and technical jargon
            4. Be concise (maximum 250 words)
            5. Include a simple metaphor to aid understanding
            """
        elif audience_level == "intermediate":
            prompt = f"""
            Explain the concept of '{concept}' in the context of the Reflective Coherence (ΨC) 
            framework for someone with a basic understanding of science and mathematics 
            (undergraduate level).

            Your explanation should:
            1. Use accessible language but include key technical terms
            2. Include a simplified mathematical formulation where appropriate
            3. Connect the concept to related scientific ideas they might be familiar with
            4. Provide a concrete example of the concept in action
            5. Be somewhat comprehensive but still accessible (300-500 words)
            """
        else:  # "advanced"
            prompt = f"""
            Provide a detailed explanation of '{concept}' in the context of the 
            Reflective Coherence (ΨC) framework for an advanced audience with 
            background in mathematics, complex systems, or cognitive science.

            Your explanation should:
            1. Present the full mathematical formulation with all relevant parameters
            2. Discuss theoretical implications and connections to other frameworks
            3. Address potential limitations or edge cases
            4. Reference related concepts from complex systems, dynamical systems theory, or information theory
            5. Be precise and technically accurate
            """
        
        # Query the selected provider
        if provider == "openai":
            explanation = self.query_openai(prompt)
        else:  # claude
            explanation = self.query_claude(prompt)
            
        return explanation
    
    def suggest_experiments(
        self,
        experiment_results: List[Dict[str, Any]],
        goal: str,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest follow-up experiments based on previous results.
        
        Args:
            experiment_results: List of dictionaries with previous experiment results
            goal: The research question or goal to pursue
            provider: LLM provider to use (overrides default)
            
        Returns:
            Dictionary with suggested experiments
        """
        provider = provider or self.default_provider
        
        # Format previous experiment data
        formatted_experiments = []
        for i, exp in enumerate(experiment_results):
            exp_summary = f"""
            Experiment {i+1}: {exp.get('experiment_id', f'Experiment {i+1}')}
            Parameters: α={exp.get('parameters', {}).get('alpha', 'unknown')}, 
                        β={exp.get('parameters', {}).get('beta', 'unknown')}, 
                        K={exp.get('parameters', {}).get('K', 'unknown')}
            Results: Initial coherence={exp.get('results', {}).get('coherence', {}).get('start', 'unknown')}, 
                    Final coherence={exp.get('results', {}).get('coherence', {}).get('end', 'unknown')},
                    Mean entropy={exp.get('results', {}).get('entropy', {}).get('mean', 'unknown')},
                    Threshold={exp.get('results', {}).get('threshold', 'unknown')},
                    Phase state={exp.get('interpretation', {}).get('phase_state', 'unknown')}
            Key finding: {exp.get('interpretation', {}).get('key_finding', 'unknown')}
            """
            formatted_experiments.append(exp_summary)
        
        previous_experiments = "\n".join(formatted_experiments)
        
        # Build prompt
        prompt = f"""
        Based on the following previous experiments investigating Reflective Coherence (ΨC), 
        suggest 3 new experiments to address this research goal: "{goal}"

        Previous experiment results:
        {previous_experiments}

        For each suggested experiment, please provide:
        1. A clear title
        2. The exact parameter values to use (α, β, K, initial coherence)
        3. A brief justification for why this experiment would be valuable
        4. What specific hypothesis it would test
        5. What we should expect to see if the hypothesis is true

        Format your response as JSON with the structure:
        {{
            "experiments": [
                {{
                    "title": "Experiment title",
                    "parameters": {{
                        "alpha": value,
                        "beta": value,
                        "K": value,
                        "initial_coherence": value
                    }},
                    "justification": "Reason for this experiment",
                    "hypothesis": "What this tests",
                    "expected_outcome": "What we expect to see"
                }},
                // Additional experiments...
            ]
        }}
        """
        
        # Query the selected provider
        if provider == "openai":
            response = self.query_openai(prompt)
        else:  # claude
            response = self.query_claude(prompt)
        
        # Parse the JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            # Parse JSON
            suggestions = json.loads(json_str)
            return suggestions
        except Exception as e:
            print(f"Error parsing suggestions: {e}")
            print(f"Raw response: {response}")
            # Return a simple dict with the raw response
            return {"raw_response": response, "error": str(e)}
    
    def is_available(self, provider: Optional[str] = None) -> bool:
        """
        Check if a specific provider is available.
        
        Args:
            provider: The provider to check ("openai" or "claude")
            
        Returns:
            True if the provider is available, False otherwise
        """
        provider = provider or self.default_provider
        
        if provider == "openai":
            return self.openai_available
        elif provider == "claude":
            return self.claude_available
        else:
            return False 