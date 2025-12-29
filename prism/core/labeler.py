"""LLM-based subprocess labeling."""

import os
from prism.core.base import SubprocessLabeler, Subprocess


class LLMLabeler(SubprocessLabeler):
    """Generate subprocess labels using an LLM via Groq API."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
    ):
        """
        Initialize the LLM labeler.

        Args:
            model: The model to use (e.g., "llama-3.1-8b-instant", "llama-3.3-70b-versatile").
            api_key: Groq API key. If None, uses GROQ_API_KEY from env/.env file.
        """
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import Groq
            from dotenv import load_dotenv

            load_dotenv()
            api_key = self.api_key or os.getenv("GROQ_API_KEY")
            self._client = Groq(api_key=api_key)
        return self._client

    def label(self, subprocess: Subprocess, context: dict | None = None) -> str:
        """
        Generate a label for a subprocess using an LLM.

        Args:
            subprocess: The subprocess to label.
            context: Optional context (e.g., {"domain": "healthcare"}).

        Returns:
            A concise, meaningful name for the subprocess.
        """
        nodes = list(subprocess.nodes)

        # For singletons, just return the node name
        if len(nodes) == 1:
            return nodes[0]

        # Build prompt
        activities_str = ", ".join(sorted(nodes))

        domain_hint = ""
        if context and "domain" in context:
            domain_hint = f"\nDomain context: {context['domain']}"

        prompt = f"""You are a process mining expert. Given a group of business process activities, 
generate a short, descriptive name (2-4 words) that captures their common purpose.

Activities in this group: {activities_str}{domain_hint}

Rules:
- The name should be concise (2-4 words max)
- Use business terminology appropriate for the activities
- The name should describe WHAT this group of activities accomplishes together
- Do not use generic names like "Process" or "Activities"

Respond with ONLY the name, nothing else."""

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content is None:
                return f"Group ({len(nodes)} activities)"
            name = content.strip()
            # Clean up: remove quotes if present
            name = name.strip('"\'')
            return name
        except Exception as e:
            # Fallback to simple naming
            print(f"LLM labeling failed: {e}")
            return f"Group ({len(nodes)} activities)"

    def label_batch(
        self, subprocesses: list[Subprocess], context: dict | None = None
    ) -> dict[str, str]:
        """
        Generate labels for multiple subprocesses efficiently.

        Args:
            subprocesses: List of subprocesses to label.
            context: Optional context for all subprocesses.

        Returns:
            Dict mapping subprocess ID to generated label.
        """
        # Filter out singletons (they keep their original name)
        to_label = [sp for sp in subprocesses if len(sp.nodes) > 1]

        if not to_label:
            return {sp.id: list(sp.nodes)[0] for sp in subprocesses}

        # Build batch prompt
        groups_desc = []
        for i, sp in enumerate(to_label):
            activities = ", ".join(sorted(sp.nodes))
            groups_desc.append(f"Group {i + 1}: {activities}")

        groups_str = "\n".join(groups_desc)

        domain_hint = ""
        if context and "domain" in context:
            domain_hint = f"\nDomain context: {context['domain']}"

        prompt = f"""You are a process mining expert. For each group of business process activities,
generate a short, descriptive name (2-4 words) that captures their common purpose.

{groups_str}{domain_hint}

Rules:
- Each name should be concise (2-4 words max)
- Use business terminology appropriate for the activities
- Names should describe WHAT each group accomplishes together
- Do not use generic names like "Process" or "Activities"
- Each name should be unique

Respond with ONLY the names, one per line, in the same order as the groups.
Format: "Group N: Name" """

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )

            # Parse response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")
            lines = content.strip().split("\n")
            labels = {}

            for i, sp in enumerate(to_label):
                if i < len(lines):
                    line = lines[i]
                    # Extract name after "Group N:"
                    if ":" in line:
                        name = line.split(":", 1)[1].strip().strip('"\'')
                    else:
                        name = line.strip().strip('"\'')
                    labels[sp.id] = name
                else:
                    labels[sp.id] = f"Group ({len(sp.nodes)} activities)"

            # Add singleton labels
            for sp in subprocesses:
                if len(sp.nodes) == 1:
                    labels[sp.id] = list(sp.nodes)[0]

            return labels

        except Exception as e:
            print(f"Batch LLM labeling failed: {e}")
            # Fallback
            return {
                sp.id: (
                    list(sp.nodes)[0]
                    if len(sp.nodes) == 1
                    else f"Group ({len(sp.nodes)} activities)"
                )
                for sp in subprocesses
            }


class SimpleLabeler(SubprocessLabeler):
    """Simple labeler that uses common prefix/suffix or activity count."""

    def label(self, subprocess: Subprocess, context: dict | None = None) -> str:
        """Generate a simple label based on activity names."""
        nodes = list(subprocess.nodes)

        if len(nodes) == 1:
            return nodes[0]

        # Try to find common prefix
        common_prefix = self._find_common_prefix(nodes)
        if common_prefix and len(common_prefix) > 3:
            return f"{common_prefix}..."

        # Try to find common words
        common_words = self._find_common_words(nodes)
        if common_words:
            return " ".join(common_words[:2])

        return f"Cluster ({len(nodes)} activities)"

    def _find_common_prefix(self, strings: list[str]) -> str:
        """Find common prefix among strings."""
        if not strings:
            return ""
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        return prefix.rstrip(" _-")

    def _find_common_words(self, strings: list[str]) -> list[str]:
        """Find words that appear in multiple activity names."""
        from collections import Counter

        all_words: list[str] = []
        for s in strings:
            words = s.replace("_", " ").replace("-", " ").split()
            all_words.extend(w.lower() for w in words if len(w) > 2)

        word_counts = Counter(all_words)
        # Words appearing in at least half the activities
        threshold = len(strings) // 2
        common = [w for w, c in word_counts.most_common() if c >= threshold]
        return common
