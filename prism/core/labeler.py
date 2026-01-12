import os
from collections import Counter

from dotenv import load_dotenv
from groq import Groq
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception_type

from prism.core.base import SubprocessLabeler, Subprocess


def _normalize_label(label: str | None, fallback: str) -> str:
    cleaned = (label or "").strip().strip("\"'")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return fallback
    if not any(ch.isalnum() for ch in cleaned):
        return fallback
    return cleaned


class LabelingError(RuntimeError): ...


class LLMLabeler(SubprocessLabeler):
    """Generate subprocess labels using an LLM via Groq API."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        max_attempts: int = 6,
        delay_seconds: float = 2.0,
    ):
        """
        Initialize the LLM labeler.

        Args:
            model: The model to use (e.g., "openai/gpt-oss-120b").
            api_key: Groq API key. If None, uses GROQ_API_KEY from env/.env file.
            max_attempts: How many times to retry transient failures.
            delay_seconds: Delay between attempts.
        """
        self.model = model
        self.api_key = api_key
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            load_dotenv()
            api_key = (self.api_key or os.getenv("GROQ_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set (and no api_key was provided). "
                    "Set GROQ_API_KEY in your environment or .env to use LLMLabeler."
                )
            self._client = Groq(api_key=api_key)
        return self._client

    def _domain_hint(self, context: dict | None) -> str:
        if not context:
            return ""
        domain = context.get("domain")
        if not domain:
            return ""
        domain_clean = str(domain).replace("\n", " ").strip()
        return f"\nDomain: {domain_clean}"

    def _is_valid_label(self, label: str) -> bool:
        cleaned = " ".join(label.strip().split())
        if not cleaned:
            return False
        if len(cleaned.split()) < 2 or len(cleaned.split()) > 4:
            return False
        if any(
            ch in cleaned
            for ch in [
                "\n",
                "\r",
                "\t",
                ":",
                ";",
                '"',
                "'",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
            ]
        ):
            return False
        lowered = cleaned.lower()
        banned = ["process", "workflow", "activities", "activity", "cluster", "group"]
        if any(word in lowered.split() for word in banned):
            return False
        return True

    def _call_llm(
        self, system_prompt: str, user_prompt: str, max_completion_tokens: int
    ) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                # NOTE: For openai/gpt-oss-* on Groq, providing a system message can lead
                # to empty `content`. Using a single user message is more reliable.
                messages=[
                    {
                        "role": "user",
                        "content": f"{system_prompt}\n\n{user_prompt}",
                    }
                ],
                reasoning_format="hidden",
                max_completion_tokens=max_completion_tokens,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LabelingError("Empty response from LLM")
            return content
        except LabelingError:
            raise
        except Exception as exc:
            raise LabelingError(f"LLM call failed: {exc}") from exc

    def _retrying(self) -> Retrying:
        return Retrying(
            reraise=True,
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_fixed(self.delay_seconds),
            retry=retry_if_exception_type(LabelingError),
        )

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

        if len(nodes) == 1:
            return nodes[0]

        activities_str = ", ".join(sorted(nodes))

        domain_hint = self._domain_hint(context)

        system_prompt = (
            "You are an expert in process mining. "
            "Your job is to output a SHORT VERB PHRASE label for a cluster of activities. "
            "You must output ONLY the label, nothing else."
        )

        user_prompt = (
            "Task: Name this cluster of process activities.\n"
            "Output MUST be an imperative verb phrase (start with a verb).\n\n"
            f"Activities: {activities_str}{domain_hint}\n\n"
            "Good examples (verb phrases):\n"
            "- Approve Invoice\n"
            "- Request Quotation\n"
            "- Create Purchase Order\n"
            "- Pay Invoice\n\n"
            "Bad examples (do NOT output):\n"
            "- Invoice\n"
            "- Purchasing\n"
            "- Process Activities\n"
            "- Process RFQ and Quotations\n"
            "- Cluster 1\n\n"
            "Rules:\n"
            "- Output exactly ONE line.\n"
            "- 2 to 4 words.\n"
            "- Start with a verb.\n"
            "- No quotes. No punctuation. No numbering.\n"
            "- Forbidden words: Process, Workflow, Activities, Activity, Cluster, Group.\n"
            "- Output ONLY the label text."
        )

        for attempt in self._retrying():
            with attempt:
                raw = self._call_llm(
                    system_prompt, user_prompt, max_completion_tokens=200
                )
                cleaned = _normalize_label(raw, fallback="")
                if not self._is_valid_label(cleaned):
                    preview = raw.replace("\n", " ").replace("\r", " ")
                    preview = " ".join(preview.split())
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    raise LabelingError(f"Invalid label: {cleaned!r} (raw={preview!r})")
                return cleaned

        raise LabelingError("Labeling failed")

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
        to_label = [sp for sp in subprocesses if len(sp.nodes) > 1]

        if not to_label:
            return {sp.id: list(sp.nodes)[0] for sp in subprocesses}

        groups_desc = []
        for i, sp in enumerate(to_label):
            activities = ", ".join(sorted(sp.nodes))
            groups_desc.append(f"Group {i + 1}: {activities}")

        groups_str = "\n".join(groups_desc)

        domain_hint = self._domain_hint(context)

        system_prompt = (
            "You are an expert in process mining. "
            "For each group you must output a SHORT VERB PHRASE label. "
            "Output must contain ONLY the labels, one per line, nothing else."
        )

        user_prompt = (
            "Task: Name EACH group.\n"
            "Return EXACTLY one label per line, in the SAME ORDER as the groups.\n"
            "Each label MUST be an imperative verb phrase (start with a verb).\n\n"
            f"Groups:\n{groups_str}{domain_hint}\n\n"
            "Example format:\n"
            "Input groups:\n"
            "Group 1: Submit Request, Approve Request\n"
            "Group 2: Create Invoice, Send Invoice, Pay Invoice\n\n"
            "Output labels:\n"
            "Approve Request\n"
            "Pay Invoice\n\n"
            "Rules:\n"
            "- Each label: 2 to 4 words.\n"
            "- Start with a verb.\n"
            "- No quotes. No punctuation. No numbering.\n"
            "- Forbidden words: Process, Workflow, Activities, Activity, Cluster, Group.\n"
            "- Bad example (do NOT output): Process RFQ and Quotations\n"
            "- Output ONLY the labels."
        )

        for attempt in self._retrying():
            with attempt:
                content = self._call_llm(
                    system_prompt, user_prompt, max_completion_tokens=512
                )
                lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]
                if len(lines) < len(to_label):
                    raise LabelingError(
                        f"Batch labeling returned {len(lines)} lines for {len(to_label)} groups"
                    )

                labels: dict[str, str] = {}
                for i, sp in enumerate(to_label):
                    parsed = lines[i].strip().strip("\"'")
                    if ":" in parsed and parsed.lower().startswith("group"):
                        parsed = parsed.split(":", 1)[1].strip().strip("\"'")

                    cleaned = _normalize_label(parsed, fallback="")
                    if not self._is_valid_label(cleaned):
                        preview = parsed.replace("\n", " ").replace("\r", " ")
                        preview = " ".join(preview.split())
                        if len(preview) > 200:
                            preview = preview[:200] + "..."
                        raise LabelingError(
                            f"Invalid batch label for {sp.id}: {cleaned!r} (raw={preview!r})"
                        )
                    labels[sp.id] = cleaned

                for sp in subprocesses:
                    if len(sp.nodes) == 1:
                        labels[sp.id] = list(sp.nodes)[0]

                return labels

        raise LabelingError("Batch labeling failed")


class SimpleLabeler(SubprocessLabeler):
    """Simple labeler that uses common prefix/suffix or activity count."""

    def label(self, subprocess: Subprocess, context: dict | None = None) -> str:
        """Generate a simple label based on activity names."""
        nodes = list(subprocess.nodes)

        if len(nodes) == 1:
            return nodes[0]

        common_prefix = self._find_common_prefix(nodes)
        if common_prefix and len(common_prefix) > 3:
            return f"{common_prefix}..."

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
        all_words: list[str] = []
        for s in strings:
            words = s.replace("_", " ").replace("-", " ").split()
            all_words.extend(w.lower() for w in words if len(w) > 2)

        word_counts = Counter(all_words)
        threshold = len(strings) // 2
        common = [w for w, c in word_counts.most_common() if c >= threshold]
        return common
