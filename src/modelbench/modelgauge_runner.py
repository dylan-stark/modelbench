import dataclasses
from enum import Enum
from typing import Type, Union

from modelgauge.secret_values import InjectSecret
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import TogetherApiKey, TogetherCompletionsSUT, TogetherChatSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    display_name: str
    modelgauge_class: Union[None, Type] = None
    modelgauge_key: str = ""

    def __hash__(self):
        return super().__hash__()


class ModelGaugeSut(SutDescription, Enum):
    ALPACA_7B = "alpaca-7b", "Stanford Alpaca (7B)"
    CHRONOS_HERMES_13B = (
        "chronos-hermes-13b",
        "Chronos Hermes (13B)",
        TogetherChatSUT,
        "Austism/chronos-hermes-13b",
    )
    DEEPSEEK_67B = (
        "deepseek-67b",
        "DeepSeek LLM Chat (67B)",
        TogetherChatSUT,
        "deepseek-ai/deepseek-llm-67b-chat",
    )
    GEMMA_7B = "gemma-7b", "Gemma Instruct (7B)", TogetherChatSUT, "google/gemma-7b-it"
    LLAMA_2_7B = "llama-2-7b-chat", "Meta Llama 2, 7b parameters"
    LLAMA_2_13B = "llama-2-13b-chat", "Meta Llama 2, 13b parameters"
    LLAMA_2_70B = "llama-2-70b-chat", "Meta Llama 2, 70b parameters"
    MISTRAL_7B = "mistral-7b", "Mistral 7B Instruct v0.2", TogetherChatSUT, "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_8X_7B = (
        "mixtral-8x-7b",
        "Mixtral-8x7B Instruct v0.1 (46.7B)",
        TogetherChatSUT,
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    OPENCHAT_3_5 = "openchat-3_5", "OpenChat 3.5 (7B)", TogetherChatSUT, "openchat/openchat-3.5-1210"
    QWEN_72B = "qwen-72b", "Qwen 1.5 (72B)", TogetherChatSUT, "Qwen/Qwen1.5-72B-Chat"
    NOUS = (
        "stripedhyena-nous-7b",
        "Together StripedHyena Nous 7B",
        TogetherChatSUT,
        "togethercomputer/StripedHyena-Nous-7B",
    )
    VICUNA_13B = "vicuna-13b", "LM Sys Vicuna v1.5 (13B)", TogetherChatSUT, "lmsys/vicuna-13b-v1.5"
    WIZARDLM_13B = "wizardlm-13b", "WizardLM v1.2 (13B)", TogetherChatSUT, "WizardLM/WizardLM-13B-V1.2"
    # YI_34B_CHAT = "yi-34b", "01-ai Yi Chat (34B)", TogetherChatSUT, "zero-one-ai/Yi-34B-Chat"


@dataclasses.dataclass
class AnnotatorDescription:
    key: str
    display_name: str
    modelgauge_class: Union[None, Type] = None
    modelgauge_key: str = ""

    def __hash__(self):
        return super().__hash__()


class ModelGaugeAnnotator(AnnotatorDescription, Enum):
    LLAMA_GUARD_1 = "llama-guard-1", "Meta LlamaGuard 1 (7B)", TogetherCompletionsSUT, "Meta-Llama/Llama-Guard-7b"
    LLAMA_GUARD_2 = "llama-guard-2", "Meta LlamaGuard 2 (8B)", TogetherCompletionsSUT, "meta-llama/LlamaGuard-2-8b"


required_secrets = {
    TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
    TogetherChatSUT: (InjectSecret(TogetherApiKey),),
}

for sut in ModelGaugeSut:
    if sut.modelgauge_class is not None and sut.key is not None and sut.key not in dict(SUTS.items()):
        SUTS.register(sut.modelgauge_class, sut.key, sut.modelgauge_key, *required_secrets[sut.modelgauge_class])

for annotator in ModelGaugeAnnotator:
    if annotator.modelgauge_class is not None and annotator.key is not None and annotator.key not in dict(ANNOTATORS.items()):
        ANNOTATORS.register(annotator.modelgauge_class, annotator.key, annotator.modelgauge_key, *required_secrets[annotator.modelgauge_class])
