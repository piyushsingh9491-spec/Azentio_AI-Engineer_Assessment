"""Label set shared by every module (single source of truth)."""

LABELS: list[str] = [
    "BENIGN",
    "GAMBLING_BETTING",
    "CRYPTO_P2P",
    "LOAN_APP",
    "IMPERSONATION_SCAM",
    "MULE_TASK_JOB",
]
RISKY_LABELS: list[str] = LABELS[1:]
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
BENIGN_ID: int = LABEL_TO_ID["BENIGN"]
CHANNELS: tuple[str, ...] = ("UPI", "NEFT", "IMPS")
MAX_NARRATION_LEN: int = 200
