"""Narration building blocks for the synthetic generator.

Each class has *seen* templates/handles/entities (used in train, val and test)
and *holdout* ones that only ever appear in test, so the test set measures
generalisation to new phrasings and new payees rather than memorisation.

Placeholders inside remark templates:
  {app}  a class-specific app / brand word     {n}  a person's first name
  {a}    a round amount                         {m}  a month abbreviation
  {d}    a short number (order id, task no, ...)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassSpec:
    prefix: str
    seen_templates: list[str]
    holdout_templates: list[str]
    seen_handles: list[str]
    holdout_handles: list[str]
    seen_entities: list[str]
    holdout_entities: list[str]
    apps: list[str]
    amount_range: tuple[int, int]
    extra: dict = field(default_factory=dict)


SPECS: dict[str, ClassSpec] = {
    "BENIGN": ClassSpec(
        prefix="BEN",
        seen_templates=[
            "rent {m}", "house rent {m} month", "grocery", "milk bill {m}", "swiggy order {d}",
            "zomato order", "electricity bill {m}", "mobile recharge", "dth recharge", "petrol",
            "salary {m}", "sal adv {m}", "school fees {m}", "tuition fee {n}", "emi {m} loan a/c {d}",
            "home loan emi {m}", "sip {app} mutual fund", "mf purchase {app}", "zerodha fund add",
            "dream11 winnings refund", "fantasy contest refund {d}", "ipo application {app}",
            "lic premium {m}", "health insurance premium", "gift for {n}", "bday gift {n}",
            "paise wapas {n}", "dinner split", "trip share goa", "maid salary {m}", "car service",
            "kirana store", "medicine pharmacy", "doctor fees", "kyc updated at branch thanks",
            "refund for returned item", "courier charges {d}", "freelance invoice {d}",
            "society maintenance {m}", "gym membership", "netflix", "water bill", "gas cylinder",
            "credit card bill", "loan repayment to {n}", "chit fund {m}", "temple donation",
            "ppf deposit", "fd booking", "nps contribution",
        ],
        holdout_templates=[
            "broadband bill {m}", "tiffin service {m}", "pet clinic", "book store", "parking fee",
            "fastag recharge", "coaching class {n}", "wedding shagun {n}", "bajaj emi {d} {m}",
            "groww sip {m}", "upstox add funds", "my11circle refund {d}",
        ],
        seen_handles=[
            "swiggy@icici", "zomato@hdfcbank", "bescom@axisbank", "airtel@paytm", "jio@sbi",
            "zerodha.broking@hdfcbank", "lic.india@sbi", "rahul.sharma@okaxis", "priya.k@oksbi",
            "amit1990@ybl", "neha.verma@okhdfcbank", "bigbasket@ybl", "dream11@icici",
            "hdfcltd.emi@hdfcbank", "school.dps@kotak", "apollo.pharmacy@axisbank",
            "maint.society@sbi", "sunil.k@paytm", "kiranastore.3421@ybl", "irctc@sbi",
        ],
        holdout_handles=[
            "actfibernet@icici", "upstox@hdfcbank", "my11circle@ybl", "groww.mf@axisbank",
            "bajajfinserv.emi@hdfcbank", "fastag.netc@icici", "vikram.rao@okicici",
            "tiffinwala@paytm", "anjali.m@oksbi",
        ],
        seen_entities=[
            "RAHUL SHARMA", "PRIYA K", "ACME TECH PVT LTD", "HDFC LTD", "ZERODHA BROKING LTD",
            "LIC OF INDIA", "DPS SCHOOL", "SUNRISE APARTMENTS SOC", "AMIT KUMAR", "NEHA VERMA",
        ],
        holdout_entities=[
            "BAJAJ FINANCE LTD", "UPSTOX SECURITIES", "VIKRAM RAO", "GROWW INVEST TECH",
        ],
        apps=["axis", "hdfc", "icici pru", "sbi", "nippon", "parag parikh", "kotak"],
        amount_range=(50, 60000),
    ),
    "GAMBLING_BETTING": ClassSpec(
        prefix="GAM",
        seen_templates=[
            "{app} deposit", "{app} id recharge", "bet {d} {app}", "cricket bet ipl {d}",
            "teenpatti chips", "rummy cash add", "satta {d}", "aviator game add money",
            "{app} wallet topup", "matka result {d}", "casino coins", "id lena hai {app}",
            "betting id {app} {d}", "online game add bal", "lucky draw ticket {d}",
            "jeeto paisa {app}", "color prediction game", "ipl satta bhai",
        ],
        holdout_templates=[
            "andar bahar chips {d}", "{app} exch id", "live casino deposit", "khaiwal payment {d}",
            "roulette credits",
        ],
        seen_handles=[
            "betking777@ybl", "rummywin@paytm", "luckyexch@okaxis", "satta.king@ybl",
            "aviator.pay@axl", "gamezone99@ibl", "royalbet@upi", "teenpatti.gold@paytm",
            "winbig24@ybl", "jackpot.pay@okhdfcbank", "idbook.online@ybl",
        ],
        holdout_handles=[
            "kingexch365@axl", "lotus.book@ibl", "khaiwal.raju@ybl", "casino.royale@upi",
        ],
        seen_entities=[
            "BETKING ENTERPRISES", "LUCKY GAMING SOLUTIONS", "ROYAL PLAY TECH", "WINBIG TRADERS",
        ],
        holdout_entities=["LOTUS BOOK SERVICES", "KINGEXCH TRADING"],
        apps=["1xbet", "parimatch", "mahadev", "betway", "fairplay", "reddy anna", "11exch"],
        amount_range=(100, 25000),
    ),
    "CRYPTO_P2P": ClassSpec(
        prefix="CRY",
        seen_templates=[
            "usdt buy", "usdt {a} p2p", "p2p order {d}", "{app} p2p", "btc purchase",
            "crypto buy {app}", "usdt sell order {d}", "p2p merchant {d}", "tether purchase",
            "binance p2p usdt", "coin purchase", "usdt bhejo", "trc20 usdt", "eth buy p2p",
            "crypto trade {d}", "usdt rate {d}",
        ],
        holdout_templates=[
            "bep20 usdt {d}", "{app} merchant order", "stable coin buy", "usdc p2p {d}",
            "doge purchase p2p",
        ],
        seen_handles=[
            "usdt.trader@ybl", "p2pmerchant@paytm", "cryptokaran@okaxis", "coinbazar@ybl",
            "btc.desk@ibl", "usdtindia@axl", "binancep2p.seller@ybl", "tetherhub@upi",
        ],
        holdout_handles=["usdc.otc@okicici", "bep20.deals@axl", "coinswap.in@ybl"],
        seen_entities=["COINBAZAR TRADERS", "DIGITAL ASSET HUB", "KARAN CRYPTO DESK"],
        holdout_entities=["OTC DESK SOLUTIONS", "SWAP ASSETS LLP"],
        apps=["binance", "bybit", "okx", "wazirx", "kucoin", "bitget"],
        amount_range=(1000, 200000),
    ),
    "LOAN_APP": ClassSpec(
        prefix="LON",
        seen_templates=[
            "loan repay {app}", "{app} instant loan", "processing fee loan", "loan app fee",
            "7 day loan repay", "cash advance repay {d}", "quick loan {app}", "file charge loan",
            "overdue penalty {app}", "loan extension fee", "instant cash app", "kredit repay {d}",
            "loan approve fee", "late fine loan {d}", "insurance fee for loan",
        ],
        holdout_templates=[
            "{app} rollover charge", "gst fee loan sanction", "loan disbursal charge {d}",
            "personal loan pre fee", "app loan clear {d}",
        ],
        seen_handles=[
            "rupeeflash@ybl", "cashkaro.loan@paytm", "quickrupee@axl", "loanfast24@ybl",
            "kreditfast@ibl", "smartcoin.pay@upi", "easycash@okaxis", "minuteloan@ybl",
        ],
        holdout_handles=["rollover.cash@ibl", "pocketrupee@axl", "loanzip.in@ybl"],
        seen_entities=["RUPEEFLASH FINTECH", "QUICKRUPEE SOLUTIONS", "EASYCASH CREDITS"],
        holdout_entities=["POCKETRUPEE TECH", "LOANZIP SERVICES"],
        apps=["rupeeflash", "cashmama", "quickrupee", "loanfast", "pocketcash", "goldbowl"],
        amount_range=(500, 15000),
    ),
    "IMPERSONATION_SCAM": ClassSpec(
        prefix="IMP",
        seen_templates=[
            "kyc update fee", "kyc pending sbi", "account block hoga kyc", "refund processing fee",
            "courier customs charge", "fedex parcel clearance", "electricity kyc verify",
            "sim kyc update", "pan link fee", "refund verify {d}", "customer care refund {d}",
            "parcel held customs {d}", "paytm kyc renewal", "bank officer verification",
            "aadhaar update charge", "cashback claim {d}",
        ],
        holdout_templates=[
            "dhl duty payment {d}", "income tax refund verify", "credit card reward claim",
            "police verification fee", "trai sim block fee",
        ],
        seen_handles=[
            "sbi.kyc.update@ybl", "refund.desk@paytm", "courier.help@axl", "kyc.verify24@ybl",
            "customercare.refund@ibl", "fedex.clear@upi", "paytm.kyc.team@ybl", "bank.helpdesk@okaxis",
        ],
        holdout_handles=["dhl.duty@axl", "itrefund.dept@ybl", "rewards.claim@ibl"],
        seen_entities=["KYC SERVICES", "REFUND HELP DESK", "CUSTOMS CLEARANCE AGENT"],
        holdout_entities=["TAX REFUND CELL", "REWARD POINTS DESK"],
        apps=["sbi", "hdfc", "paytm", "amazon", "flipkart", "fedex"],
        amount_range=(10, 50000),
    ),
    "MULE_TASK_JOB": ClassSpec(
        prefix="MUL",
        seen_templates=[
            "task {d} commission", "youtube like task", "prepaid task {d}", "telegram task fee",
            "part time job reg", "task completed pay", "rating task {app}", "merchant task {d}",
            "work from home deposit", "commission withdraw {d}", "hotel review task",
            "ghar baithe kamao", "daily task income", "vip task upgrade", "job joining fee",
        ],
        holdout_templates=[
            "google map review task {d}", "data entry security dep", "lvl {d} task unlock",
            "insta follow task", "parcel forwarding job",
        ],
        seen_handles=[
            "taskpay88@ybl", "parttime.hr@paytm", "ytlike.task@axl", "commission.desk@ybl",
            "wfh.jobs@ibl", "telegram.task@upi", "earnmoney24@okaxis", "vip.task@ybl",
        ],
        holdout_handles=["mapreview.pay@axl", "dataentry.hr@ibl", "instafollow.task@ybl"],
        seen_entities=["DIGITAL TASK SOLUTIONS", "STAR HR SERVICES", "QUICK EARN ENTERPRISE"],
        holdout_entities=["REVIEW WORKS LLP", "DATAENTRY HR CONSULT"],
        apps=["amazon", "meesho", "youtube", "tripadvisor", "booking", "google"],
        amount_range=(200, 30000),
    ),
}

# Drift-only material: a scam family and handle format that never appear in
# train/val/test ("digital arrest" / disconnection threats via a new PSP suffix).
DRIFT_TEMPLATES: list[str] = [
    "digital arrest clearance {d}", "cbi case settlement", "bijli connection katega aaj",
    "power disconnection notice fee", "narcotics parcel penalty", "ed notice compliance {d}",
    "video kyc arrest warrant", "customs cyber cell fine {d}",
]
DRIFT_HANDLES: list[str] = [
    "cybercell.nodal@superyes", "bijli.vibhag@navipay", "cbi.settle@superyes",
    "ed.compliance@navipay", "powergrid.notice@superyes",
]

# Personal-looking handles that mules and scammers also receive money on, so a
# payee handle alone is not a giveaway (seen vs holdout, like everything else).
PERSONAL_HANDLES_SEEN: list[str] = [
    "rohit.k99@okaxis", "pooja.s@ybl", "deepak1987@paytm", "kavya.r@oksbi", "arjun.m@ibl",
]
PERSONAL_HANDLES_HOLDOUT: list[str] = ["meera.j@okicici", "ravi.t2020@axl"]

NAMES = ["rahul", "priya", "amit", "neha", "sunil", "pooja", "vikas", "anita", "rohit", "kavya",
         "arjun", "meera", "deepak", "sneha", "ravi"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
BANK_CODES = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB", "YESB"]

ABBREVIATIONS = {
    "payment": "pymt", "transfer": "trf", "refund": "rfnd", "deposit": "dep", "amount": "amt",
    "account": "ac", "please": "pls", "recharge": "rchg", "commission": "comm", "processing": "proc",
    "verification": "verif", "charges": "chgs", "charge": "chg", "salary": "sal", "purchase": "purch",
}
