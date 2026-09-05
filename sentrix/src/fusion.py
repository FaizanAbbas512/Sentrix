"""
fusion.py
---------
Yeh hai "intelligent decision" layer. Akele har analyzer ek crude signal
deta hai. Fusion unhe milake ek soch-samajh ke threat level banata hai.

Kyun zaroori hai:
  - Sirf "crowd" = thodi baat (LOW)
  - "Fall" akela = serious (HIGH)
  - "Loitering + Abandoned bag ek saath" = score jud ke HIGH ho jata hai
    (matlab koi der se khada tha aur bag chhod gaya -> zyada mashkook)

Toh system ek signal pe overreact nahi karta, aur kai signals milke
zyada confident alert dete hain. Yahi research mein "decision fusion"
kehlata hai.
"""


class ThreatFusion:
    def __init__(self, cfg):
        f = cfg["fusion"]
        self.weights = f["weights"]
        self.med = f["levels"]["medium"]
        self.high = f["levels"]["high"]

    def assess(self, active_signals):
        """
        active_signals: dict jaise
            {"FALL": 1, "FIGHT": 0, "ABANDONED": 2, "LOITERING": 1, "CROWD": 0}
            (har type ke kitne active hain)
        return: (level_str, score, contributing_list)
        """
        score = 0
        contributing = []
        for sig, count in active_signals.items():
            if count > 0:
                w = self.weights.get(sig, 1)
                score += w * min(count, 3)   # cap taake ek type haavi na ho
                contributing.append(sig)

        if score >= self.high:
            level = "HIGH"
        elif score >= self.med:
            level = "MEDIUM"
        elif score > 0:
            level = "LOW"
        else:
            level = "NONE"

        return level, score, contributing
