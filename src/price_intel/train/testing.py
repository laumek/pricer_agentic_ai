import math
from typing import Dict, Optional, Sequence

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}


def color_for(error, truth):
    if error<40 or error/truth < 0.2:
        return "green"
    elif error<80 or error/truth < 0.4:
        return "orange"
    else:
        return "red"


def summarize(guesses: Sequence[Optional[float]], truths: Sequence[float]) -> Dict[str, Optional[float]]:
    """
    Metrics for a list of guesses against the true prices.
    A guess of None is a failure: it counts towards failure_rate but is left out of the
    error metrics, so always read the error metrics together with the failure rate.

    - average_error: mean absolute error in $
    - rmsle: root mean squared log error
    - hit_rate: share of "green" guesses (error < $40 or < 20%)
    - within_20_rate: share of guesses within 20% of the true price
    """
    pairs = [(guess, truth) for guess, truth in zip(guesses, truths) if guess is not None]
    items = len(truths)
    failures = items - len(pairs)
    summary = {
        "items": items,
        "failures": failures,
        "failure_rate": failures / items if items else None,
        "average_error": None,
        "rmsle": None,
        "hit_rate": None,
        "within_20_rate": None,
    }
    if not pairs:
        return summary

    errors = [abs(guess - truth) for guess, truth in pairs]
    sles = [(math.log(truth + 1) - math.log(max(guess, 0) + 1)) ** 2 for guess, truth in pairs]
    summary["average_error"] = sum(errors) / len(pairs)
    summary["rmsle"] = math.sqrt(sum(sles) / len(pairs))
    summary["hit_rate"] = sum(color_for(e, t) == "green" for e, (_, t) in zip(errors, pairs)) / len(pairs)
    summary["within_20_rate"] = sum(e / t <= 0.2 for e, (_, t) in zip(errors, pairs)) / len(pairs)
    return summary


class Tester:

    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = min(size, len(data))
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
        self.failures = 0

    def color_for(self, error, truth):
        return color_for(error, truth)

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        if guess is None:
            self.failures += 1
            print(f"{RED}{i+1}: Guess: FAILED Truth: ${truth:,.2f} Item: {title}{RESET}")
            return
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(max(guess, 0)+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        scored = len(self.guesses)
        if not scored:
            print(f"{self.title}: every prediction failed ({self.failures}/{self.size})")
            return
        average_error = sum(self.errors) / scored
        rmsle = math.sqrt(sum(self.sles) / scored)
        hits = sum(1 for color in self.colors if color=="green")
        title = (
            f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} "
            f"Hits={hits/scored*100:.1f}% Failures={self.failures/self.size*100:.1f}%"
        )
        self.chart(title)

    def run(self):
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()
