class Weight:
    def __init__(self, initial_value=None):
        self.value = initial_value or 0

class ProbabilityDistribution:
    def __init__(self, number_of_classes:int):
        self.probs = [1/number_of_classes for _ in range(number_of_classes)]

    def normalize(self):
        self.probs = [abs(p) for p in self.probs]
        total = sum(self.probs)
        # Divide each number by their total. All values will sum to one. 
        # This assumes they are positive so we must take the absolute value
        self.probs = [self.probs[i]/total for i in range(len(self.probs))]

    def compute_weighted_probabilities(self, weights: list[Weight]):
        n = len(self.probs)

        if len(weights) != n:
            return "Mismatched shape, cannot multiply weights and probabilities"

        for i in range(n):
            self.probs[i] *= weights[i].value

        self.normalize()

weights = [Weight(1), Weight(2), Weight(3)]
p = ProbabilityDistribution(3)
p.compute_weighted_probabilities(weights)
print(p.probs)

# check sanity
print(sum(p.probs))

class WeightedProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, number_of_classes:int, weights: list[Weight]):
        super().__init__(number_of_classes)
        self.weights = weights

    def compute_weighted_probabilities(self):
        return super().compute_weighted_probabilities(self.weights)

weights = [Weight(1), Weight(2), Weight(3)]
wp = WeightedProbabilityDistribution(number_of_classes=3, weights=weights)
wp.compute_weighted_probabilities()
print(wp.probs)


