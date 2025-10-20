"""
Ensemble Scorer
Combines multiple scoring strategies for robust verification
"""

from typing import List, Dict, Optional, Callable
import numpy as np


class EnsembleScorer:
    """Ensemble multiple scorers with weighted combination."""

    def __init__(
        self,
        scorers: List[Dict[str, any]],
        aggregation: str = "weighted_average",  # "weighted_average", "majority_vote", "max"
    ):
        """
        Initialize ensemble scorer.

        Args:
            scorers: List of scorer configs, each containing:
                - 'scorer': Scorer function
                - 'weight': Weight for this scorer (default: 1.0)
                - 'name': Optional name for logging
            aggregation: How to combine scores
        """
        self.scorers = scorers
        self.aggregation = aggregation

        # Normalize weights
        total_weight = sum(s.get("weight", 1.0) for s in scorers)
        for scorer in self.scorers:
            scorer["weight"] = scorer.get("weight", 1.0) / total_weight

    def score(
        self,
        answer: str,
        image_path: Optional[str] = None,
        image_embedding: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Score using ensemble of scorers.

        Args:
            answer: Answer to score
            image_path: Path to image
            image_embedding: Pre-computed image embedding
            **kwargs: Additional arguments for scorers

        Returns:
            Aggregated score
        """
        scores = []
        weights = []

        for scorer_config in self.scorers:
            scorer_func = scorer_config["scorer"]
            weight = scorer_config["weight"]

            try:
                # Call scorer with appropriate arguments
                if "image_embedding" in scorer_func.__code__.co_varnames:
                    score = scorer_func(answer, image_embedding=image_embedding, **kwargs)
                elif "image_path" in scorer_func.__code__.co_varnames:
                    score = scorer_func(answer, image_path=image_path, **kwargs)
                else:
                    score = scorer_func(answer, **kwargs)

                scores.append(score)
                weights.append(weight)
            except Exception as e:
                print(f"Warning: Scorer {scorer_config.get('name', 'unknown')} failed: {e}")
                continue

        if not scores:
            return 0.0

        # Aggregate scores
        if self.aggregation == "weighted_average":
            return np.average(scores, weights=weights)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "majority_vote":
            # Convert to binary (> 0.5 means valid)
            votes = [1 if s > 0.5 else 0 for s in scores]
            weighted_votes = np.average(votes, weights=weights)
            return weighted_votes
        else:
            return np.mean(scores)


class AdaptiveEnsembleScorer(EnsembleScorer):
    """
    Adaptive ensemble that adjusts weights based on performance.
    """

    def __init__(
        self,
        scorers: List[Dict[str, any]],
        aggregation: str = "weighted_average",
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize adaptive ensemble scorer.

        Args:
            scorers: List of scorer configs
            aggregation: How to combine scores
            adaptation_rate: How fast to adapt weights (0-1)
        """
        super().__init__(scorers, aggregation)
        self.adaptation_rate = adaptation_rate
        self.performance_history = {i: [] for i in range(len(scorers))}

    def update_weights(self, scorer_accuracies: List[float]):
        """
        Update scorer weights based on recent performance.

        Args:
            scorer_accuracies: Accuracy for each scorer
        """
        # Update performance history
        for i, acc in enumerate(scorer_accuracies):
            self.performance_history[i].append(acc)

            # Keep only recent history
            if len(self.performance_history[i]) > 10:
                self.performance_history[i] = self.performance_history[i][-10:]

        # Compute new weights based on average performance
        new_weights = []
        for i in range(len(self.scorers)):
            if self.performance_history[i]:
                avg_perf = np.mean(self.performance_history[i])
            else:
                avg_perf = 0.5
            new_weights.append(avg_perf)

        # Normalize weights
        total_weight = sum(new_weights)
        if total_weight > 0:
            for i, scorer in enumerate(self.scorers):
                # Smooth update with adaptation rate
                old_weight = scorer["weight"]
                new_weight = new_weights[i] / total_weight
                scorer["weight"] = (
                    old_weight * (1 - self.adaptation_rate) + new_weight * self.adaptation_rate
                )


def create_clip_blip2_ensemble(
    clip_scorer_func,
    blip2_scorer_func,
    clip_weight: float = 0.6,
    blip2_weight: float = 0.4,
) -> EnsembleScorer:
    """
    Create ensemble combining CLIP and BLIP2 scorers.

    Args:
        clip_scorer_func: CLIP scoring function
        blip2_scorer_func: BLIP2 scoring function
        clip_weight: Weight for CLIP scorer
        blip2_weight: Weight for BLIP2 scorer

    Returns:
        EnsembleScorer instance
    """
    scorers = [
        {"scorer": clip_scorer_func, "weight": clip_weight, "name": "CLIP"},
        {"scorer": blip2_scorer_func, "weight": blip2_weight, "name": "BLIP2"},
    ]

    return EnsembleScorer(scorers, aggregation="weighted_average")


def score(
    answer: str, scorers: List[Callable], weights: Optional[List[float]] = None, **kwargs
) -> float:
    """
    Convenience function for ensemble scoring.

    Args:
        answer: Answer to score
        scorers: List of scorer functions
        weights: Optional weights for each scorer
        **kwargs: Additional arguments

    Returns:
        Aggregated score
    """
    if weights is None:
        weights = [1.0] * len(scorers)

    scorer_configs = [
        {"scorer": scorer, "weight": weight} for scorer, weight in zip(scorers, weights)
    ]

    ensemble = EnsembleScorer(scorer_configs)
    return ensemble.score(answer, **kwargs)
