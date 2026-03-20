"""Central JA4+ classifier pipeline."""

from __future__ import annotations

from pathlib import Path

from current.services.loaders import (
    build_local_database_from_dataset,
    default_dataset_path,
    default_foxio_db_path,
    default_local_db_path,
    default_model_path,
    load_database_records,
)
from current.services.local_matcher import LocalDatabaseMatcher
from current.core.models import ClassificationResult, FoxIOResult, MatchResult, Observation, RandomForestResult
from current.services.foxio_adapter import FoxIOAdapter
from current.services.inference import RandomForestInference
from current.services.random_forest import JA4RandomForestModel
from current.core.decision_engine import DecisionEngine
from current.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class JA4ClassifierPipeline:
    """Orchestrates local DB matching, RF inference and FoxIO support lookups."""

    def __init__(
        self,
        *,
        local_matcher: LocalDatabaseMatcher | None,
        rf_inference: RandomForestInference | None = None,
        foxio_adapter: FoxIOAdapter | None = None,
        decision_engine: DecisionEngine | None = None,
        runtime_paths: dict[str, str | None] | None = None,
    ):
        self.local_matcher = local_matcher
        self.rf_inference = rf_inference or RandomForestInference(None)
        self.foxio_adapter = foxio_adapter or FoxIOAdapter(None)
        self.decision_engine = decision_engine or DecisionEngine()
        self.runtime_paths = runtime_paths or {}

    @classmethod
    def from_paths(
        cls,
        *,
        local_db_path: str | Path | None = None,
        foxio_db_path: str | Path | None = None,
        rf_model_path: str | Path | None = None,
        training_dataset_path: str | Path | None = None,
        auto_train_rf: bool = True,
        auto_build_local_db: bool = False,
    ) -> "JA4ClassifierPipeline":
        """Create a pipeline using repo defaults when explicit paths are omitted."""
        local_path = Path(local_db_path) if local_db_path else default_local_db_path()
        foxio_path = Path(foxio_db_path) if foxio_db_path else default_foxio_db_path()
        dataset_path = Path(training_dataset_path) if training_dataset_path else default_dataset_path()
        model_path = Path(rf_model_path) if rf_model_path else default_model_path()

        if local_path and not local_path.exists() and dataset_path and dataset_path.exists() and auto_build_local_db:
            LOGGER.info("Building local database from dataset at %s", dataset_path)
            build_local_database_from_dataset(dataset_path, local_path)

        local_matcher = None
        if local_path and local_path.exists():
            local_matcher = LocalDatabaseMatcher.from_records(load_database_records(local_path))
            LOGGER.info("Loaded local database from %s", local_path)
        else:
            LOGGER.warning("Local database not available. Pipeline will rely on RF and FoxIO only.")

        model = None
        if model_path.exists():
            try:
                model = JA4RandomForestModel.load(model_path)
            except (ImportError, ModuleNotFoundError) as exc:
                LOGGER.warning("Random Forest model at %s could not be loaded: %s", model_path, exc)
            else:
                if auto_train_rf and dataset_path and dataset_path.exists() and _should_refresh_model(model, model_path, dataset_path):
                    LOGGER.info("Refreshing Random Forest model from %s", dataset_path)
                    model = JA4RandomForestModel.train_from_json(dataset_path)
                    model.save(model_path)
                    LOGGER.info("Saved refreshed Random Forest model to %s", model_path)
                else:
                    LOGGER.info("Loaded Random Forest model from %s", model_path)
        elif auto_train_rf and dataset_path and dataset_path.exists():
            try:
                LOGGER.info("Training Random Forest model from %s", dataset_path)
                model = JA4RandomForestModel.train_from_json(dataset_path)
                model.save(model_path)
                LOGGER.info("Saved Random Forest model to %s", model_path)
            except (ImportError, ModuleNotFoundError) as exc:
                LOGGER.warning("Random Forest training is unavailable: %s", exc)
        else:
            LOGGER.warning("Random Forest model not available.")

        foxio_adapter = FoxIOAdapter.from_json(foxio_path)
        if foxio_path and foxio_path.exists():
            LOGGER.info("Loaded FoxIO reference database from %s", foxio_path)
        else:
            LOGGER.warning("FoxIO reference database not available.")

        return cls(
            local_matcher=local_matcher,
            rf_inference=RandomForestInference(model),
            foxio_adapter=foxio_adapter,
            decision_engine=DecisionEngine(),
            runtime_paths={
                "local_db_path": str(local_path) if local_path else None,
                "foxio_db_path": str(foxio_path) if foxio_path else None,
                "rf_model_path": str(model_path) if model_path else None,
                "training_dataset_path": str(dataset_path) if dataset_path else None,
            },
        )

    def classify_observation(self, observation: Observation) -> ClassificationResult:
        """Classify one observation and return an explainable structured result."""
        local_result = self._run_local_match(observation)
        foxio_result = self.foxio_adapter.lookup(observation)
        rf_result = self._run_rf_if_needed(observation, local_result)

        final_decision = self.decision_engine.decide(
            local_result=local_result,
            rf_result=rf_result,
            foxio_result=foxio_result,
        )
        true_application = observation.true_application
        true_category = observation.true_category
        application_prediction = final_decision.application_prediction
        category_prediction = final_decision.category_prediction
        return ClassificationResult(
            observation_id=observation.observation_id,
            input=observation.to_dict(),
            true_application=true_application,
            true_category=true_category,
            application_prediction=application_prediction,
            category_prediction=category_prediction,
            application_correct=_casefold_match(application_prediction, true_application),
            category_correct=_casefold_match(category_prediction, true_category),
            application_confidence=final_decision.application_confidence,
            category_confidence=final_decision.category_confidence,
            decision_source=final_decision.decision_source,
            reasoning=final_decision.reasoning,
            local_db=local_result.to_dict(),
            random_forest=rf_result.to_dict(),
            foxio=foxio_result.to_dict(),
            final_decision=final_decision.to_dict(),
        )

    def classify_batch(self, observations: list[Observation]) -> list[ClassificationResult]:
        """Classify a list of observations with consistent output formatting."""
        return [self.classify_observation(observation) for observation in observations]

    def _run_local_match(self, observation: Observation) -> MatchResult:
        if self.local_matcher is None:
            return MatchResult(
                status="unavailable",
                match_mode=None,
                evidence=["Local database matcher is not configured."],
                confidence=0.0,
            )
        return self.local_matcher.match_observation(observation)

    def _run_rf_if_needed(
        self,
        observation: Observation,
        local_result: MatchResult,
    ) -> RandomForestResult:
        if local_result.status == "unique":
            return RandomForestResult(
                status="skipped",
                evidence=["Random Forest skipped because a unique local match was already found."],
                confidence_score=0.0,
            )
        return self.rf_inference.predict(observation)


def _casefold_match(predicted: str | None, truth: str | None) -> bool | None:
    if truth is None:
        return None
    if predicted is None:
        return False
    return predicted.casefold() == truth.casefold()


def _should_refresh_model(
    model: JA4RandomForestModel,
    model_path: Path,
    dataset_path: Path,
) -> bool:
    if dataset_path.stat().st_mtime > model_path.stat().st_mtime:
        return True
    return not any(category for category in model.bundle.label_to_category.values())
