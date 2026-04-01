import unittest

from backend.services import ml_intent_wrapper


class _StubModel:
    def __init__(self, payload):
        self.payload = payload

    def predict(self, text: str):
        return self.payload


class _FailingModel:
    def predict(self, text: str):
        raise RuntimeError("model failure")


class _NoSchemeModel:
    def predict(self, text: str):
        return {
            "intent": "scheme_query",
            "scheme_name": None,
            "entities": {},
            "confidence": 0.9,
            "response_template": "Here is the information",
        }


class _NoSchemeGenericModel:
    def predict(self, text: str):
        return {
            "intent": "general_query",
            "scheme_name": None,
            "entities": {},
            "confidence": 0.81,
            "response_template": None,
        }


class IntentWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_model = ml_intent_wrapper.ml_model

    def tearDown(self) -> None:
        ml_intent_wrapper.ml_model = self._original_model

    def test_strong_scheme_match(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "PM Awas Yojana",
                "entities": {},
                "confidence": 0.91,
                "response_template": "template",
            }
        )
        result = ml_intent_wrapper.get_intent("pm awas yojana kya hai")
        self.assertIsNotNone(result)
        self.assertEqual(result["scheme_name"], "PM Awas Yojana")

    def test_generic_query_removes_scheme(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "PM Kisan Samman Nidhi",
                "entities": {},
                "confidence": 0.93,
                "response_template": None,
            }
        )
        result = ml_intent_wrapper.get_intent("kisan yojana batao")
        self.assertIsNotNone(result)
        self.assertIsNone(result["scheme_name"])

    def test_weak_query_removes_scheme(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "Any Yojana",
                "entities": {},
                "confidence": 0.83,
                "response_template": None,
            }
        )
        result = ml_intent_wrapper.get_intent("yojana kya hai")
        self.assertIsNotNone(result)
        self.assertIsNone(result["scheme_name"])

    def test_low_confidence_removes_scheme(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "PM Awas Yojana",
                "entities": {},
                "confidence": 0.4,
                "response_template": None,
            }
        )
        result = ml_intent_wrapper.get_intent("pm awas yojana kya hai")
        self.assertIsNotNone(result)
        self.assertIsNone(result["scheme_name"])

    def test_hindi_generic_query_removes_scheme(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "PM Awas Yojana",
                "entities": {},
                "confidence": 0.86,
                "response_template": None,
            }
        )
        result = ml_intent_wrapper.get_intent("मुझे योजना बताओ")
        self.assertIsNotNone(result)
        self.assertIsNone(result["scheme_name"])

    def test_valid_non_pm_scheme_kept(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "Ujjwala Yojana",
                "entities": {},
                "confidence": 0.88,
                "response_template": None,
            }
        )
        result = ml_intent_wrapper.get_intent("ujjwala yojana kya hai")
        self.assertIsNotNone(result)
        self.assertEqual(result["scheme_name"], "Ujjwala Yojana")

    def test_none_or_empty_returns_none(self) -> None:
        self.assertIsNone(ml_intent_wrapper.get_intent(None))
        self.assertIsNone(ml_intent_wrapper.get_intent(""))


class FallbackIntentTests(unittest.TestCase):
    def test_apply_help_detection(self) -> None:
        result = ml_intent_wrapper.fallback_intent("kaise apply kare")
        self.assertEqual(result["intent"], "apply_help")
        self.assertIsNone(result["scheme_name"])
        self.assertEqual(result["source"], "fallback")

    def test_eligibility_detection(self) -> None:
        result = ml_intent_wrapper.fallback_intent("kya main eligible hu")
        self.assertEqual(result["intent"], "eligibility_check")
        self.assertIsNone(result["scheme_name"])

    def test_scheme_search_detection(self) -> None:
        result = ml_intent_wrapper.fallback_intent("mujhe yojana batao")
        self.assertEqual(result["intent"], "scheme_search")
        self.assertIsNone(result["scheme_name"])

    def test_general_query_detection(self) -> None:
        result = ml_intent_wrapper.fallback_intent("random text")
        self.assertEqual(result["intent"], "general_query")
        self.assertIsNone(result["scheme_name"])

    def test_hindi_detection(self) -> None:
        result = ml_intent_wrapper.fallback_intent("मुझे योजना बताओ")
        self.assertEqual(result["intent"], "scheme_search")
        self.assertIsNone(result["scheme_name"])

    def test_anti_bias_scheme_name_always_none(self) -> None:
        result = ml_intent_wrapper.fallback_intent("kisan yojana batao")
        self.assertEqual(result["intent"], "scheme_search")
        self.assertIsNone(result["scheme_name"])

    def test_none_or_empty_safe_general_query(self) -> None:
        none_result = ml_intent_wrapper.fallback_intent(None)
        empty_result = ml_intent_wrapper.fallback_intent("")
        self.assertEqual(none_result["intent"], "general_query")
        self.assertEqual(empty_result["intent"], "general_query")
        self.assertIsNone(none_result["scheme_name"])
        self.assertIsNone(empty_result["scheme_name"])


class ProcessUserQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_model = ml_intent_wrapper.ml_model
        self._original_cleaned_loader = ml_intent_wrapper._get_cached_cleaned_scheme_dataset
        self._original_original_loader = ml_intent_wrapper._get_cached_original_scheme_dataset

    def tearDown(self) -> None:
        ml_intent_wrapper.ml_model = self._original_model
        ml_intent_wrapper._get_cached_cleaned_scheme_dataset = self._original_cleaned_loader
        ml_intent_wrapper._get_cached_original_scheme_dataset = self._original_original_loader

    def test_valid_scheme_query_returns_scheme_info(self) -> None:
        ml_intent_wrapper.ml_model = _StubModel(
            {
                "intent": "scheme_query",
                "scheme_name": "PM Awas Yojana",
                "entities": {},
                "confidence": 0.92,
                "response_template": "Awas details",
            }
        )
        result = ml_intent_wrapper.process_user_query("pm awas yojana kya hai")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "scheme_info")
        self.assertEqual(result["data"]["scheme"], "PM Awas Yojana")

    def test_generic_query_returns_scheme_search(self) -> None:
        ml_intent_wrapper.ml_model = _FailingModel()
        result = ml_intent_wrapper.process_user_query("mujhe yojana batao")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "scheme_search")

    def test_eligibility_query_returns_eligibility(self) -> None:
        ml_intent_wrapper.ml_model = _FailingModel()
        result = ml_intent_wrapper.process_user_query("kya main eligible hu")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "eligibility")

    def test_apply_query_returns_application_help(self) -> None:
        ml_intent_wrapper.ml_model = _FailingModel()
        result = ml_intent_wrapper.process_user_query("kaise apply kare")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "application_help")

    def test_random_text_returns_general(self) -> None:
        ml_intent_wrapper.ml_model = _FailingModel()
        result = ml_intent_wrapper.process_user_query("random text")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "general")

    def test_empty_input_returns_error(self) -> None:
        result = ml_intent_wrapper.process_user_query("   ")
        self.assertFalse(result["success"])
        self.assertEqual(result["type"], "error")

    def test_ml_failure_uses_fallback(self) -> None:
        ml_intent_wrapper.ml_model = _FailingModel()
        result = ml_intent_wrapper.process_user_query("kya main eligible hu")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "eligibility")

    def test_dataset_resolver_integration_when_ml_has_no_scheme(self) -> None:
        ml_intent_wrapper.ml_model = _NoSchemeModel()
        ml_intent_wrapper._get_cached_cleaned_scheme_dataset = lambda: [
            {"scheme_name": "Ujjwala Yojana", "keywords": ["ujjwala", "gas"]}
        ]
        ml_intent_wrapper._get_cached_original_scheme_dataset = lambda: []
        result = ml_intent_wrapper.process_user_query("ujjwala yojana kya hai")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "scheme_info")
        self.assertEqual(result["data"].get("scheme"), "Ujjwala Yojana")

    def test_dataset_resolver_runs_when_ml_returns_generic_intent(self) -> None:
        ml_intent_wrapper.ml_model = _NoSchemeGenericModel()
        ml_intent_wrapper._get_cached_cleaned_scheme_dataset = lambda: [
            {"scheme_name": "Ujjwala Yojana", "keywords": ["ujjwala", "gas"]}
        ]
        ml_intent_wrapper._get_cached_original_scheme_dataset = lambda: []
        result = ml_intent_wrapper.process_user_query("ujjwala yojana kya hai")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "scheme_info")
        self.assertEqual(result["data"].get("scheme"), "Ujjwala Yojana")

    def test_two_stage_resolver_falls_back_to_original_dataset(self) -> None:
        ml_intent_wrapper.ml_model = _NoSchemeGenericModel()
        ml_intent_wrapper._get_cached_cleaned_scheme_dataset = lambda: []
        ml_intent_wrapper._get_cached_original_scheme_dataset = lambda: [
            {"scheme_name": "Ujjwala Yojana", "keywords": ["ujjwala", "gas"]}
        ]
        result = ml_intent_wrapper.process_user_query("ujjwala yojana kya hai")
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "scheme_info")
        self.assertEqual(result["data"].get("scheme"), "Ujjwala Yojana")


class SchemeResolverTests(unittest.TestCase):
    def test_detects_ujjwala_scheme(self) -> None:
        dataset = [
            {"scheme_name": "Ujjwala Yojana", "keywords": ["ujjwala", "gas", "lpg"]},
            {"scheme_name": "PM Awas Yojana", "keywords": ["awas", "housing"]},
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("ujjwala yojana kya hai", dataset)
        self.assertEqual(result, "Ujjwala Yojana")

    def test_detects_solar_rooftop_scheme(self) -> None:
        dataset = [
            {
                "scheme_name": "Chief Minister's Solar Rooftop Capital Incentive Scheme",
                "keywords": ["solar", "rooftop", "capital incentive"],
            }
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("chief minister solar rooftop scheme kya hai", dataset)
        self.assertEqual(result, "Chief Minister's Solar Rooftop Capital Incentive Scheme")

    def test_detects_long_scheme_name(self) -> None:
        dataset = [
            {
                "scheme_name": "Prime Minister's Special Scholarship Scheme for the Students of Union Territories of Jammu & Kashmir and Ladakh",
                "keywords": ["special scholarship", "jammu", "ladakh"],
            }
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset(
            "prime minister special scholarship for jammu and ladakh students",
            dataset,
        )
        self.assertEqual(
            result,
            "Prime Minister's Special Scholarship Scheme for the Students of Union Territories of Jammu & Kashmir and Ladakh",
        )

    def test_detects_punctuation_heavy_query(self) -> None:
        dataset = [
            {
                "scheme_name": "Chief Minister's Solar Rooftop Capital Incentive Scheme",
                "keywords": ["solar", "rooftop", "capital incentive"],
            }
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset(
            "chief minister solar rooftop scheme kya hai???",
            dataset,
        )
        self.assertEqual(result, "Chief Minister's Solar Rooftop Capital Incentive Scheme")

    def test_detects_typo_query_with_similarity(self) -> None:
        dataset = [
            {
                "scheme_name": "Chief Minister's Solar Rooftop Capital Incentive Scheme",
                "keywords": ["solar", "rooftop", "capital incentive"],
            }
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset(
            "chief minister solr rooftp capitel incentve scheme",
            dataset,
        )
        self.assertEqual(result, "Chief Minister's Solar Rooftop Capital Incentive Scheme")

    def test_generic_query_returns_none(self) -> None:
        dataset = [
            {"scheme_name": "Ujjwala Yojana", "keywords": ["ujjwala", "gas"]},
            {"scheme_name": "PM Awas Yojana", "keywords": ["awas", "housing"]},
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("yojana kya hai", dataset)
        self.assertIsNone(result)

    def test_equal_score_tie_returns_none(self) -> None:
        dataset = [
            {"scheme_name": "Scheme Alpha", "keywords": ["shared"]},
            {"scheme_name": "Scheme Beta", "keywords": ["shared"]},
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("shared benefit", dataset)
        self.assertIsNone(result)

    def test_margin_rule_returns_none_for_close_candidates(self) -> None:
        dataset = [
            {"scheme_name": "Alpha Rooftop Scheme", "keywords": ["alpha", "rooftop", "incentive"]},
            {"scheme_name": "Beta Rooftop Scheme", "keywords": ["beta", "rooftop", "incentive"]},
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("rooftop incentive scheme kya hai", dataset)
        self.assertIsNone(result)

    def test_rarity_weighting_prefers_specific_token_match(self) -> None:
        dataset = [
            {"scheme_name": "Shared Token Program", "keywords": ["shared", "benefit"]},
            {"scheme_name": "Unique Disablement Pension", "keywords": ["disablement", "pension"]},
        ]
        result = ml_intent_wrapper.resolve_scheme_from_dataset("unique disablement pension scheme", dataset)
        self.assertEqual(result, "Unique Disablement Pension")


if __name__ == "__main__":
    unittest.main()
