import unittest

from backend.text_normalizer import normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_hindi_input(self) -> None:
        text = "पीएम किसान योजना क्या है"
        normalized = normalize_text(text)
        self.assertEqual(normalized, "pm kisan yojana kya hai")

    def test_hinglish_variants(self) -> None:
        text = "pm kishan yojnaa kya hai"
        normalized = normalize_text(text)
        self.assertEqual(normalized, "pm kisan yojana kya hai")

    def test_noisy_input(self) -> None:
        text = "  PM!!! kisan?? yojanaaa   "
        normalized = normalize_text(text)
        self.assertEqual(normalized, "pm kisan yojana")

    def test_mixed_input(self) -> None:
        text = "मुझे आवेदन कैसे करना है"
        normalized = normalize_text(text)
        self.assertEqual(normalized, "mujhe apply how karna hai")

    def test_none_and_empty_are_safe(self) -> None:
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(""), "")

    def test_anti_bias_generic_text(self) -> None:
        text = "kisan yojana batao"
        normalized = normalize_text(text)
        self.assertEqual(normalized, "kisan yojana batao")


if __name__ == "__main__":
    unittest.main()
