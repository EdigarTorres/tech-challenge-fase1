# Testes automatizados para validação do pipeline, persistência do modelo e classe BreastCancerPredictor.

import os
import sys
import unittest

src_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
sys.path.insert(0, src_dir)

from pipeline import (  # noqa: E402
    MODELS_DIR,
    BreastCancerPredictor,
    load_and_prepare_data,
    split_dataset,
    train_and_evaluate_all,
)


class TestTechChallengePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(MODELS_DIR, "modelo_final.joblib")
        if not os.path.exists(model_path):
            train_and_evaluate_all()

        cls.df = load_and_prepare_data()
        cls.predictor = BreastCancerPredictor()

    def test_dataset_dimensions_and_target(self):
        """Verifica dimensões do dataset e a codificação do alvo."""
        self.assertEqual(len(self.df), 569)
        self.assertIn("target", self.df.columns)
        self.assertEqual(int((self.df["target"] == 1).sum()), 212)
        self.assertEqual(int((self.df["target"] == 0).sum()), 357)

    def test_stratified_split_no_overlap(self):
        """Verifica as proporções 70/15/15 e a ausência de sobreposição."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(self.df)

        self.assertEqual(len(X_train), 398)
        self.assertEqual(len(X_val), 85)
        self.assertEqual(len(X_test), 86)

        self.assertEqual(len(set(X_train.index) & set(X_val.index)), 0)
        self.assertEqual(len(set(X_train.index) & set(X_test.index)), 0)
        self.assertEqual(len(set(X_val.index) & set(X_test.index)), 0)

    def test_predictor_inference(self):
        """Verifica diagnósticos e probabilidades retornados pelo predictor."""
        sample = self.df.head(5)
        predictions = self.predictor.predict(sample)

        self.assertEqual(len(predictions), 5)

        for prediction in predictions:
            self.assertIn(
                prediction["diagnostico_previsto"],
                ["Benigno", "Maligno"],
            )
            self.assertGreaterEqual(
                prediction["probabilidade_maligno"],
                0.0,
            )
            self.assertLessEqual(
                prediction["probabilidade_maligno"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()